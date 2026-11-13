/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB 3.4.4 OpenMP (NPB3.4-OMP/MG)
 */
package NPB.SER;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import NPB.Benchmark;
import NPB.common.io.BMArgs;
import NPB.common.io.BMResults;
import NPB.common.numerics.Misc;
import NPB.common.numerics.Random;
import NPB.common.timer.Timer;
import NPB.exceptions.BadClassException;
import NPB.exceptions.BadInputException;
import NPB.exceptions.NPBException;

public class MG implements Benchmark{

	private static final String BM_NAME="MG";
	private char CLASS;

	private boolean detailedTiming;

	private Random rng;
	private Timer timer;
	private BMResults res;

	private final double a = 1220703125; //5^13;
	private final double x = 314159265;

	private final int maxLevel = 11;	//For CLASS D
	private int niter, lm, lt;
	private int nm, nr, nv, lb, m;
	private int is1, is2, is3, ie1, ie2, ie3;
	private int nx[] = new int[maxLevel];
	private int ny[] = new int[maxLevel];
	private int nz[] = new int[maxLevel];
	private int ir[] = new int[maxLevel];
	private int m1[] = new int[maxLevel];
	private int m2[] = new int[maxLevel];
	private int m3[] = new int[maxLevel];
	private int debugVec[] = {0,0,0,0,0,0,0,0};	

	/*
	 * Different timers
	 */
	private final int T_INIT = 1;
	private final int T_BENCH = 2;
	private final int T_MG3P = 3;
	private final int T_PSINV = 4;
	private final int T_RESID = 5;
	private final int T_RESID2 = 6;
	private final int T_RPRJ3 = 7;
	private final int T_INTERP = 8;
	private final int T_NORM2 = 9;

	public static void main(String[] args) throws NPBException {
		MG mg = null;

		BMArgs.parseCmdLineArgs(args,BM_NAME);
		try{
			mg = new MG(BMArgs.CLASS,BMArgs.DETAILED_TIMING);
		}catch(OutOfMemoryError e){
			BMArgs.outOfMemoryMessage();
			throw e;
		}
		mg.runBenchmark(args);
	}

	public BMResults getResults(){
		return res;
	}

	public MG(char classSize, boolean dtiming){

		CLASS = classSize;
		detailedTiming = dtiming;
	}

	public void build(String[] args) throws NPBException{

		int problemSize = 32;
		int ndim1, ndim2, ndim3;

		switch( CLASS )
		{
		case 'S':
			problemSize = 32;
			niter = 4;
			break;
		case 'W':
			problemSize = 128;
			niter = 4;
			break;
		case 'A':
			problemSize = 256;
			niter = 4;
			break;
		case 'B':
			problemSize = 256;
			niter = 20;
			break;
		case 'C':
			problemSize = 512;
			niter = 20;
			break;
		case 'D':
			problemSize = 1024;
			niter = 50;
			break;
		default:
			throw new BadClassException("Specified class (" + CLASS + ") not valid for this benchmark\n"
					+ "Valid classes are S, W, A, B, C and D");
		}

		ndim1 = ndim2 = ndim3 = lm = lt = Misc.ilog2(problemSize);
		nm = 2+(2<<(lm-1));	// In the original Java it is lm-1, but in the original Fortran it is lm...
		m = nm + 1;
		nv = (2+(2<<(ndim1-1)))*(2+(2<<(ndim2-1)))*(2+(2<<(ndim3-1)));
		nr = ((nv+(nm*nm)+5*nm+7*lm+6)/7)*8;
		nx[lt-1] = problemSize;
		ny[lt-1] = problemSize;
		nz[lt-1] = problemSize;

		rng = new Random();
	}

	private void terminate(){
	}

	public void runBenchmark(String[] args) throws NPBException {

		this.build(args);

		timer = new Timer();
		
		int k, n1, n2, n3;
		int nnn[];
		/* The rnm array will contain both rnm2 and rnmu */
		double rnm[] = new double[2];
		double u[] = new double[nr];
		double v[] = new double[nv];
		double r[] = new double[nr];
		double a[] = new double[4];
		double c[] = new double[4];

		timer.start(T_INIT);

		readData();

		a[0] = (double)-8/3;
		a[1] = (double)0;
		a[2] = (double)1/6; 
		a[3] = (double)1/12;

		/* Coefficients for the S(a) smoother */
		if(CLASS == 'A' || CLASS == 'S' || CLASS == 'W'){
			c[0] = (double)-3/8;
			c[1] = (double)1/32;
			c[2] = (double)-1/64;
			c[3] = (double)0;
		}
		/* Coefficients for the S(b) smoother */
		else{
			c[0] = (double)-3/17;
			c[1] = (double)1/33;
			c[2] = (double)-1/61;
			c[3] = (double)0;
		}

		lb = 1;
		k = lt;

		nnn = setup();
		n1 = nnn[0];
		n2 = nnn[1];
		n3 = nnn[2];

		zero3(u,0,n1,n2,n3);
		zran3(v,n1,n2,n3,nx[lt-1],ny[lt-1],k);		
		norm2u3(v,0,n1,n2,n3,rnm,nx[lt-1],ny[lt-1],nz[lt-1]);

		System.out.println(" Size:  "+nx[lt-1]+"x "+ny[lt-1]+"x "+nz[lt-1]+"  (class "+CLASS+")");
		System.out.println(" Iterations:  "+niter);

		resid(u,0,v,0,r,0,n1,n2,n3,a,k-1);
		norm2u3(r,0,n1,n2,n3,rnm,nx[lt-1],ny[lt-1],nz[lt-1]);

		/* One iteration for startup */
		mg3P(u,v,r,a,c,n1,n2,n3,k);
		resid(u,0,v,0,r,0,n1,n2,n3,a,k-1);
		nnn = setup();
		n1 = nnn[0];
		n2 = nnn[1];
		n3 = nnn[2];
		zero3(u,0,n1,n2,n3);
		zran3(v,n1,n2,n3,nx[lt-1],ny[lt-1],k);

		timer.stop(T_INIT);

		System.out.format(java.util.Locale.ENGLISH, "\n Initialization time:\t%8.3f seconds\n\n", timer.read(T_INIT));

		timer.resetAll();
		
		timer.start(T_BENCH);

		if(detailedTiming) timer.start(T_RESID2);
		resid(u,0,v,0,r,0,n1,n2,n3,a,k-1);
		if(detailedTiming) timer.stop(T_RESID2);

		norm2u3(r,0,n1,n2,n3,rnm,nx[lt-1],ny[lt-1],nz[lt-1]);

		for(int it=1; it<=niter; it++){
			if(it == 1 || it == niter || it%5 == 0)
				System.out.println("   iter "+it);

			if(detailedTiming) timer.start(T_MG3P);
			mg3P(u,v,r,a,c,n1,n2,n3,k);
			if(detailedTiming) timer.stop(T_MG3P);
			if(detailedTiming) timer.start(T_RESID2);
			resid(u,0,v,0,r,0,n1,n2,n3,a,k-1);
			if(detailedTiming) timer.stop(T_RESID2);
		}

		norm2u3(r,0,n1,n2,n3,rnm,nx[lt-1],ny[lt-1],nz[lt-1]);

		timer.stop(T_BENCH);

		double times[] = {0,0};
		times[0] = timer.read(T_BENCH);		
		int verified = 0;
		double verifyValue = 0;

		System.out.println();
		System.out.println(" Benchmark completed");

		double epsilon = 1E-8;

		if(CLASS != 'U'){
			if(CLASS == 'S'){
				verifyValue = 0.530770700573E-4;
			}
			else if(CLASS == 'W'){
				verifyValue = 0.646732937534E-5;
			}
			else if(CLASS == 'A'){
				verifyValue = 0.243336530907E-5;
			}
			else if(CLASS == 'B'){
				verifyValue = 0.180056440136E-5;
			}
			else if(CLASS == 'C'){
				verifyValue = 0.570673228574E-6;
			}
			else if(CLASS == 'D'){
				verifyValue = 0.158327506044E-9;
			}

			double err = Math.abs(rnm[0]-verifyValue)/verifyValue;
			if(err <= epsilon){
				verified = 1;
				BMResults.printVerificationStatus(CLASS, verified);
				System.out.println(" L2 Norm is "+rnm[0]);
				System.out.println(" Error is   "+err);
			}
			else{
				verified = 0;
				BMResults.printVerificationStatus(CLASS, verified);
				System.out.println(" L2 Norm is "+rnm[0]);
				System.out.println(" The correct L2 Norm is   "+verifyValue);
			}
		}
		else{
			verified = 0;
			BMResults.printVerificationStatus(CLASS, verified);
		}

		double nn = nx[lt-1]*ny[lt-1]*nz[lt-1];

		double mflops = 0;
		if(times[0] != 0)
			mflops = (nn*1E-6)*(58*niter)/times[0];

		res = new BMResults(BM_NAME,
				CLASS,
				nx[lt-1],
				ny[lt-1],
				nz[lt-1],
				niter,
				times,
				mflops,
				"floating point",
				verified,
				1,
				detailedTiming);

		res.printSER();

		/*  Print additional timers  */
		if (detailedTiming)
			printAdditionalTimers();

		terminate();
	}

	private void printAdditionalTimers()
	{
		double tmax = timer.read(T_BENCH);
		double t, tt;

		if(tmax <= 0.0)
			tmax = 1.0;

		System.out.format(java.util.Locale.ENGLISH, " SECTION\tTime (secs)\n");
		System.out.format(java.util.Locale.ENGLISH, " benchmk :     %8.4f (%5.2f%%)\n", tmax, tmax*100./tmax);
		t = timer.read(T_MG3P);
		System.out.format(java.util.Locale.ENGLISH, " mg3P    :     %8.4f (%5.2f%%)\n", t, t*100./tmax);
		t = timer.read(T_PSINV);
		System.out.format(java.util.Locale.ENGLISH, " psinv   :     %8.4f (%5.2f%%)\n", t, t*100./tmax);
		t = timer.read(T_RESID);
		System.out.format(java.util.Locale.ENGLISH, " resid   :     %8.4f (%5.2f%%)\n", t, t*100./tmax);
		t = timer.read(T_RESID2);
		tt = timer.read(T_RESID);
		System.out.format(java.util.Locale.ENGLISH, " mg-resid:     %8.4f (%5.2f%%)\n", tt-t, (tt-t)*100./tmax);
		t = timer.read(T_RPRJ3);
		System.out.format(java.util.Locale.ENGLISH, " rprj3   :     %8.4f (%5.2f%%)\n", t, t*100./tmax);
		t = timer.read(T_INTERP);
		System.out.format(java.util.Locale.ENGLISH, " interp  :     %8.4f (%5.2f%%)\n", t, t*100./tmax);
		t = timer.read(T_NORM2);
		System.out.format(java.util.Locale.ENGLISH, " norm2   :     %8.4f (%5.2f%%)\n", t, t*100./tmax);	
	}

	/*
	 * Read in and broadcast input data
	 */	
	private void readData() throws NPBException{

		boolean inData = true;

		BMArgs.bannerSER(BM_NAME, CLASS);

		try{
			BufferedReader input = new BufferedReader(new FileReader("mg.input"));

			System.out.println(" Reading from input file mg.input");

			String line[] = input.readLine().split(" ");
			lt = Integer.valueOf(line[0]);

			line = input.readLine().split(" ");
			nx[lt-1] = Integer.valueOf(line[0]);
			ny[lt-1] = Integer.valueOf(line[1]);
			nz[lt-1] = Integer.valueOf(line[2]);

			line = input.readLine().split(" ");
			niter = Integer.valueOf(line[0]);

			line = input.readLine().split(" ");
			for(int i = 0; i<debugVec.length; i++){
				debugVec[i] = Integer.valueOf(line[i]);
			}

			input.close();
		}
		catch(FileNotFoundException e){
			inData = false;
		}
		catch(IOException e){
			throw new BadInputException(" Error reading input file: "+e.getMessage());
		}
		if(!inData){
			System.out.println(" No input file. Using compiled defaults");
		}

		if( nx[lt-1] != ny[lt-1] || nx[lt-1] != nz[lt-1])
			CLASS = 'U';
		else if(nx[lt-1] == 32 && niter == 4)
			CLASS = 'S';
		else if(nx[lt-1] == 128 && niter == 4)
			CLASS = 'W';
		else if(nx[lt-1] == 256 && niter == 4)
			CLASS = 'A';
		else if(nx[lt-1] == 256 && niter == 20)
			CLASS = 'B';
		else if(nx[lt-1] == 512 && niter == 20)
			CLASS = 'C';
		else if(nx[lt-1] == 1024 && niter == 50)
			CLASS = 'D';
		else
			CLASS = 'U';

		/*
		 * Use these for debug info:
		 * debugVec[0] = 1 !=> report all norms
		 * debugVec[1] = 1 !=> some setup information
		 * debugVec[1] = 2 !=> more setup information
		 * debugVec[2] = k => at level k or below, show result of resid
		 * debugVec[3] = k => at level k or below, show result of psinv
		 * debugVec[4] = k => at level k or below, show result of rprj
		 * debugVec[5] = k => at level k or below, show result of interp
		 * debugVec[6] = 1 => (unused)
		 * debugVec[7] = 1 => (unused)
		 */
	}

	private int[] setup(){
		int n1, n2, n3, k, j, j1, ax;	
		int ng[][] = new int[3][maxLevel];
		int mi[][] = new int[3][maxLevel];

		ng[0][lt-1] = nx[lt-1];
		ng[1][lt-1] = ny[lt-1];
		ng[2][lt-1] = nz[lt-1];

		for(ax = 0; ax<3; ax++){
			for(k = lt-2; k>=0 ; k--){
				ng[ax][k] = ng[ax][k+1]>>1;
			}
		}

		for(k = lt-1; k>=0; k--){
			nx[k] = ng[0][k];
			ny[k] = ng[1][k];
			nz[k] = ng[2][k];
		}

		for(k = lt-1; k>=0; k--){
			for(ax = 0; ax < 3; ax++){
				mi[ax][k] = 2 + ng[ax][k];
			}				

			m1[k] = mi[0][k];
			m2[k] = mi[1][k];
			m3[k] = mi[2][k];
		}

		k = lt-1;
		is1 = 2 + ng[0][k] - ng[0][lt-1];
		ie1 = 1 + ng[0][k];
		n1 = 3 + ie1 - is1;
		is2 = 2 + ng[1][k] - ng[1][lt-1]; 		
		ie2 = 1 + ng[1][k];
		n2 = 3 + ie2 - is2;
		is3 = 2 + ng[2][k] - ng[2][lt-1];
		ie3 = 1 + ng[2][k];
		n3 = 3 + ie3 - is3;

		ir[lt-1] = 1;
		for(j = lt-2; j>=0; j--){
			j1 = j+1;
			ir[j] = ir[j1]+m1[j1]*m2[j1]*m3[j1];
		}

		if(debugVec[1] >= 1){
			System.out.println(" in setup, ");
			System.out.println(" k  lt  nx  ny  nz  n1  n2  n3  is1  is2  is3  ie1  ie2  ie3 ");
			System.out.println(" "+k+"  "+lt+"  "+ng[0][k]+"  "+ng[1][k]+"  "+ng[2][k]+"  "+n1
					+"  "+n2+"  "+n3+"  "+is1+"  "+is2+"  "+is3+"  "+ie1+"  "+ie2+"  "+ie3);
		}

		int result[] = {n1,n2,n3};
		return result;
	}

	private void zero3(double z[], int offset, int n1, int n2, int n3){
		int i1, i2, i3, pos;

		for(i3 = 1; i3<=n3; i3++){
			for(i2 = 1; i2<=n2; i2++){
				pos = position3D(1,i2,i3,n1,n2,n3)+offset;
				for(i1 = 1; i1<=n1; i1++){
					z[pos++] = 0;
				}
			}
		}
	}

	/*
	 * This method organizes the memory in the same way that
	 * fortran does, to facilitate the code translation and 
	 * efficiency.
	 */
	private int position3D(int x, int y, int z, int dimx, int dimy, int dimz){
		return x - 1 + (y - 1) * dimx + (z - 1) * dimx * dimy;
	}

	/*
	 * Multigrid V-cycle routine
	 */
	private void mg3P(double u[], double v[], double r[], double a[], double c[], int n1, int n2, int n3, int k){
		/*
		 * Down cycle.
		 * Restrict the residual from the find grid to the coarse
		 */
		int j = lt-2;
		for(k = lt-1; k>=lb; k--){
			rprj3(r,ir[k]-1,m1[k],m2[k],m3[k],r,ir[j]-1,m1[j],m2[j],m3[j],k);
			j--;
		}

		k = lb-1;

		/* Compute an approximate solution on the coarsest grid */
		zero3(u,ir[k]-1,m1[k],m2[k],m3[k]);
		psinv(r,ir[k]-1,u,ir[k]-1,m1[k],m2[k],m3[k],c,k);

		j = lb-1;
		for(k = lb; k<lt-1 ;k++){
			/* Prolongate from level k-1 to k */
			zero3(u,ir[k]-1,m1[k],m2[k],m3[k]);
			interp(u,ir[j]-1,m1[j],m2[j],m3[j],u,ir[k]-1,m1[k],m2[k],m3[k],k);

			/* Compute residual for level k */
			resid(u,ir[k]-1,r,ir[k]-1,r,ir[k]-1,m1[k],m2[k],m3[k],a,k);

			/* Apply smoother */
			psinv(r,ir[k]-1,u,ir[k]-1,m1[k],m2[k],m3[k],c,k);
			j++;
		}

		j = lt-2;
		k = lt-1;

		interp(u,ir[j]-1,m1[j],m2[j],m3[j],u,0,n1,n2,n3,k);
		resid(u,0,v,0,r,0,n1,n2,n3,a,k);
		psinv(r,0,u,0,n1,n2,n3,c,k);		
	}

	/*
	 * psinv applies an approximate inverse as smoother:  u = u + Cr
	 * 
	 * This  implementation costs  15A + 4M per result, where
	 * A and M denote the costs of Addition and Multiplication.  
	 * Presuming coefficient c(3) is zero (the NPB assumes this,
	 * but it is thus not a general case), 2A + 1M may be eliminated,
	 * resulting in 13A + 3M.
	 * Note that this vectorizes, and is also fine for cache 
	 * based machines.
	 */
	private void psinv(double r[], int roffset, double u[], int uoffset, int n1, int n2, int n3, double c[], int k){


		if(detailedTiming) timer.start(T_PSINV);

		final double c0 = c[0], c1 = c[1], c2 = c[2];
		final int n1n2 = n1*n2;

		for(int i3=2; i3<=n3-1; i3++){
			for(int i2=2; i2<=n2-1; i2++){
				/* r1/r2 of the Fortran code (sums over the neighbouring lines) are kept in a sliding window of
				 * locals instead of in line arrays: same values, same operations, same order */
				int pr = position3D(1,i2,i3,n1,n2,n3)+roffset;
				int pu = position3D(2,i2,i3,n1,n2,n3)+uoffset;
				double r1m = r[pr-n1] + r[pr+n1]
				             + r[pr-n1n2] + r[pr+n1n2];
				double rm = r[pr];
				pr++;
				double r10 = r[pr-n1] + r[pr+n1]
				             + r[pr-n1n2] + r[pr+n1n2];
				double r20 = r[pr-n1-n1n2] + r[pr+n1-n1n2]
				             + r[pr-n1+n1n2] + r[pr+n1+n1n2];
				double r0 = r[pr];
				for(int i1=1; i1<n1-1; i1++){
					pr++;
					final double r1p = r[pr-n1] + r[pr+n1]
					                   + r[pr-n1n2] + r[pr+n1n2];
					final double r2p = r[pr-n1-n1n2] + r[pr+n1-n1n2]
					                   + r[pr-n1+n1n2] + r[pr+n1+n1n2];
					final double rp = r[pr];
					/* Fortran: u = u + c0*r + c1*(..) + c2*(..) is evaluated left to right, ((u + c0*r) + c1*(..)) + c2*(..);
					 * `u += c0*r + ..` would add the whole right-hand side to u at the end and round differently.
					 * Assume c[3] = 0 (otherwise add  + c3*(r2[i1-1] + r2[i1+1])) */
					u[pu] = u[pu] + c0*r0
					        + c1*(rm + rp + r10)
					        + c2*(r20 + r1m + r1p);
					rm = r0; r0 = rp;
					r1m = r10; r10 = r1p;
					r20 = r2p;
					pu++;
				}
			}
		}

		/* Exchange boundary points */
		comm3(u,uoffset,n1,n2,n3,k);

		if(detailedTiming) timer.stop(T_PSINV);

		if(debugVec[0]>=1)
			repNrm(u,uoffset,n1,n2,n3,"   psinv ",k);
		if(debugVec[3]-1>=k)
			showAll(u,uoffset,n1,n2,n3);
	}

	/*
	 * resid computes the residual:  r = v - Au
	 *
	 * This  implementation costs  15A + 4M per result, where
	 * A and M denote the costs of Addition (or Subtraction) and 
	 * Multiplication, respectively. 
	 * Presuming coefficient a(1) is zero (the NPB assumes this,
	 * but it is thus not a general case), 3A + 1M may be eliminated,
	 * resulting in 12A + 3M.
	 * Note that this vectorizes, and is also fine for cache 
	 * based machines.
	 */
	private void resid(double u[], int uoffset, double v[], int voffset, double r[], int roffset, int n1, int n2, int n3, double a[], int k){
		int n1n2, pos, i1, i2, i3;

		if(detailedTiming) timer.start(T_RESID);

		n1n2 = n1*n2;
		final double a0 = a[0], a2 = a[2], a3 = a[3];

		for(i3 = 2; i3<=n3-1; i3++){
			for(i2 = 2; i2<=n2-1; i2++){

				/* u1/u2 of the Fortran code (sums over the neighbouring lines) are kept in a sliding window of
				 * locals instead of in line arrays: same values, same operations, same order */
				int pn = position3D(1,i2,i3,n1,n2,n3)+uoffset;
				double u1m = u[pn-n1] + u[pn+n1]
				             +u[pn-n1n2] + u[pn+n1n2];
				double u2m = u[pn-n1-n1n2] + u[pn+n1-n1n2]
				             +u[pn-n1+n1n2] + u[pn+n1+n1n2];
				pn++;
				double u10 = u[pn-n1] + u[pn+n1]
				             +u[pn-n1n2] + u[pn+n1n2];
				double u20 = u[pn-n1-n1n2] + u[pn+n1-n1n2]
				             +u[pn-n1+n1n2] + u[pn+n1+n1n2];
				pos = position3D(2,i2,i3,n1,n2,n3);
				for(i1 = 1; i1<n1-1; i1++){
					pn++;
					final double u1p = u[pn-n1] + u[pn+n1]
					                   +u[pn-n1n2] + u[pn+n1n2];
					final double u2p = u[pn-n1-n1n2] + u[pn+n1-n1n2]
					                   +u[pn-n1+n1n2] + u[pn+n1+n1n2];
					/* Assume a[1] = 0 (otherwise add  - a1*(u[pos-1] + u[pos+1] + u1[i1])) */
					r[pos+roffset] = v[pos+voffset] - a0*u[pos+uoffset]
					                 - a2 * ( u20 + u1m + u1p )
					                 - a3 * ( u2m + u2p );
					u1m = u10; u10 = u1p;
					u2m = u20; u20 = u2p;
					pos++;
				}
			}
		}

		/* Exchange boundary data */
		comm3(r,roffset,n1,n2,n3,k);

		if(detailedTiming) timer.stop(T_RESID);

		if(debugVec[0] >= 1)
			repNrm(r,roffset,n1,n2,n3,"   resid ",k);
		if(debugVec[2] >= k)
			showAll(r,roffset,n1,n2,n3);
	}

	/*
	 * rprj3 projects onto the next coarser grid, 
	 * using a trilinear Finite Element projection:  s = r' = P r
	 * 
	 * This  implementation costs  20A + 4M per result, where
	 * A and M denote the costs of Addition and Multiplication.  
	 * Note that this vectorizes, and is also fine for cache 
	 * based machines.  
	 */
	private void rprj3(double r[], int roffset, int m1k, int m2k, int m3k, double s[], int soffset, int m1j, int m2j, int m3j, int k){

		int d1, d2, d3, j1, i2, j2, i3, j3;
		double x2, y2;

		if(detailedTiming) timer.start(T_RPRJ3);

		if(m1k == 3)
			d1 = 2;
		else
			d1 = 1;

		if(m2k == 3)
			d2 = 2;
		else
			d2 = 1;

		if(m3k == 3)
			d3 = 2;
		else
			d3 = 1;

		final int mk12 = m1k*m2k;

		for(j3 = 2; j3<=m3j-1; j3++){
			i3 = 2*j3-d3;
			for(j2 = 2; j2<=m2j-1; j2++){
				i2 = 2*j2-d2;
				/* x1/y1 of the Fortran code (sums at the odd columns i1-1 and i1+1) are kept in locals instead of
				 * in line arrays: same values, same operations, same order. p walks the fine line (i2,i3) */
				int p = position3D(3-d1,i2,i3,m1k,m2k,m3k)+roffset;
				int ps = position3D(2,j2,j3,m1j,m2j,m3j)+soffset;
				double x1m = r[p-m1k] + r[p+m1k]
				             + r[p-mk12] + r[p+mk12];
				double y1m = r[p-m1k-mk12] + r[p-m1k+mk12]
				             + r[p+m1k-mk12] + r[p+m1k+mk12];
				for(j1 = 2; j1<=m1j-1; j1++){
					p++;
					y2 = r[p-m1k-mk12] + r[p-m1k+mk12]
					     + r[p+m1k-mk12] + r[p+m1k+mk12];
					x2 = r[p-m1k] + r[p+m1k]
					     + r[p-mk12] + r[p+mk12];
					p++;
					final double x1p = r[p-m1k] + r[p+m1k]
					                   + r[p-mk12] + r[p+mk12];
					final double y1p = r[p-m1k-mk12] + r[p-m1k+mk12]
					                   + r[p+m1k-mk12] + r[p+m1k+mk12];
					s[ps] =
						0.5*r[p-1]
						      + 0.25*(r[p-2]+r[p]+x2)
						      + 0.125*(x1m+x1p+y2)
						      + 0.0625*(y1m+y1p);
					x1m = x1p;
					y1m = y1p;
					ps++;
				}
			}
		}

		int j = k-1;

		comm3(s,soffset,m1j,m2j,m3j,j);

		if(detailedTiming) timer.stop(T_RPRJ3);

		if(debugVec[0] >= 1)
			repNrm(s,soffset,m1j,m2j,m3j,"   rprj3 ",k-1);
		if(debugVec[4] >= k)
			showAll(s,soffset,m1j,m2j,m3j);
	}

	/*
	 * interp adds the trilinear interpolation of the correction
	 * from the coarser grid to the current approximation:  u = u + Qu'
	 * 
	 * Observe that this  implementation costs  16A + 4M, where
	 * A and M denote the costs of Addition and Multiplication.  
	 * Note that this vectorizes, and is also fine for cache 
	 * based machines.  Vector machines may get slightly better 
	 * performance however, with 8 separate "do i1" loops, rather than 4.
	 */
	private void interp(double z[], int zoffset, int mm1, int mm2, int mm3, double u[], int uoffset, int n1, int n2, int n3, int k){

		int d1,d2,d3,t1,t2,t3,i1,i2,i3;

		if(detailedTiming) timer.start(T_INTERP);

		if(n1 != 3 && n2 != 3 && n3 != 3){
			final int mm12 = mm1*mm2, n12 = n1*n2;
			for(i3 = 1; i3<=mm3-1; i3++){
				for(i2 = 1; i2<=mm2-1; i2++){
					/* z1/z2/z3 of the Fortran code are kept in locals (values at i1 and i1+1) instead of in line
					 * arrays: same values, same operations, same order. z (coarse) and u (fine) never overlap */
					int pz = position3D(1,i2,i3,mm1,mm2,mm3)+zoffset;
					int pu = position3D(1,2*i2-1,2*i3-1,n1,n2,n3)+uoffset;
					double z0 = z[pz];
					double z10 = z[pz+mm1] + z0;
					double z20 = z[pz+mm12] + z0;
					double z30 = z[pz+mm1+mm12] + z[pz+mm12] + z10;
					for(i1 = 1; i1<=mm1-1; i1++){
						pz++;
						final double zp = z[pz];
						final double z1p = z[pz+mm1] + zp;
						final double z2p = z[pz+mm12] + zp;
						final double z3p = z[pz+mm1+mm12] + z[pz+mm12] + z1p;
						u[pu] += z0;
						u[pu+1] += 0.5*(zp+z0);
						u[pu+n1] += 0.5*z10;
						u[pu+n1+1] += 0.25*(z10 + z1p);
						u[pu+n12] += 0.5*z20;
						u[pu+n12+1] += 0.25*(z20 + z2p);
						u[pu+n1+n12] += 0.25*z30;
						u[pu+n1+n12+1] += 0.125*(z30 + z3p);
						z0 = zp; z10 = z1p; z20 = z2p; z30 = z3p;
						pu += 2;
					}
				}
			}
		}
		else{
			if(n1 == 3){
				d1 = 2;
				t1 = 1;
			}
			else{
				d1 = 1;
				t1 = 0;
			}
			if(n2 == 3){
				d2 = 2;
				t2 = 1;
			}
			else{
				d2 = 1;
				t2 = 0;
			}
			if(n3 == 3){
				d3 = 2;
				t3 = 1;
			}
			else{
				d3 = 1;
				t3 = 0;
			}

			for(i3=d3; i3<=mm3-1;i3++){
				for(i2=d2; i2<=mm2-1;i2++){
					for(i1=d1; i1<=mm1-1;i1++){
						u[position3D(2*i1-d1,2*i2-d2,2*i3-d3,n1,n2,n3)+uoffset] += z[position3D(i1,i2,i3,mm1,mm2,mm3)+zoffset];
					}
					for(i1=1; i1<=mm1-1;i1++){
						u[position3D(2*i1-t1,2*i2-d2,2*i3-d3,n1,n2,n3)+uoffset] += 0.5*(z[position3D(i1+1,i2,i3,mm1,mm2,mm3)+zoffset]+z[position3D(i1,i2,i3,mm1,mm2,mm3)+zoffset]);
					}
				}
				for(i2=1; i2<=mm2-1;i2++){
					for(i1=d1; i1<=mm1-1;i1++){
						u[position3D(2*i1-d1,2*i2-t2,2*i3-d3,n1,n2,n3)+uoffset] += 0.5*(z[position3D(i1,i2+1,i3,mm1,mm2,mm3)+zoffset]+z[position3D(i1,i2,i3,mm1,mm2,mm3)+zoffset]);
					}
					for(i1=1; i1<=mm1-1;i1++){
						u[position3D(2*i1-t1,2*i2-t2,2*i3-d3,n1,n2,n3)+uoffset] += 0.25*(z[position3D(i1+1,i2+1,i3,mm1,mm2,mm3)+zoffset]+z[position3D(i1+1,i2,i3,mm1,mm2,mm3)+zoffset]
						                                                                                                                   +z[position3D(i1,i2+1,i3,mm1,mm2,mm3)+zoffset]+z[position3D(i1,i2,i3,mm1,mm2,mm3)+zoffset]);
					}
				}
			}
			for(i3=1; i3<=mm3-1;i3++){
				for(i2=d2; i2<=mm2-1;i2++){
					for(i1=d1; i1<=mm1-1;i1++){
						u[position3D(2*i1-d1,2*i2-d2,2*i3-t3,n1,n2,n3)+uoffset] += 0.5*(z[position3D(i1,i2,i3+1,mm1,mm2,mm3)+zoffset]+z[position3D(i1,i2,i3,mm1,mm2,mm3)+zoffset]);
					}
					for(i1=1; i1<=mm1-1;i1++){
						u[position3D(2*i1-t1,2*i2-d2,2*i3-t3,n1,n2,n3)+uoffset] += 0.25*(z[position3D(i1+1,i2,i3+1,mm1,mm2,mm3)+zoffset]+z[position3D(i1,i2,i3+1,mm1,mm2,mm3)+zoffset]
						                                                                                                                   +z[position3D(i1+1,i2,i3,mm1,mm2,mm3)+zoffset]+z[position3D(i1,i2,i3,mm1,mm2,mm3)+zoffset]);
					}
				}
				for(i2=1; i2<=mm2-1;i2++){
					for(i1=d1; i1<=mm1-1;i1++){
						u[position3D(2*i1-d1,2*i2-t2,2*i3-t3,n1,n2,n3)+uoffset] += 0.25*(z[position3D(i1,i2+1,i3+1,mm1,mm2,mm3)+zoffset]+z[position3D(i1,i2,i3+1,mm1,mm2,mm3)+zoffset]
						                                                                                                                   +z[position3D(i1,i2+1,i3,mm1,mm2,mm3)+zoffset]+z[position3D(i1,i2,i3,mm1,mm2,mm3)+zoffset]);
					}
					for(i1=1; i1<=mm1-1;i1++){
						u[position3D(2*i1-t1,2*i2-t2,2*i3-t3,n1,n2,n3)+uoffset] += 0.125*(z[position3D(i1+1,i2+1,i3+1,mm1,mm2,mm3)+zoffset]+z[position3D(i1+1,i2,i3+1,mm1,mm2,mm3)+zoffset]
						                                                                                                                      +z[position3D(i1,i2+1,i3+1,mm1,mm2,mm3)+zoffset]+z[position3D(i1,i2,i3+1,mm1,mm2,mm3)+zoffset]
						                                                                                                                                                                         +z[position3D(i1+1,i2+1,i3,mm1,mm2,mm3)+zoffset]+z[position3D(i1+1,i2,i3,mm1,mm2,mm3)+zoffset]
						                                                                                                                                                                                                                            +z[position3D(i1,i2+1,i3,mm1,mm2,mm3)+zoffset]+z[position3D(i1,i2,i3,mm1,mm2,mm3)+zoffset]);
					}
				}
			}	
		}

		if(detailedTiming) timer.stop(T_INTERP);

		if(debugVec[0] >= 1){
			repNrm(z,zoffset,mm1,mm2,mm3,"z: inter ",k-1);
			repNrm(u,uoffset,n1,n2,n3,"u: inter ",k);
		}

		if(debugVec[5] >= k){
			showAll(z,zoffset,mm1,mm2,mm3);
			showAll(u,uoffset,n1,n2,n3);			
		}
	}

	/*
	 * norm2u3 evaluates approximations to the L2 norm and the
	 * uniform (or L-infinity or Chebyshev) norm, under the
	 * assumption that the boundaries are periodic or zero.  Add the
	 * boundaries in with half weight (quarter weight on the edges
	 * and eighth weight at the corners) for inhomogeneous boundaries.
	 */
	private void norm2u3(double r[],int offset, int n1, int n2, int n3, double rnm[], int nx, int ny, int nz){
		double dn = 1.0*nx*ny*nz;
		double s = 0;
		double a;
		rnm[1] = 0;
		int i1,i2,i3,pos;

		if(detailedTiming) timer.start(T_NORM2);

		for(i3 = 2; i3<=n3-1; i3++){
			for(i2 = 2; i2<=n2-1; i2++){
				pos = position3D(2,i2,i3,n1,n2,n3)+offset;
				for(i1 = 2; i1<=n1-1; i1++){
					s += r[pos]*r[pos];	
					a = Math.abs(r[pos]);
					if(a > rnm[1])
						rnm[1]=a;
					pos++;
				}
			}
		}

		rnm[0] = Math.sqrt(s / dn);

		if(detailedTiming) timer.stop(T_NORM2);
	}

	/*
	 * Report on norm
	 */
	private void repNrm(double u[], int offset, int n1, int n2, int n3, String title, int kk){

		double rnm[] = new double[2];
		norm2u3(u,offset,n1,n2,n3,rnm,nx[kk],ny[kk],nz[kk]);
		System.out.println(" Level "+kk+" in "+title+": "+rnm[0]+" "+rnm[1]);
	}

	private void comm3(double u[], int uoffset, int n1, int n2, int n3, int kk){

		int i1,i2,i3, pos11, pos12, pos21, pos22;

		for(i3 = 2; i3<=n3-1 ;i3++){
			pos11 = position3D(1,2,i3,n1,n2,n3)+uoffset;
			pos12 = position3D(n1-1,2,i3,n1,n2,n3)+uoffset;
			pos21 = position3D(n1,2,i3,n1,n2,n3)+uoffset;
			pos22 = position3D(2,2,i3,n1,n2,n3)+uoffset;

			for(i2 = 2; i2<=n2-1 ;i2++){
				u[pos11] = u[pos12];
				u[pos21] = u[pos22];
				pos11 += n1;
				pos12 += n1;
				pos21 += n1;
				pos22 += n1;
			}
		}

		for(i3 = 2; i3<=n3-1 ;i3++){
			pos11 = position3D(1,1,i3,n1,n2,n3)+uoffset;
			pos12 = position3D(1,n2-1,i3,n1,n2,n3)+uoffset;;
			pos21 = position3D(1,n2,i3,n1,n2,n3)+uoffset;
			pos22 = position3D(1,2,i3,n1,n2,n3)+uoffset;

			for(i1 = 1; i1<=n1 ;i1++){
				u[pos11] = u[pos12];
				u[pos21] = u[pos22];
				pos11++;
				pos12++;
				pos21++;
				pos22++;
			}
		}

		for(i2 = 1; i2<=n2; i2++){
			pos11 = position3D(1,i2,1,n1,n2,n3)+uoffset;
			pos12 = position3D(1,i2,n3-1,n1,n2,n3)+uoffset;
			pos21 = position3D(1,i2,n3,n1,n2,n3)+uoffset;
			pos22 = position3D(1,i2,2,n1,n2,n3)+uoffset;

			for(i1 = 1; i1<=n1; i1++){
				u[pos11] = u[pos12];
				u[pos21] = u[pos22];
				pos11++;
				pos12++;
				pos21++;
				pos22++;
			}
		}
	}

	/*
	 * zran3  loads +1 at ten randomly chosen points,
	 * loads -1 at a different ten random points,
	 * and zero elsewhere.
	 */
	private void zran3(double z[],int n1, int n2, int n3, int nx, int ny, int k){

		double x1, xx;
		int mm = 10, pos;
		double ten[][] = new double[mm][2];
		int j1[][] = new int[mm][2];
		int j2[][] = new int[mm][2];
		int j3[][] = new int[mm][2];

		double a1 = rng.power(a, nx);
		double a2 = rng.power(a, nx*ny);

		zero3(z,0,n1,n2,n3);

		int i = is1-2+nx*(is2-2+ny*(is3-2));

		double ai = rng.power(a, i);
		int d1 = ie1 - is1 + 1;
		int e2 = ie2 - is2 + 2;
		int e3 = ie3 - is3 + 2;
		double x0 = x;
		double values[] = rng.vrandlc(x0, ai);
		x0 = values[0];

		for(int i3 = 2; i3<=e3; i3++){
			x1 = x0;
			pos = position3D(2,2,i3,n1,n2,n3);
			for(int i2 = 2; i2<=e2; i2++){
				xx = x1;
				rng.vranlc(d1, xx, a, z, pos);
				values = rng.vrandlc(x1, a1);
				x1 = values[0];
				pos+=n1;
			}
			values = rng.vrandlc(x0, a2);
			x0 = values[0];
		}

		/* Each processor looks for twenty candidates */
		for(i = 0; i<mm; i++){
			ten[i][1] = 0;
			j1[i][1] = 0;
			j2[i][1] = 0;
			j3[i][1] = 0;
			ten[i][0] = 1;
			j1[i][0] = 0;
			j2[i][0] = 0;
			j3[i][0] = 0;
		}

		for(int i3 = 2; i3<n3; i3++){
			for(int i2 = 2; i2<n2; i2++){
				pos = position3D(2,i2,i3,n1,n2,n3);
				for(int i1 = 2; i1<n1; i1++){
					if(z[pos] > ten[0][1]){
						ten[0][1] = z[pos];
						j1[0][1] = i1;
						j2[0][1] = i2;
						j3[0][1] = i3;
						bubble(ten, j1, j2, j3, mm, 1);
					}
					if(z[pos] < ten[0][0]){
						ten[0][0] = z[pos];
						j1[0][0] = i1;
						j2[0][0] = i2;
						j3[0][0] = i3;
						bubble(ten, j1, j2, j3, mm, 0);
					}
					pos++;
				}
			}
		}

		/* Now which of these are globally best? */
		int i1 = mm;
		int i0 = mm;
		int i1m = i1-1;
		int i0m = i0-1;
		double best = 0;
		int jg[][][] = new int[4][mm][2];

		for(i = mm-1; i>=0; i--){

			best = 0;

			if(best < ten[i1m][1]){
				jg[0][i][1] = 0;
				jg[1][i][1] = is1-2+j1[i1m][1];
				jg[2][i][1] = is2-2+j2[i1m][1];
				jg[3][i][1] = is3-2+j3[i1m][1];
				i1m--;
				i1--;
			}
			else{
				jg[0][i][1] = 0;
				jg[1][i][1] = 0;
				jg[2][i][1] = 0;
				jg[3][i][1] = 0;
			}

			best = 1;

			if(best >= ten[i0m][0]){
				jg[0][i][0] = 0;
				jg[1][i][0] = is1-2+j1[i0m][0];
				jg[2][i][0] = is2-2+j2[i0m][0];
				jg[3][i][0] = is3-2+j3[i0m][0];
				i0m--;
				i0--;
			}
			else{
				jg[0][i][0] = 0;
				jg[1][i][0] = 0;
				jg[2][i][0] = 0;
				jg[3][i][0] = 0;
			}
		}

		int m1 = 1;
		int m0 = 1;

		for(int i3 = 1; i3<=n3; i3++){
			for(int i2 = 1; i2<=n2; i2++){
				pos = position3D(1,i2,i3,n1,n2,n3);
				for(i1 = 1; i1<=n1; i1++){
					z[pos++] = 0;
				}
			}
		}

		for(i = mm-1; i>=m0-1; i--){
			z[position3D(jg[1][i][0],jg[2][i][0],jg[3][i][0],n1,n2,n3)] = -1;
		}

		for(i = mm-1; i>=m1-1; i--){
			z[position3D(jg[1][i][1],jg[2][i][1],jg[3][i][1],n1,n2,n3)] = 1;
		}
		comm3(z,0,n1,n2,n3,k-1);
	}

	private void showAll(double z[], int offset, int n1, int n2, int n3){
		int m1 = Math.min(n1, 18);
		int m2 = Math.min(n2, 14);
		int m3 = Math.min(n3, 18);

		System.out.println(" ");
		for(int i3 = 1; i3<=m3; i3++){
			for(int i1 = 1; i1<=m1; i1++){
				for(int i2 = 1; i2<=m2; i2++){
					System.out.printf(" %f ",z[position3D(i1,i2,i3,n1,n2,n3)+offset]);
					if(i2%m2==0)
						System.out.printf("\n");
				}
				System.out.printf("\n");
			}
			System.out.println(" - - - - - - - ");
		}
		System.out.println(" ");
	}		

	/*
	 * Does a bubble sort in direction dir
	 */
	private void bubble(double ten[][], int j1[][], int j2[][], int j3[][], int m, int ind){

		double temp;
		int jTemp;

		if(ind == 1){
			for(int i = 0; i<m-1; i++){
				if(ten[i][ind] > ten[i+1][ind]){
					temp = ten[i+1][ind];
					ten[i+1][ind] = ten[i][ind];
					ten[i][ind] = temp;

					jTemp = j1[i+1][ind];
					j1[i+1][ind] = j1[i][ind];
					j1[i][ind] = jTemp;

					jTemp = j2[i+1][ind];
					j2[i+1][ind] = j2[i][ind];
					j2[i][ind] = jTemp;

					jTemp = j3[i+1][ind];
					j3[i+1][ind] = j3[i][ind];
					j3[i][ind] = jTemp;
				}
				else{
					return;
				}
			}
		}
		else{
			for(int i = 0; i<m-1; i++){
				if(ten[i][ind] < ten[i+1][ind]){
					temp = ten[i+1][ind];
					ten[i+1][ind] = ten[i][ind];
					ten[i][ind] = temp;

					jTemp = j1[i+1][ind];
					j1[i+1][ind] = j1[i][ind];
					j1[i][ind] = jTemp;

					jTemp = j2[i+1][ind];
					j2[i+1][ind] = j2[i][ind];
					j2[i][ind] = jTemp;

					jTemp = j3[i+1][ind];
					j3[i+1][ind] = j3[i][ind];
					j3[i][ind] = jTemp;
				}
				else{
					return;
				}
			}
		}
	}	
}
