/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB-MPJ by Damián Álvarez Mallón
 */
package NPB.MPJ;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import mpi.Intracomm;
import mpi.MPI;
import mpi.Request;

import NPB.Benchmark;
import NPB.common.io.BMArgs;
import NPB.common.io.BMResults;
import NPB.common.numerics.Misc;
import NPB.common.numerics.Random;
import NPB.common.timer.Timer;
import NPB.exceptions.BadClassException;
import NPB.exceptions.IncorrectNumProcessException;
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
	private int nm, nm2, nr, nv, lb, m;
	private int is1, is2, is3, ie1, ie2, ie3;
	private int nx[] = new int[maxLevel];
	private int ny[] = new int[maxLevel];
	private int nz[] = new int[maxLevel];
	private int ir[] = new int[maxLevel];
	private int m1[] = new int[maxLevel];
	private int m2[] = new int[maxLevel];
	private int m3[] = new int[maxLevel];	
	private int debugVec[] = {0,0,0,0,0,0,0,0};

	private int nbr[][][] = new int[3][3][maxLevel];
	private int msgType[][] = new int[3][3];
	private Request msgId[][][] = new Request[3][3][2];
	private boolean dead[] = new boolean[maxLevel];
	private boolean takeEx[][] = new boolean[3][maxLevel];
	private boolean giveEx[][] = new boolean[3][maxLevel];	
	private double buff[][];

	/*
	 * Different timers
	 */
	private final int T_INIT = 0;
	private final int T_COMP = 1;
	private final int T_COMM = 2;

	/*
	 * MPJ variables
	 */
	private int myRank;
	private int numProcess; /* active processes (a power of two / a square) */
	private int npTotal; /* processes launched */
	private boolean active = true; /* false for the excess processes masked out */
	private Intracomm commWork; /* communicator of the active processes */

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

		MPI.Init(args);

		/* Active processes (NPB-MPI 3.4: get_active_nprocs, NPB_NPROCS_STRICT) */
		ActiveProcs ap;
		try {
			ap = ActiveProcs.select(ActiveProcs.POW2);
		} catch (IncorrectNumProcessException e) {
			terminate();
			throw e;
		}
		myRank = ap.rank;
		npTotal = ap.total;
		numProcess = ap.active;
		commWork = ap.comm;
		active = ap.isActive;
		if (!active)
			return; /* excess process masked out: runBenchmark() finalizes MPI */

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
		case 'E':
			problemSize = 2048;
			niter = 50;
			break;
		default:
			terminate();
			throw new BadClassException("Specified class (" + CLASS + ") not valid for this benchmark\n"
					+ "Valid classes are S, W, A, B, C, D and E");
		}

		int log2NumProcs = Misc.ilog2(numProcess);
		lt = Misc.ilog2(problemSize);		
		lm = lt - log2NumProcs/3;
		ndim1 = lm;
		ndim2 = lt - (log2NumProcs+1)/3;
		ndim3 = lt - (log2NumProcs+2)/3;

		nm = 2+(2<<(lm-1)); // The original Java uses lm-1, but the original Fortran uses lm...
		nm2 = 2*nm*nm;		
		m = nm + 1;
		/*
		 * Per-process array sizes. The official nr = (8*(nv+nm**2+5*nm+14*lt-7*lm))/7 is a
		 * default (32-bit) integer in NPB-MPI, so its 8*(...) overflows for class E with
		 * fewer than 64 processes. Computed in long here; a size that does not fit in a
		 * Java array is reported clearly.
		 */
		long nvLong = (2L+(2L<<(ndim1-1)))*(2L+(2L<<(ndim2-1)))*(2L+(2L<<(ndim3-1)));
		long nrLong = (8L*(nvLong+((long)nm*nm)+5L*nm+14L*lt-7L*lm))/7L;
		if (nrLong > Integer.MAX_VALUE) {
			terminate();
			throw new IncorrectNumProcessException("Class " + CLASS + " needs more processes: the partition of "
					+ nrLong + " points per process is too large for a Java array");
		}
		nv = (int) nvLong;
		nr = (int) nrLong;

		nx[lt-1] = problemSize;
		ny[lt-1] = problemSize;
		nz[lt-1] = problemSize;
		buff = new double[4][nm2];
		rng = new Random();
	}

	private void terminate(){
		MPI.Finalize();
	}

	public void runBenchmark(String[] args) throws NPBException {

		this.build(args);

		/* excess process masked out (NPB_NPROCS_STRICT off): nothing to do */
		if (!active) {
			terminate();
			return;
		}

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

		commWork.Barrier();

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

		if(myRank == 0){
			System.out.println(" Size:  "+nx[lt-1]+"x "+ny[lt-1]+"x "+nz[lt-1]+"  (class "+CLASS+")");
			System.out.println(" Iterations:  "+niter);
			System.out.println(" Total number of processes:  "+npTotal);
			if (numProcess != npTotal)
				System.out.println(" WARNING: Number of processes is not a power of two ("+numProcess+" active)");
		}

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

		if(myRank == 0)
			System.out.format(java.util.Locale.ENGLISH, "\n Initialization time:\t%8.3f seconds\n\n", timer.read(T_INIT));

		commWork.Barrier();

		timer.start(T_COMP);

		resid(u,0,v,0,r,0,n1,n2,n3,a,k-1);
		norm2u3(r,0,n1,n2,n3,rnm,nx[lt-1],ny[lt-1],nz[lt-1]);

		for(int it=1; it<=niter; it++){
			if(myRank == 0){
				if(it == 1 || it == niter || it%5 == 0)
					System.out.println("   iter "+it);
			}
			mg3P(u,v,r,a,c,n1,n2,n3,k);
			resid(u,0,v,0,r,0,n1,n2,n3,a,k-1);
		}

		norm2u3(r,0,n1,n2,n3,rnm,nx[lt-1],ny[lt-1],nz[lt-1]);

		timer.stop(T_COMP);

		double times[] = { timer.read(T_COMP), timer.read(T_COMM) };
		double maxtimes[] = {0,0};

		commWork.Reduce(times, 0, maxtimes, 0, 2, MPI.DOUBLE, MPI.MAX, 0);

		int verified = 0;
		double verifyValue = 0;

		if(myRank == 0){
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
				else if(CLASS == 'E'){
					verifyValue = 0.5630442584711E-10;
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
			if(maxtimes[0] != 0)
				mflops = (nn*1E-6)*(58*niter)/maxtimes[0];

			res = new BMResults(BM_NAME,
					CLASS,
					nx[lt-1],
					ny[lt-1],
					nz[lt-1],
					niter,
					maxtimes,
					mflops,
					"floating point",
					verified,
					npTotal,
					numProcess,
					detailedTiming);

			res.printMPJ();
		}
		terminate();
	}

	/*
	 * Read in and broadcast input data
	 */	
	private void readData(){

		boolean inData = true;

		if(myRank == 0){
			BMArgs.bannerMPJ(BM_NAME, CLASS, npTotal);

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
				System.err.println(" Error reading input file: "+e.getMessage());
				commWork.Abort(-7);
			}
			if(!inData){
				System.out.println(" No input file. Using compiled defaults");
			}
		}

		int buff[] = new int[13];
		int auxBuff[] = {lt,niter,nx[lt-1],ny[lt-1],nz[lt-1]};
		System.arraycopy(auxBuff, 0, buff, 0, 5);
		System.arraycopy(debugVec, 0, buff, 5, 8);

		commWork.Bcast(buff, 0, 13, MPI.INT, 0);

		lt = buff[0];
		niter = buff[1];
		nx[lt-1] = buff[2];
		ny[lt-1] = buff[3];
		nz[lt-1] = buff[4];
		debugVec[0] = buff[5];
		debugVec[1] = buff[6];
		debugVec[2] = buff[7];
		debugVec[3] = buff[8];
		debugVec[4] = buff[9];
		debugVec[5] = buff[10];
		debugVec[6] = buff[11];
		debugVec[7] = buff[12];

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
		int next[] = new int[3];
		int idi[] = new int[3];
		int idin[][] = new int[3][3];
		int pi[] = new int[3];
		int mi[][] = new int[3][maxLevel];
		int mip[][] = new int[3][maxLevel];
		int dx, dy,logP, d;

		for(j = 0; j<3 ;j++){
			for(d = 0; d<3; d++){
				msgType[d][j] = 100*((j-1)+2+10*(d+1));
			}
		}

		ng[0][lt-1] = nx[lt-1];
		ng[1][lt-1] = ny[lt-1];
		ng[2][lt-1] = nz[lt-1];

		for(ax = 0; ax<3; ax++){
			next[ax] = 1;
			for(k = lt-2; k>=0 ; k--){
				ng[ax][k] = ng[ax][k+1]>>1;
			}
		}

		for(k = lt-1; k>=0; k--){
			nx[k] = ng[0][k];
			ny[k] = ng[1][k];
			nz[k] = ng[2][k];
		}

		logP = (int)(Math.log((float)(numProcess+0.0001))/Math.log(2.0));
		dx = logP/3;

		pi[0] = 1<<dx; //(int)Math.pow(2,dx);
		idi[0] = myRank%pi[0];

		dy = (logP-dx)/2;
		pi[1] = 1<<dy; //(int)Math.pow(2,dy);
		idi[1] = (myRank/pi[0])%pi[1];

		pi[2] = numProcess/(pi[0]*pi[1]);
		idi[2] = myRank/(pi[0]*pi[1]);

		for(k = lt-1; k>=0; k--){
			dead[k] = false;
			for(ax = 0; ax < 3; ax++){
				takeEx[ax][k] = false;
				giveEx[ax][k] = false;

				mi[ax][k] = 2 +
				((idi[ax]+1)*ng[ax][k])/pi[ax] -
				((idi[ax])*ng[ax][k])/pi[ax];
				mip[ax][k] = 2 +
				((next[ax]+idi[ax]+1)*ng[ax][k])/pi[ax] -
				((next[ax]+idi[ax])*ng[ax][k])/pi[ax];

				if(mip[ax][k] == 2 || mi[ax][k] == 2)
					next[ax] = next[ax]<<1;

				if(k+2 <= lt){
					if(mip[ax][k] == 2 && mi[ax][k] == 3)
						giveEx[ax][k+1] = true;
					if(mip[ax][k] == 3 && mi[ax][k] == 2)
						takeEx[ax][k+1] = true;
				}
			}

			if(mi[0][k] == 2 || mi[1][k] == 2 || mi[2][k] == 2){
				dead[k] = true;
			}

			m1[k] = mi[0][k];
			m2[k] = mi[1][k];
			m3[k] = mi[2][k];

			for(ax = 0; ax<3; ax++){
				idin[ax][2] = (idi[ax]+next[ax]+pi[ax])%pi[ax];
				idin[ax][0] = (idi[ax]-next[ax]+pi[ax])%pi[ax];
			}
			for(int dir = 2; dir>=0; dir-=2){
				nbr[0][dir][k] = idin[0][dir] + pi[0] * (idi[1] + pi[1]* idi[2]);
				nbr[1][dir][k] = idi[0] + pi[0] * (idin[1][dir] + pi[1]* idi[2]);
				nbr[2][dir][k] = idi[0] + pi[0] * (idi[1] + pi[1]* idin[2][dir]);
			}

		}

		k = lt-1;
		is1 = 2 + ng[0][k] - ((pi[0]  - idi[0])*ng[0][lt-1])/pi[0];
		ie1 = 1 + ng[0][k] - ((pi[0]-1- idi[0])*ng[0][lt-1])/pi[0];
		n1 = 3 + ie1 - is1;
		is2 = 2 + ng[1][k] - ((pi[1]  - idi[1])*ng[1][lt-1])/pi[1];
		ie2 = 1 + ng[1][k] - ((pi[1]-1- idi[1])*ng[1][lt-1])/pi[1];
		n2 = 3 + ie2 - is2;
		is3 = 2 + ng[2][k] - ((pi[2]  - idi[2])*ng[2][lt-1])/pi[2];
		ie3 = 1 + ng[2][k] - ((pi[2]-1- idi[2])*ng[2][lt-1])/pi[2];
		n3 = 3 + ie3 - is3;

		ir[lt-1] = 1;
		for(j = lt-2; j>=0; j--){
			j1 = j+1;
			ir[j] = ir[j1]+m1[j1]*m2[j1]*m3[j1];
		}
		if(debugVec[1] >= 1){
			if(myRank == 0){
				System.out.println(" in setup, ");
				System.out.println(" me  k  lt  nx  ny  nz  n1  n2  n3  is1  is2  is3  ie1  ie2  ie3 ");
			}
			for(int i = 0; i < numProcess; i++){
				if(myRank == i){
					System.out.println(" "+myRank+"  "+k+"  "+lt+"  "+ng[0][k]+"  "+ng[1][k]+"  "+ng[2][k]+"  "+n1
							+"  "+n2+"  "+n3+"  "+is1+"  "+is2+"  "+is3+"  "+ie1+"  "+ie2+"  "+ie3);
				}
				commWork.Barrier();
			}
		}
		if(debugVec[1] >= 2){
			for(int i = 0; i<numProcess; i++){
				if(myRank == i){
					System.out.println(" ");
					System.out.println(" processor = "+myRank);
					for(k = lt-1; k>=0; k--){
						System.out.println(k+" idi= "+idi[0]+" "+idi[1]+" "+idi[2]+" "+
								"nbr=  "+nbr[0][0][k]+" "+nbr[0][2][k]+"  "+nbr[1][0][k]+" "+
								+nbr[1][2][k]+"  "+nbr[2][0][k]+" "+nbr[2][2][k]+" "+
								"mi= "+mi[0][k]+" "+mi[1][k]+" "+mi[2][k]+" ");
					}
					System.out.println("idi[s]= "+idi[0]+" "+idi[1]+" "+idi[2]);
					System.out.println("dead[1], dead[0] = "+dead[1]+" "+dead[0]);
					for(ax = 0; ax<3; ax++){
						System.out.println("giveEx[ax][1]"+giveEx[ax][1]);
						System.out.println("takEx[ax][1]"+takeEx[ax][1]);
					}
				}
				commWork.Barrier();
			}
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
		comm3Ex(u,uoffset,n1,n2,n3,k);

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

		double outBuff[] = new double[1];
		outBuff[0] = rnm[1];

		if(detailedTiming) timer.start(T_COMM);
		commWork.Allreduce(outBuff, 0, rnm, 1, 1, MPI.DOUBLE, MPI.MAX);
		if(detailedTiming) timer.stop(T_COMM);

		outBuff[0] = s;
		if(detailedTiming) timer.start(T_COMM);
		commWork.Allreduce(outBuff, 0, rnm, 0, 1, MPI.DOUBLE, MPI.SUM);
		if(detailedTiming) timer.stop(T_COMM);

		s = rnm[0];
		rnm[0] = Math.sqrt(s / dn);		
	}

	/*
	 * Report on norm
	 */
	private void repNrm(double u[], int offset, int n1, int n2, int n3, String title, int kk){

		double rnm[] = new double[2];		
		norm2u3(u,offset,n1,n2,n3,rnm,nx[kk],ny[kk],nz[kk]);
		if(myRank == 0){
			System.out.println(" Level "+kk+" in "+title+": "+rnm[0]+" "+rnm[1]);
		}
	}	

	private void comm3(double u[], int uoffset, int n1, int n2, int n3, int kk){

		if(!dead[kk]){
			for(int axis = 0; axis<3; axis++){
				if(numProcess != 1){
					ready(axis,0,kk);
					ready(axis,2,kk);

					give3(axis,2,u,uoffset,n1,n2,n3,kk);
					give3(axis,0,u,uoffset,n1,n2,n3,kk);

					take3(axis,0,u,uoffset,n1,n2,n3);
					take3(axis,2,u,uoffset,n1,n2,n3);
				}
				else{
					comm1p(axis, u, uoffset, n1, n2, n3, kk);
				}
			}
		}
		else{
			zero3(u,uoffset,n1,n2,n3);
		}
	}

	/*
	 * zran3  loads +1 at ten randomly chosen points,
	 * loads -1 at a different ten random points,
	 * and zero elsewhere.
	 */
	private void zran3(double z[],int n1, int n2, int n3, int nx, int ny, int k){

		double x1, xx;
		int mm = 10, pos, i;
		double ten[][] = new double[mm][2];
		int j1[][] = new int[mm][2];
		int j2[][] = new int[mm][2];
		int j3[][] = new int[mm][2];

		double a1 = rng.power(a, nx, 1, 0);
		double a2 = rng.power(a, nx, ny, 0);

		zero3(z,0,n1,n2,n3);

		double ai = rng.power(a, nx, is2-2+ny*(is3-2), is1-2);
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

		if(detailedTiming) timer.start(T_COMM);
		commWork.Barrier();
		if(detailedTiming) timer.stop(T_COMM);

		/* Now which of these are globally best? */
		int i1 = mm;
		int i0 = mm;
		int i1m = i1-1;
		int i0m = i0-1;
		double best = 0;
		int jg[][][] = new int[4][mm][2];
		double buff[] = new double[1];
		int jgTemp1[] = new int[4];
		int jgTemp2[] = new int[4];

		for(i = mm-1; i>=0; i--){

			pos = position3D(j1[i1m][1],j2[i1m][1],j3[i1m][1],n1,n2,n3);

			if(detailedTiming) timer.start(T_COMM);
			commWork.Allreduce(z, pos, buff, 0, 1, MPI.DOUBLE, MPI.MAX);
			if(detailedTiming) timer.stop(T_COMM);

			best = buff[0];

			if(best == z[pos]){
				jg[0][i][1] = myRank;
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

			ten[i][1] = best;

			jgTemp1[0] = jg[0][i][1];
			jgTemp1[1] = jg[1][i][1];
			jgTemp1[2] = jg[2][i][1];
			jgTemp1[3] = jg[3][i][1];

			if(detailedTiming) timer.start(T_COMM);
			commWork.Allreduce(jgTemp1, 0, jgTemp2, 0, 4, MPI.INT, MPI.MAX);
			if(detailedTiming) timer.stop(T_COMM);

			jg[0][i][1] = jgTemp2[0];
			jg[1][i][1] = jgTemp2[1];
			jg[2][i][1] = jgTemp2[2];
			jg[3][i][1] = jgTemp2[3];

			pos = position3D(j1[i0m][0],j2[i0m][0],j3[i0m][0],n1,n2,n3);

			if(detailedTiming) timer.start(T_COMM);
			commWork.Allreduce(z, pos, buff, 0, 1, MPI.DOUBLE, MPI.MIN);
			if(detailedTiming) timer.stop(T_COMM);

			best = buff[0];

			if(best == z[pos]){
				jg[0][i][0] = myRank;
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

			ten[i][0] = best;

			jgTemp1[0] = jg[0][i][0];
			jgTemp1[1] = jg[1][i][0];
			jgTemp1[2] = jg[2][i][0];
			jgTemp1[3] = jg[3][i][0];
			if(detailedTiming) timer.start(T_COMM);
			commWork.Allreduce(jgTemp1, 0, jgTemp2, 0, 4, MPI.INT, MPI.MAX);
			if(detailedTiming) timer.stop(T_COMM);

			jg[0][i][0] = jgTemp2[0];
			jg[1][i][0] = jgTemp2[1];
			jg[2][i][0] = jgTemp2[2];
			jg[3][i][0] = jgTemp2[3];

		}

		int m1 = i1+1;
		int m0 = i0+1;

		if(detailedTiming) timer.start(T_COMM);
		commWork.Barrier();
		if(detailedTiming) timer.stop(T_COMM);

		for(int i3 = 1; i3<=n3; i3++){
			for(int i2 = 1; i2<=n2; i2++){
				pos = position3D(1,i2,i3,n1,n2,n3);
				for(i1 = 1; i1<=n1; i1++){
					z[pos++] = 0;
				}
			}
		}

		for(i = mm-1; i>=m0-1; i--){
			z[position3D(j1[i][0],j2[i][0],j3[i][0],n1,n2,n3)] = -1;
		}

		for(i = mm-1; i>=m1-1; i--){
			z[position3D(j1[i][1],j2[i][1],j3[i][1],n1,n2,n3)] = 1;
		}
		comm3(z,0,n1,n2,n3,k-1);
	}

	private void showAll(double z[], int offset, int n1, int n2, int n3){
		int m1 = Math.min(n1, 18);
		int m2 = Math.min(n2, 14);
		int m3 = Math.min(n3, 18);

		System.out.println(" ");
		for(int i = 0; i<numProcess; i++){
			if(myRank == i){
				System.out.println(" id = "+myRank);
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
			commWork.Barrier();
		}
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

	/*
	 * comm3Ex  communicates to expand the number of processors
	 */
	private void comm3Ex(double u[],int offset, int n1, int n2, int n3,int kk){

		int axis;

		for(axis = 0; axis<3; axis++){
			if(numProcess != 1){
				if(takeEx[axis][kk]){
					ready(axis,0,kk);
					ready(axis,2,kk);		
					take3Ex(axis,0,u,offset,n1,n2,n3);
					take3Ex(axis,2,u,offset,n1,n2,n3);
				}
				if(giveEx[axis][kk]){
					give3Ex(axis,2,u,offset,n1,n2,n3,kk);
					give3Ex(axis,0,u,offset,n1,n2,n3,kk);
				}
			}
			else{
				comm1pEx(axis, u, offset, n1, n2, n3, kk);
			}
		}
	}

	/*
	 * Ready allocates a buffer to take in a message
	 */
	private void ready(int axis, int dir, int k){

		int buffId = 1 + dir;	
		int buffLen = nm2;

		for(int i = 0; i<nm2; i++)
			buff[buffId][i] = 0;

		/* Fake message request type */
		if(detailedTiming) timer.start(T_COMM);
		msgId[axis][dir][0] = 
			commWork.Irecv(buff[buffId], 0, buffLen, MPI.DOUBLE, nbr[axis][fakeNegate(dir)][k], msgType[axis][dir]);
		if(detailedTiming) timer.stop(T_COMM);
	}

	/*
	 * This method simulates the behaviour of the negation
	 * of a number used as index between -1 and 1
	 */
	private int fakeNegate(int index){
		if(index == 2){
			return 0;
		}
		else if(index == 0){
			return 2;
		}
		else if(index == 1){
			return 1;
		}
		return Integer.MAX_VALUE;
	}

	/*
	 * give3 sends border data out in the requested direction
	 */
	private void give3(int axis, int dir, double u[], int offset, int n1, int n2, int n3, int k){

		int buffId = dir;
		int buffLen = 0;

		if(axis == 0){
			if(dir == 0){
				for(int i3 = 2; i3<=n3-1 ;i3++){
					int pos = position3D(2,2,i3,n1,n2,n3)+offset;
					for(int i2 = 2; i2<=n2-1 ;i2++){
						buff[buffId][buffLen] = u[pos];
						pos += n1;
						buffLen++;
					}
				}
			}
			else if(dir == 2){
				for(int i3 = 2; i3<=n3-1 ;i3++){
					int pos = position3D(n1-1,2,i3,n1,n2,n3)+offset;
					for(int i2 = 2; i2<=n2-1 ;i2++){
						buff[buffId][buffLen] = u[pos];
						pos += n1;
						buffLen++;
					}
				}				
			}
		}
		else if(axis == 1){
			if(dir == 0){
				for(int i3 = 2; i3<=n3-1 ;i3++){
					int pos = position3D(1,2,i3,n1,n2,n3)+offset;
					for(int i1 = 1; i1<=n1 ;i1++){
						buff[buffId][buffLen] = u[pos];
						pos++;
						buffLen++;
					}
				}
			}
			else if(dir == 2){
				for(int i3 = 2; i3<=n3-1 ;i3++){
					int pos = position3D(1,n2-1,i3,n1,n2,n3)+offset;
					for(int i1 = 1; i1<=n1 ;i1++){
						buff[buffId][buffLen] = u[pos];
						pos++;
						buffLen++;
					}
				}				
			}
		}
		else if(axis == 2){
			if(dir == 0){
				for(int i2 = 1; i2<=n2 ;i2++){
					int pos = position3D(1,i2,2,n1,n2,n3)+offset;
					for(int i1 = 1; i1<=n1 ;i1++){
						buff[buffId][buffLen] = u[pos];
						pos++;
						buffLen++;
					}
				}
			}
			else if(dir == 2){
				for(int i2 = 1; i2<=n2 ;i2++){
					int pos = position3D(1,i2,n3-1,n1,n2,n3)+offset;
					for(int i1 = 1; i1<=n1 ;i1++){
						buff[buffId][buffLen] = u[pos];
						pos++;
						buffLen++;
					}
				}				
			}
		}

		if(detailedTiming) timer.start(T_COMM);
		commWork.Send(buff[buffId], 0, buffLen, MPI.DOUBLE, nbr[axis][dir][k], msgType[axis][dir]);
		if(detailedTiming) timer.stop(T_COMM);
	}

	/*
	 * take3 copies in border data from the requested direction
	 */
	private void take3(int axis, int dir, double u[], int offset, int n1, int n2, int n3){

		if(detailedTiming) timer.start(T_COMM);
		msgId[axis][dir][0].Wait();
		if(detailedTiming) timer.stop(T_COMM);

		int buffId = 1+dir;
		int indx = 0;

		if(axis == 0){
			if(dir == 0){
				for(int i3 = 2; i3<=n3-1; i3++){
					int pos = position3D(n1,2,i3,n1,n2,n3)+offset;
					for(int i2 = 2; i2<=n2-1; i2++){
						u[pos] = buff[buffId][indx];
						pos += n1;
						indx++;
					}
				}
			}
			else if(dir == 2){
				for(int i3 = 2; i3<=n3-1; i3++){
					int pos = position3D(1,2,i3,n1,n2,n3)+offset;
					for(int i2 = 2; i2<=n2-1; i2++){
						u[pos] = buff[buffId][indx];
						pos += n1;
						indx++;
					}
				}
			}
		}
		else if(axis == 1){
			if(dir == 0){
				for(int i3 = 2; i3<=n3-1; i3++){
					int pos = position3D(1,n2,i3,n1,n2,n3)+offset;
					for(int i1 = 1; i1<=n1; i1++){
						u[pos] = buff[buffId][indx];
						pos++;
						indx++;
					}
				}
			}
			else if(dir == 2){
				for(int i3 = 2; i3<=n3-1; i3++){
					int pos = position3D(1,1,i3,n1,n2,n3)+offset;
					for(int i1 = 1; i1<=n1; i1++){
						u[pos] = buff[buffId][indx];
						pos++;
						indx++;
					}
				}				
			}
		}
		else if(axis == 2){
			if(dir == 0){
				for(int i2 = 1; i2<=n2; i2++){
					int pos = position3D(1,i2,n3,n1,n2,n3)+offset;
					for(int i1 = 1; i1<=n1; i1++){
						u[pos] = buff[buffId][indx];
						pos++;
						indx++;
					}
				}
			}
			else if(dir == 2){
				for(int i2 = 1; i2<=n2; i2++){
					int pos = position3D(1,i2,1,n1,n2,n3)+offset;
					for(int i1 = 1; i1<=n1; i1++){
						u[pos] = buff[buffId][indx];
						pos++;
						indx++;
					}
				}
			}
		}		
	}

	private void comm1p(int axis, double u[], int offset, int n1, int n2, int n3, int kk){

		int dir = 0;
		int buffId = 1+dir;

		for(int i = 0; i<nm2; i++)
			buff[buffId][i] = 0;

		dir = 2;
		buffId = 1+dir;

		for(int i = 0; i<nm2; i++)
			buff[buffId][i] = 0;

		dir = 2;
		buffId = dir;
		int buffLen = 0;

		if(axis == 0){
			for(int i3 = 2; i3<=n3-1; i3++){
				int pos = position3D(n1-1,2,i3,n1,n2,n3)+offset;
				for(int i2 = 2; i2<=n2-1; i2++){
					buff[buffId][buffLen] = u[pos];
					buffLen++;
					pos += n1;
				}
			}
		}
		else if(axis == 1){
			for(int i3 = 2; i3<=n3-1; i3++){
				int pos = position3D(1,n2-1,i3,n1,n2,n3)+offset;
				for(int i1 = 1; i1<=n1; i1++){
					buff[buffId][buffLen] = u[pos];
					buffLen++;
					pos++;
				}
			}
		}
		else if(axis == 2){
			for(int i2 = 1; i2<=n2; i2++){
				int pos = position3D(1,i2,n3-1,n1,n2,n3)+offset;
				for(int i1 = 1; i1<=n1; i1++){
					buff[buffId][buffLen] = u[pos];
					buffLen++;
					pos++;
				}
			}
		}

		dir = 0;
		buffId = dir;
		buffLen = 0;

		if(axis == 0){
			for(int i3 = 2; i3<=n3-1; i3++){
				int pos = position3D(2,2,i3,n1,n2,n3)+offset;
				for(int i2 = 2; i2<=n2-1; i2++){
					buff[buffId][buffLen] = u[pos];
					buffLen++;
					pos += n1;
				}
			}
		}
		else if(axis == 1){
			for(int i3 = 2; i3<=n3-1; i3++){
				int pos = position3D(1,2,i3,n1,n2,n3)+offset;
				for(int i1 = 1; i1<=n1; i1++){
					buff[buffId][buffLen] = u[pos];
					buffLen++;
					pos++;
				}
			}
		}
		else if(axis == 2){
			for(int i2 = 1; i2<=n2; i2++){
				int pos = position3D(1,i2,2,n1,n2,n3)+offset;
				for(int i1 = 1; i1<=n1; i1++){
					buff[buffId][buffLen] = u[pos];
					buffLen++;
					pos++;
				}
			}
		}

		for(int i = 0; i<nm2; i++){
			buff[3][i] = buff[2][i];
			buff[1][i] = buff[0][i];
		}

		dir = 0;
		buffId = 1+dir;
		int indx = 0;

		if(axis == 0){
			for(int i3 = 2; i3<=n3-1; i3++){
				int pos = position3D(n1,2,i3,n1,n2,n3)+offset;
				for(int i2 = 2; i2<=n2-1; i2++){
					u[pos] = buff[buffId][indx];
					indx++;
					pos += n1;
				}
			}
		}
		else if(axis == 1){
			for(int i3 = 2; i3<=n3-1; i3++){
				int pos = position3D(1,n2,i3,n1,n2,n3)+offset;
				for(int i1 = 1; i1<=n1; i1++){
					u[pos] = buff[buffId][indx];
					indx++;
					pos++;
				}
			}
		}
		else if(axis == 2){
			for(int i2 = 1; i2<=n2; i2++){
				int pos = position3D(1,i2,n3,n1,n2,n3)+offset;
				for(int i1 = 1; i1<=n1; i1++){
					u[pos] = buff[buffId][indx];
					indx++;
					pos++;
				}
			}
		}

		dir = 2;
		buffId = 1+dir;
		indx = 0;

		if(axis == 0){
			for(int i3 = 2; i3<=n3-1; i3++){
				int pos = position3D(1,2,i3,n1,n2,n3)+offset;
				for(int i2 = 2; i2<=n2-1; i2++){
					u[pos] = buff[buffId][indx];
					indx++;
					pos += n1;
				}
			}
		}
		else if(axis == 1){
			for(int i3 = 2; i3<=n3-1; i3++){
				int pos = position3D(1,1,i3,n1,n2,n3)+offset;
				for(int i1 = 1; i1<=n1; i1++){
					u[pos] = buff[buffId][indx];
					indx++;
					pos++;
				}
			}
		}
		else if(axis == 2){
			for(int i2 = 1; i2<=n2; i2++){
				int pos = position3D(1,i2,1,n1,n2,n3)+offset;
				for(int i1 = 1; i1<=n1; i1++){
					u[pos] = buff[buffId][indx];
					indx++;
					pos++;
				}
			}
		}

	}

	private void take3Ex(int axis, int dir, double u[], int offset, int n1, int n2, int n3){

		if(detailedTiming) timer.start(T_COMM);
		msgId[axis][dir][0].Wait();
		if(detailedTiming) timer.stop(T_COMM);

		int buffId = 1 + dir;
		int indx = 0;
		int i1, i2, i3, pos;

		if(axis == 0){
			if(dir == 0){
				for(i3=1; i3<=n3; i3++){
					pos = position3D(n1,1,i3,n1,n2,n3)+offset;
					for(i2=1; i2<=n2; i2++){
						u[pos] = buff[buffId][indx];
						indx++;
						pos+=n1;
					}
				}
			}
			else if(dir == 2){
				for(i3=1; i3<=n3; i3++){
					for(i2=1; i2<=n2; i2++){
						pos = position3D(1,i2,i3,n1,n2,n3)+offset;
						for(i1=1; i1<=2; i1++){
							u[pos] = buff[buffId][indx];
							indx++;
							pos++;
						}
					}
				}
			}
		}
		else if(axis == 1){
			if(dir == 0){
				for(i3=1; i3<=n3; i3++){
					pos = position3D(1,n2,i3,n1,n2,n3)+offset;
					for(i1=1; i1<=n1; i1++){
						u[pos] = buff[buffId][indx];
						indx++;
						pos++;
					}
				}
			}
			else if(dir == 2){
				for(i3=1; i3<=n3; i3++){
					for(i2=1; i2<=2; i2++){
						pos = position3D(1,i2,i3,n1,n2,n3)+offset;
						for(i1=1; i1<=n1; i1++){
							u[pos] = buff[buffId][indx];
							indx++;
							pos++;
						}
					}
				}
			}			
		}
		else if(axis == 2){
			if(dir == 0){
				for(i2=1; i2<=n2; i2++){
					pos = position3D(1,i2,n3,n1,n2,n3)+offset;
					for(i1=1; i1<=n1; i1++){
						u[pos] = buff[buffId][indx];
						indx++;
						pos++;
					}
				}
			}
			else if(dir == 2){
				for(i3=1; i3<=2; i3++){
					for(i2=1; i2<=n2; i2++){
						pos = position3D(1,i2,i3,n1,n2,n3)+offset;
						for(i1=1; i1<=n1; i1++){
							u[pos] = buff[buffId][indx];
							indx++;
							pos++;
						}
					}
				}
			}
		}
	}

	/*
	 * give3Ex sends border data out to expand number of processors
	 */
	private void give3Ex(int axis, int dir, double u[], int offset, int n1, int n2, int n3, int k){

		int buffId = dir;
		int buffLen = 0;

		if(axis == 0){
			if(dir == 0){
				for(int i3=1; i3<=n3; i3++){
					int pos = position3D(2,1,i3,n1,n2,n3)+offset;
					for(int i2=1; i2<=n2; i2++){
						buff[buffId][buffLen] = u[pos];
						buffLen++;
						pos+=n1;
					}
				}
			}
			else if(dir == 2){
				for(int i3=1; i3<=n3; i3++){
					for(int i2=1; i2<=n2; i2++){
						/* the two last values of each row, as in give3_ex of NPB-MPI (take3Ex reads two) */
						int pos = position3D(n1-1,i2,i3,n1,n2,n3)+offset;
						for(int i1=n1-1; i1<=n1; i1++){
							buff[buffId][buffLen] = u[pos];
							buffLen++;
							pos++;
						}
					}
				}
			}
		}
		else if(axis == 1){
			if(dir == 0){
				for(int i3=1; i3<=n3; i3++){
					int pos = position3D(1,2,i3,n1,n2,n3)+offset;
					for(int i1=1; i1<=n1; i1++){
						buff[buffId][buffLen] = u[pos];
						buffLen++;
						pos++;
					}
				}
			}
			else if(dir == 2){
				for(int i3=1; i3<=n3; i3++){
					for(int i2=n2-1; i2<=n2; i2++){
						int pos = position3D(1,i2,i3,n1,n2,n3)+offset;
						for(int i1=1; i1<=n1; i1++){
							buff[buffId][buffLen] = u[pos];
							buffLen++;
							pos++;
						}
					}
				}
			}			
		}
		else if(axis == 2){
			if(dir == 0){
				for(int i2=1; i2<=n2; i2++){
					int pos = position3D(1,i2,2,n1,n2,n3)+offset;
					for(int i1=1; i1<=n1; i1++){
						buff[buffId][buffLen] = u[pos];
						buffLen++;
						pos++;
					}
				}
			}
			else if(dir == 2){
				for(int i3=n3-1; i3<=n3; i3++){
					for(int i2=1; i2<=n2; i2++){
						int pos = position3D(1,i2,i3,n1,n2,n3)+offset;
						for(int i1=1; i1<=n1; i1++){
							buff[buffId][buffLen] = u[pos];
							buffLen++;
							pos++;
						}
					}
				}
			}			
		}

		if(detailedTiming) timer.start(T_COMM);
		commWork.Send(buff[buffId], 0, buffLen, MPI.DOUBLE, nbr[axis][dir][k], msgType[axis][dir]);
		if(detailedTiming) timer.stop(T_COMM);		
	}

	private void comm1pEx(int axis, double u[], int offset, int n1, int n2, int n3, int kk){

		if(takeEx[axis][kk]){
			int dir = 0;
			int buffId = 1+dir;

			for(int i=0; i<nm2; i++)
				buff[buffId][i] = 0;

			dir = 2;
			buffId = 1+dir;

			for(int i=0; i<nm2; i++)
				buff[buffId][i] = 0;

			dir = 0;
			buffId = 1+dir;
			int indx = 0;

			if(axis == 0){
				for(int i3 = 1; i3<=n3; i3++){
					int pos = position3D(n1,1,i3,n1,n2,n3)+offset;
					for(int i2 = 1; i2<=n2; i2++){
						u[pos] = buff[buffId][indx];
						indx++;
						pos+=n1;
					}
				}
			}
			else if(axis == 1){
				for(int i3 = 1; i3<=n3; i3++){
					int pos = position3D(1,n2,i3,n1,n2,n3)+offset;
					for(int i1 = 1; i1<=n1; i1++){
						u[pos] = buff[buffId][indx];
						indx++;
						pos++;
					}
				}
			}
			else if(axis == 2){
				for(int i2 = 1; i2<=n2; i2++){
					int pos = position3D(1,i2,n3,n1,n2,n3)+offset;
					for(int i1 = 1; i1<=n1; i1++){
						u[pos] = buff[buffId][indx];
						indx++;
						pos++;
					}
				}				
			}

			dir = 2;
			buffId = 1+dir;
			indx = 0;

			if(axis == 0){
				for(int i3=1; i3<=n3; i3++){
					for(int i2=1; i2<=n2; i2++){
						int pos = position3D(1,i2,i3,n1,n2,n3)+offset;
						for(int i1=1; i1<=2; i1++){
							u[pos] = buff[buffId][indx];
							indx++;
							pos++;
						}
					}
				}
			}
			else if(axis == 1){
				for(int i3=1; i3<=n3; i3++){
					for(int i2=1; i2<=2; i2++){
						int pos = position3D(1,i2,i3,n1,n2,n3)+offset;
						for(int i1=1; i1<=n1; i1++){
							u[pos] = buff[buffId][indx];
							indx++;
							pos++;
						}
					}
				}
			}
			else if(axis == 2){
				for(int i3=1; i3<=2; i3++){
					for(int i2=1; i2<=n2; i2++){
						int pos = position3D(1,i2,i3,n1,n2,n3)+offset;
						for(int i1=1; i1<=n1; i1++){
							u[pos] = buff[buffId][indx];
							indx++;
							pos++;
						}
					}
				}
			}

		}

		if(giveEx[axis][kk]){

			int dir = 2;
			int buffId = dir;
			int buffLen = 0;

			if(axis == 0){
				for(int i3=1; i3<=n3; i3++){
					for(int i2=1; i2<=n2; i2++){
						int pos = position3D(n1-1,i2,i3,n1,n2,n3)+offset;
						for(int i1=n1-1; i1<=n1; i1++){
							buff[buffId][buffLen] = u[pos];
							buffLen++;
							pos++;
						}	
					}	
				}
			}
			else if(axis == 1){
				for(int i3=1; i3<=n3; i3++){
					for(int i2=n2-1; i2<=n2; i2++){
						int pos = position3D(1,i2,i3,n1,n2,n3)+offset;
						for(int i1=1; i1<=n1; i1++){
							buff[buffId][buffLen] = u[pos];
							buffLen++;
							pos++;
						}	
					}	
				}
			}
			else if(axis == 2){
				for(int i3=n3-1; i3<=n3; i3++){
					for(int i2=1; i2<=n2; i2++){
						int pos = position3D(1,i2,i3,n1,n2,n3)+offset;
						for(int i1=1; i1<=n1; i1++){
							buff[buffId][buffLen] = u[pos];
							buffLen++;
							pos++;
						}	
					}	
				}
			}

			dir = 0;
			buffId = dir;
			buffLen = 0;

			if(axis == 0){
				for(int i3=1; i3<=n3; i3++){
					int pos = position3D(2,1,i3,n1,n2,n3)+offset;
					for(int i2=1; i2<=n2; i2++){
						buff[buffId][buffLen] = u[pos];
						buffLen++;
						pos+=n1;
					}
				}
			}
			else if(axis == 1){
				for(int i3=1; i3<=n3; i3++){
					int pos = position3D(1,2,i3,n1,n2,n3)+offset;
					for(int i1=1; i1<=n1; i1++){
						buff[buffId][buffLen] = u[pos];
						buffLen++;
						pos++;
					}
				}
			}
			else if(axis == 2){
				for(int i2=1; i2<=n2; i2++){
					int pos = position3D(1,i2,2,n1,n2,n3)+offset;
					for(int i1=1; i1<=n1; i1++){
						buff[buffId][buffLen] = u[pos];
						buffLen++;
						pos++;
					}
				}
			}

		}

		for(int i = 0; i<nm2; i++){
			buff[3][i] = buff[2][i];
			buff[1][i] = buff[0][i];
		}	
	}
}
