/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB 3.4.4 OpenMP (NPB3.4-OMP/LU)
 */
package NPB.SER;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import NPB.Benchmark;
import NPB.common.io.BMArgs;
import NPB.common.io.BMResults;
import NPB.common.timer.Timer;
import NPB.exceptions.BadClassException;
import NPB.exceptions.BadInputException;
import NPB.exceptions.NPBException;

public class LU implements Benchmark{

	private static final String BM_NAME="LU";
	private char CLASS;

	private boolean detailedTiming;

	/* compiled values of the class (isiz1 = isiz2 = isiz3 = problem_size, itmax_default, dt_default) */
	private int problemSize;
	private int itmaxDefault;
	private double dtDefault;

	private int ipr, inorm;
	private int itmax;
	private double dt, omega;
	private double tolrsd[] = new double[5];

	private int nx0, ny0, nz0;
	private int nx, ny, nz;
	private int ist, iend, jst, jend;
	private int ii1, ii2, ji1, ji2, ki1, ki2;

	/*
	 * Field arrays. As in the original (u(5,isiz1/2*2+1,isiz2/2*2+1,isiz3)), the
	 * first two dimensions are padded by 1 for even sizes and the 5 components
	 * of a point are contiguous: u(m,i,j,k), with the indices of the original
	 * (all 1-based), is u[5*ix(i,j,k)+m-1] (and m runs from 0 here).
	 */
	private int ldx, ldy;
	private double u[];
	private double rsd[];
	private double frct[];
	private double qs[];
	private double rhoI[];

	/* work arrays of one grid line: flux(5,isiz1), utmp(6,isiz3) and rtmp(5,isiz3) of rhs */
	private double flux[][];
	private double utmp[][];
	private double rtmp[][];

	/* jacobian blocks of the point i: a(5,5,i) is a[25*i+m+5*n] (first index fastest) */
	private double a[], b[], c[], d[];

	private double phi1[][], phi2[][];

	private double ce[][] = new double[5][13];

	private static final double c1 = 1.40, c2 = 0.40, c3 = 1.00e-01, c4 = 1.00, c5 = 1.40;

	private double dxi, deta, dzeta;
	private double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3;
	private double dx1, dx2, dx3, dx4, dx5;
	private double dy1, dy2, dy3, dy4, dy5;
	private double dz1, dz2, dz3, dz4, dz5;
	private double dssp;

	private double rsdnm[] = new double[5];
	private double errnm[] = new double[5];
	private double frc;
	private double maxtime;

	private BMResults res;
	private Timer timer;

	/*
	 * Different timers
	 */
	private final int T_TOTAL = 1;
	private final int T_RHSX = 2;
	private final int T_RHSY = 3;
	private final int T_RHSZ = 4;
	private final int T_RHS = 5;
	private final int T_JACLD = 6;
	private final int T_BLTS = 7;
	private final int T_JACU = 8;
	private final int T_BUTS = 9;
	private final int T_ADD = 10;
	private final int T_L2NORM = 11;
	private final int T_LAST = 11;

	public static void main(String[] args) throws NPBException {
		LU lu = null;

		BMArgs.parseCmdLineArgs(args,BM_NAME);
		try{
			lu = new LU(BMArgs.CLASS,BMArgs.DETAILED_TIMING);
		}catch(OutOfMemoryError e){
			BMArgs.outOfMemoryMessage();
			throw e;
		}
		lu.runBenchmark(args);
	}

	public BMResults getResults(){
		return res;
	}

	public LU(char classSize, boolean dtiming){

		CLASS = classSize;
		detailedTiming = dtiming;
	}

	public void build(String[] args) throws NPBException{

		switch( CLASS )
		{
		case 'S':
			problemSize = 12;
			itmaxDefault = 50;
			dtDefault = 0.5;
			break;
		case 'W':
			problemSize = 33;
			itmaxDefault = 300;
			dtDefault = 1.5e-3;
			break;
		case 'A':
			problemSize = 64;
			itmaxDefault = 250;
			dtDefault = 2.0;
			break;
		case 'B':
			problemSize = 102;
			itmaxDefault = 250;
			dtDefault = 2.0;
			break;
		case 'C':
			problemSize = 162;
			itmaxDefault = 250;
			dtDefault = 2.0;
			break;
		case 'D':
			problemSize = 408;
			itmaxDefault = 300;
			dtDefault = 1.0;
			break;
		default:
			throw new BadClassException("Specified class (" + CLASS + ") not valid for this benchmark\n"
					+ "Valid classes are S, W, A, B, C and D");
		}
	}

	private void terminate(){
	}

	public void runBenchmark(String[] args) throws NPBException {

		this.build(args);

		timer = new Timer();

		readInput();

		domain();

		allocSpace();

		setcoeff();

		setbv();

		setiv();

		erhs();

		/* One time step to touch all code, and reinitialize */
		ssor(1);

		setbv();

		setiv();

		ssor(itmax);

		error();

		pintgr();

		int verified = verify(CLASS);

		double mflops = 0;
		if(maxtime != 0)
			/* (the constants are REAL(4) literals in the original, as in FT; here double) */
			mflops = 1.0E-6*(double)itmax*(1984.77*(double)nx0
					*(double)ny0
					*(double)nz0
					-10923.3*(((double)(nx0+ny0+nz0)/3.0)*((double)(nx0+ny0+nz0)/3.0))
					+27770.9*(double)(nx0+ny0+nz0)/3.0
					-144010.0)
					/maxtime;

		double times[] = {0,0};
		times[0] = maxtime;

		res = new BMResults(BM_NAME,
				CLASS,
				nx0,
				ny0,
				nz0,
				itmax,
				times,
				mflops,
				"floating point",
				verified,
				1,
				detailedTiming);

		res.printSER();

		if (detailedTiming)
			printAdditionalTimers();

		terminate();
	}

	/*
	 * Reads the optional input file inputlu.data or takes the compiled
	 * defaults of the class. As in the original, the file alternates two
	 * comment lines with one line of values: ipr and inorm, itmax, dt,
	 * omega, the five tolerances and the grid size.
	 */
	private void readInput(){

		boolean inputData = true;

		BMArgs.bannerSER(BM_NAME, CLASS);

		try{

			BufferedReader input = new BufferedReader(new FileReader("inputlu.data"));

			System.out.println(" Reading from input file inputlu.data");

			String line[];

			skipLines(input, 2);
			line = values(input);
			ipr = Integer.valueOf(line[0]);
			inorm = Integer.valueOf(line[1]);

			skipLines(input, 2);
			line = values(input);
			itmax = Integer.valueOf(line[0]);

			skipLines(input, 2);
			line = values(input);
			dt = fortranDouble(line[0]);

			skipLines(input, 2);
			line = values(input);
			omega = fortranDouble(line[0]);

			skipLines(input, 2);
			line = values(input);
			for(int m = 0; m<5; m++)
				tolrsd[m] = fortranDouble(line[m]);

			skipLines(input, 2);
			line = values(input);
			nx0 = Integer.valueOf(line[0]);
			ny0 = Integer.valueOf(line[1]);
			nz0 = Integer.valueOf(line[2]);

			input.close();

		}
		catch(FileNotFoundException e){
			inputData = false;
		}
		catch(IOException e){
			System.err.println(" Error reading input file: "+e.getMessage());
			System.exit(-7);
		}
		if(!inputData){
			System.out.println(" No input file inputlu.data. Using compiled defaults");
			ipr = 1;
			inorm = itmaxDefault;
			itmax = itmaxDefault;
			dt = dtDefault;
			omega = 1.2;
			/* the default tolerances are REAL(4) literals (1.0e-08) in the original */
			for(int m = 0; m<5; m++)
				tolrsd[m] = (double)1.0e-08f;
			nx0 = problemSize;
			ny0 = problemSize;
			nz0 = problemSize;
		}

		if(nx0 < 4 || ny0 < 4 || nz0 < 4){
			System.err.println("     PROBLEM SIZE IS TOO SMALL - \n"
					+ "     SET EACH OF NX, NY AND NZ AT LEAST EQUAL TO 5");
			System.exit(1);
		}
		if(nx0 > problemSize || ny0 > problemSize || nz0 > problemSize){
			System.err.println("     PROBLEM SIZE IS TOO LARGE - \n"
					+ "     NX, NY AND NZ SHOULD BE EQUAL TO \n"
					+ "     ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY");
			System.exit(1);
		}

		System.out.println(" Size:  "+nx0+"x "+ny0+"x "+nz0);
		System.out.println(" Iterations:  "+itmax);
		System.out.println();
	}

	private static void skipLines(BufferedReader input, int n) throws IOException{
		for(int i = 0; i<n; i++)
			input.readLine();
	}

	/* The values of a line (list-directed read: separated by blanks or commas) */
	private static String[] values(BufferedReader input) throws IOException{
		return input.readLine().trim().split("[\\s,]+");
	}

	/* The input files write reals in Fortran form (0.5d0) */
	private static double fortranDouble(String s){
		return Double.valueOf(s.replace('d', 'e').replace('D', 'E'));
	}

	private void domain(){

		nx = nx0;
		ny = ny0;
		nz = nz0;

		ist = 2;
		iend = nx - 1;
		jst = 2;
		jend = ny - 1;
		ii1 = 2;
		ii2 = nx0 - 1;
		ji1 = 2;
		ji2 = ny0 - 2;
		ki1 = 3;
		ki2 = nz0 - 1;
	}

	/* Index of the scalar field point (i,j,k) (1-based, as in the original) */
	private int ix(int i, int j, int k){
		return (i-1) + ldx*((j-1) + ldy*(k-1));
	}

	private void allocSpace() throws NPBException{

		ldx = problemSize/2*2 + 1;
		ldy = problemSize/2*2 + 1;
		int kmax = problemSize;

		long points = (long)ldx*ldy*kmax;
		if(5*points > Integer.MAX_VALUE)
			throw new BadInputException("Problem size too big for Java arrays");

		int n = (int)points;

		u = new double[5*n];
		rsd = new double[5*n];
		frct = new double[5*n];
		qs = new double[n];
		rhoI = new double[n];

		int nmax = problemSize + 2;
		flux = new double[5][nmax];
		utmp = new double[6][nmax];
		rtmp = new double[5][nmax];

		a = new double[25*nmax];
		b = new double[25*nmax];
		c = new double[25*nmax];
		d = new double[25*nmax];

		phi1 = new double[nmax][nmax];
		phi2 = new double[nmax][nmax];
	}

	private void setcoeff(){

		dxi   = 1.0 / ( nx0 - 1 );
		deta  = 1.0 / ( ny0 - 1 );
		dzeta = 1.0 / ( nz0 - 1 );

		tx1 = 1.0 / ( dxi * dxi );
		tx2 = 1.0 / ( 2.0 * dxi );
		tx3 = 1.0 / dxi;

		ty1 = 1.0 / ( deta * deta );
		ty2 = 1.0 / ( 2.0 * deta );
		ty3 = 1.0 / deta;

		tz1 = 1.0 / ( dzeta * dzeta );
		tz2 = 1.0 / ( 2.0 * dzeta );
		tz3 = 1.0 / dzeta;

		dx1 = 0.75;
		dx2 = dx1;
		dx3 = dx1;
		dx4 = dx1;
		dx5 = dx1;

		dy1 = 0.75;
		dy2 = dy1;
		dy3 = dy1;
		dy4 = dy1;
		dy5 = dy1;

		dz1 = 1.00;
		dz2 = dz1;
		dz3 = dz1;
		dz4 = dz1;
		dz5 = dz1;

		dssp = ( Math.max(dx1, Math.max(dy1, dz1)) ) / 4.0;

		ce[0][0] = 2.0;
		ce[0][1] = 0.0;
		ce[0][2] = 0.0;
		ce[0][3] = 4.0;
		ce[0][4] = 5.0;
		ce[0][5] = 3.0;
		ce[0][6] = 5.0e-01;
		ce[0][7] = 2.0e-02;
		ce[0][8] = 1.0e-02;
		ce[0][9] = 3.0e-02;
		ce[0][10] = 5.0e-01;
		ce[0][11] = 4.0e-01;
		ce[0][12] = 3.0e-01;
		ce[1][0] = 1.0;
		ce[1][1] = 0.0;
		ce[1][2] = 0.0;
		ce[1][3] = 0.0;
		ce[1][4] = 1.0;
		ce[1][5] = 2.0;
		ce[1][6] = 3.0;
		ce[1][7] = 1.0e-02;
		ce[1][8] = 3.0e-02;
		ce[1][9] = 2.0e-02;
		ce[1][10] = 4.0e-01;
		ce[1][11] = 3.0e-01;
		ce[1][12] = 5.0e-01;
		ce[2][0] = 2.0;
		ce[2][1] = 2.0;
		ce[2][2] = 0.0;
		ce[2][3] = 0.0;
		ce[2][4] = 0.0;
		ce[2][5] = 2.0;
		ce[2][6] = 3.0;
		ce[2][7] = 4.0e-02;
		ce[2][8] = 3.0e-02;
		ce[2][9] = 5.0e-02;
		ce[2][10] = 3.0e-01;
		ce[2][11] = 5.0e-01;
		ce[2][12] = 4.0e-01;
		ce[3][0] = 2.0;
		ce[3][1] = 2.0;
		ce[3][2] = 0.0;
		ce[3][3] = 0.0;
		ce[3][4] = 0.0;
		ce[3][5] = 2.0;
		ce[3][6] = 3.0;
		ce[3][7] = 3.0e-02;
		ce[3][8] = 5.0e-02;
		ce[3][9] = 4.0e-02;
		ce[3][10] = 2.0e-01;
		ce[3][11] = 1.0e-01;
		ce[3][12] = 3.0e-01;
		ce[4][0] = 5.0;
		ce[4][1] = 4.0;
		ce[4][2] = 3.0;
		ce[4][3] = 2.0;
		ce[4][4] = 1.0e-01;
		ce[4][5] = 4.0e-01;
		ce[4][6] = 3.0e-01;
		ce[4][7] = 5.0e-02;
		ce[4][8] = 4.0e-02;
		ce[4][9] = 3.0e-02;
		ce[4][10] = 1.0e-01;
		ce[4][11] = 3.0e-01;
		ce[4][12] = 2.0e-01;
	}

	/* The exact solution at the point (i,j,k) (1-based) */
	private void exact(int i, int j, int k, double u000ijk[]){

		double xi, eta, zeta;

		xi   = ( (double)( i - 1 ) ) / ( nx0 - 1 );
		eta  = ( (double)( j - 1 ) ) / ( ny0 - 1 );
		zeta = ( (double)( k - 1 ) ) / ( nz - 1 );

		for(int m = 0; m<5; m++){
			u000ijk[m] =  ce[m][0]
				+ (ce[m][1]
				+ (ce[m][4]
				+ (ce[m][7]
				+  ce[m][10] * xi) * xi) * xi) * xi
				+ (ce[m][2]
				+ (ce[m][5]
				+ (ce[m][8]
				+  ce[m][11] * eta) * eta) * eta) * eta
				+ (ce[m][3]
				+ (ce[m][6]
				+ (ce[m][9]
				+  ce[m][12] * zeta) * zeta) * zeta) * zeta;
		}
	}

	/* Boundary values of u (the exact solution on the six faces) */
	private void setbv(){
		int i, j, k, m;
		double temp1[] = new double[5], temp2[] = new double[5];
		for (j = 1; j <= ny; j++){
			for (i = 1; i <= nx; i++){
				exact( i, j, 1, temp1 );
				exact( i, j, nz, temp2 );
				for (m = 0; m <= 4; m++){
					u[5*ix(i,j,1)+m] = temp1[m];
					u[5*ix(i,j,nz)+m] = temp2[m];
				}
			}
		}
		for (k = 1; k <= nz; k++){
			for (i = 1; i <= nx; i++){
				exact( i, 1, k, temp1 );
				exact( i, ny, k, temp2 );
				for (m = 0; m <= 4; m++){
					u[5*ix(i,1,k)+m] = temp1[m];
					u[5*ix(i,ny,k)+m] = temp2[m];
				}
			}
		}
		for (k = 1; k <= nz; k++){
			for (j = 1; j <= ny; j++){
				exact( 1, j, k, temp1 );
				exact( nx, j, k, temp2 );
				for (m = 0; m <= 4; m++){
					u[5*ix(1,j,k)+m] = temp1[m];
					u[5*ix(nx,j,k)+m] = temp2[m];
				}
			}
		}
	}

	/* Initial values of u: transfinite interpolation of the boundary values */
	private void setiv(){
		int i, j, k, m;
		double eta, peta, pxi, pzeta, xi, zeta;
		double ue_1jk[] = new double[5], ue_nx0jk[] = new double[5], ue_i1k[] = new double[5],
				ue_iny0k[] = new double[5], ue_ij1[] = new double[5], ue_ijnz[] = new double[5];
		for (k = 2; k <= nz - 1; k++){
			for (j = 2; j <= ny - 1; j++){
				zeta = ( (double)(k-1) ) / (nz-1);
				eta = ( (double)(j-1) ) / (ny0-1);
				for (i = 2; i <= nx - 1; i++){
					xi = ( (double)(i-1) ) / (nx0-1);
					exact(1,j,k,ue_1jk);
					exact(nx0,j,k,ue_nx0jk);
					exact(i,1,k,ue_i1k);
					exact(i,ny0,k,ue_iny0k);
					exact(i,j,1,ue_ij1);
					exact(i,j,nz,ue_ijnz);
					for (m = 0; m <= 4; m++){
						pxi = ( 1.0 - xi ) * ue_1jk[m] + xi * ue_nx0jk[m];
						peta = ( 1.0 - eta ) * ue_i1k[m] + eta * ue_iny0k[m];
						pzeta = ( 1.0 - zeta ) * ue_ij1[m] + zeta * ue_ijnz[m];
						u[5*ix(i,j,k)+m] = pxi + peta + pzeta - pxi * peta - peta * pzeta - pzeta * pxi + pxi * peta * pzeta;
					}
				}
			}
		}
	}

	/* The forcing function frct (and the exact solution in rsd) */
	private void erhs(){
		int i, j, k, m;
		double eta, q, tmp, u21, u21i, u21im1, u21j, u21jm1, u21k, u21km1, u31, u31i, u31im1, u31j, u31jm1, u31k, u31km1, u41, u41i, u41im1, u41j, u41jm1, u41k, u41km1, u51i, u51im1, u51j, u51jm1, u51k, u51km1, xi, zeta;
		for (k = 1; k <= nz; k++){
			for (j = 1; j <= ny; j++){
				for (i = 1; i <= nx; i++){
					for (m = 0; m <= 4; m++){
						frct[5*ix(i,j,k)+m] = 0.0;
					}
				}
			}
		}
		for (k = 1; k <= nz; k++){
			for (j = 1; j <= ny; j++){
				zeta = ( (double)(k-1) ) / ( nz - 1 );
				eta = ( (double)(j-1) ) / ( ny0 - 1 );
				for (i = 1; i <= nx; i++){
					xi = ( (double)(i-1) ) / ( nx0 - 1 );
					for (m = 0; m <= 4; m++){
						rsd[5*ix(i,j,k)+m] = ce[m][0] + (ce[m][1] + (ce[m][4] + (ce[m][7] + ce[m][10] * xi) * xi) * xi) * xi + (ce[m][2] + (ce[m][5] + (ce[m][8] + ce[m][11] * eta) * eta) * eta) * eta + (ce[m][3] + (ce[m][6] + (ce[m][9] + ce[m][12] * zeta) * zeta) * zeta) * zeta;
					}
				}
			}
		}
		for (k = 2; k <= nz - 1; k++){
			for (j = jst; j <= jend; j++){
				for (i = 1; i <= nx; i++){
					flux[0][i] = rsd[5*ix(i,j,k)+1];
					u21 = rsd[5*ix(i,j,k)+1] / rsd[5*ix(i,j,k)+0];
					q = 0.50 * ( rsd[5*ix(i,j,k)+1] * rsd[5*ix(i,j,k)+1] + rsd[5*ix(i,j,k)+2] * rsd[5*ix(i,j,k)+2] + rsd[5*ix(i,j,k)+3] * rsd[5*ix(i,j,k)+3] ) / rsd[5*ix(i,j,k)+0];
					flux[1][i] = rsd[5*ix(i,j,k)+1] * u21 + c2 * ( rsd[5*ix(i,j,k)+4] - q );
					flux[2][i] = rsd[5*ix(i,j,k)+2] * u21;
					flux[3][i] = rsd[5*ix(i,j,k)+3] * u21;
					flux[4][i] = ( c1 * rsd[5*ix(i,j,k)+4] - c2 * q ) * u21;
				}
				for (i = ist; i <= iend; i++){
					for (m = 0; m <= 4; m++){
						frct[5*ix(i,j,k)+m] = frct[5*ix(i,j,k)+m] - tx2 * ( flux[m][i+1] - flux[m][i-1] );
					}
				}
				for (i = ist; i <= nx; i++){
					tmp = 1.0 / rsd[5*ix(i,j,k)+0];
					u21i = tmp * rsd[5*ix(i,j,k)+1];
					u31i = tmp * rsd[5*ix(i,j,k)+2];
					u41i = tmp * rsd[5*ix(i,j,k)+3];
					u51i = tmp * rsd[5*ix(i,j,k)+4];
					tmp = 1.0 / rsd[5*ix(i-1,j,k)+0];
					u21im1 = tmp * rsd[5*ix(i-1,j,k)+1];
					u31im1 = tmp * rsd[5*ix(i-1,j,k)+2];
					u41im1 = tmp * rsd[5*ix(i-1,j,k)+3];
					u51im1 = tmp * rsd[5*ix(i-1,j,k)+4];
					flux[1][i] = (4.0/3.0) * tx3 * ( u21i - u21im1 );
					flux[2][i] = tx3 * ( u31i - u31im1 );
					flux[3][i] = tx3 * ( u41i - u41im1 );
					flux[4][i] = 0.50 * ( 1.0 - c1*c5 ) * tx3 * ( ( (u21i)*(u21i) + (u31i)*(u31i) + (u41i)*(u41i) ) - ( (u21im1)*(u21im1) + (u31im1)*(u31im1) + (u41im1)*(u41im1) ) ) + (1.0/6.0) * tx3 * ( (u21i)*(u21i) - (u21im1)*(u21im1) ) + c1 * c5 * tx3 * ( u51i - u51im1 );
				}
				for (i = ist; i <= iend; i++){
					frct[5*ix(i,j,k)+0] = frct[5*ix(i,j,k)+0] + dx1 * tx1 * ( rsd[5*ix(i-1,j,k)+0] - 2.0 * rsd[5*ix(i,j,k)+0] + rsd[5*ix(i+1,j,k)+0] );
					frct[5*ix(i,j,k)+1] = frct[5*ix(i,j,k)+1] + tx3 * c3 * c4 * ( flux[1][i+1] - flux[1][i] ) + dx2 * tx1 * ( rsd[5*ix(i-1,j,k)+1] - 2.0 * rsd[5*ix(i,j,k)+1] + rsd[5*ix(i+1,j,k)+1] );
					frct[5*ix(i,j,k)+2] = frct[5*ix(i,j,k)+2] + tx3 * c3 * c4 * ( flux[2][i+1] - flux[2][i] ) + dx3 * tx1 * ( rsd[5*ix(i-1,j,k)+2] - 2.0 * rsd[5*ix(i,j,k)+2] + rsd[5*ix(i+1,j,k)+2] );
					frct[5*ix(i,j,k)+3] = frct[5*ix(i,j,k)+3] + tx3 * c3 * c4 * ( flux[3][i+1] - flux[3][i] ) + dx4 * tx1 * ( rsd[5*ix(i-1,j,k)+3] - 2.0 * rsd[5*ix(i,j,k)+3] + rsd[5*ix(i+1,j,k)+3] );
					frct[5*ix(i,j,k)+4] = frct[5*ix(i,j,k)+4] + tx3 * c3 * c4 * ( flux[4][i+1] - flux[4][i] ) + dx5 * tx1 * ( rsd[5*ix(i-1,j,k)+4] - 2.0 * rsd[5*ix(i,j,k)+4] + rsd[5*ix(i+1,j,k)+4] );
				}
				for (m = 0; m <= 4; m++){
					frct[5*ix(2,j,k)+m] = frct[5*ix(2,j,k)+m] - dssp * ( + 5.0 * rsd[5*ix(2,j,k)+m] - 4.0 * rsd[5*ix(3,j,k)+m] + rsd[5*ix(4,j,k)+m] );
					frct[5*ix(3,j,k)+m] = frct[5*ix(3,j,k)+m] - dssp * ( - 4.0 * rsd[5*ix(2,j,k)+m] + 6.0 * rsd[5*ix(3,j,k)+m] - 4.0 * rsd[5*ix(4,j,k)+m] + rsd[5*ix(5,j,k)+m] );
				}
				for (i = 4; i <= nx - 3; i++){
					for (m = 0; m <= 4; m++){
						frct[5*ix(i,j,k)+m] = frct[5*ix(i,j,k)+m] - dssp * ( rsd[5*ix(i-2,j,k)+m] - 4.0 * rsd[5*ix(i-1,j,k)+m] + 6.0 * rsd[5*ix(i,j,k)+m] - 4.0 * rsd[5*ix(i+1,j,k)+m] + rsd[5*ix(i+2,j,k)+m] );
					}
				}
				for (m = 0; m <= 4; m++){
					frct[5*ix(nx-2,j,k)+m] = frct[5*ix(nx-2,j,k)+m] - dssp * ( rsd[5*ix(nx-4,j,k)+m] - 4.0 * rsd[5*ix(nx-3,j,k)+m] + 6.0 * rsd[5*ix(nx-2,j,k)+m] - 4.0 * rsd[5*ix(nx-1,j,k)+m] );
					frct[5*ix(nx-1,j,k)+m] = frct[5*ix(nx-1,j,k)+m] - dssp * ( rsd[5*ix(nx-3,j,k)+m] - 4.0 * rsd[5*ix(nx-2,j,k)+m] + 5.0 * rsd[5*ix(nx-1,j,k)+m] );
				}
			}
		}
		for (k = 2; k <= nz - 1; k++){
			for (i = ist; i <= iend; i++){
				for (j = 1; j <= ny; j++){
					flux[0][j] = rsd[5*ix(i,j,k)+2];
					u31 = rsd[5*ix(i,j,k)+2] / rsd[5*ix(i,j,k)+0];
					q = 0.50 * ( rsd[5*ix(i,j,k)+1] * rsd[5*ix(i,j,k)+1] + rsd[5*ix(i,j,k)+2] * rsd[5*ix(i,j,k)+2] + rsd[5*ix(i,j,k)+3] * rsd[5*ix(i,j,k)+3] ) / rsd[5*ix(i,j,k)+0];
					flux[1][j] = rsd[5*ix(i,j,k)+1] * u31;
					flux[2][j] = rsd[5*ix(i,j,k)+2] * u31 + c2 * ( rsd[5*ix(i,j,k)+4] - q );
					flux[3][j] = rsd[5*ix(i,j,k)+3] * u31;
					flux[4][j] = ( c1 * rsd[5*ix(i,j,k)+4] - c2 * q ) * u31;
				}
				for (j = jst; j <= jend; j++){
					for (m = 0; m <= 4; m++){
						frct[5*ix(i,j,k)+m] = frct[5*ix(i,j,k)+m] - ty2 * ( flux[m][j+1] - flux[m][j-1] );
					}
				}
				for (j = jst; j <= ny; j++){
					tmp = 1.0 / rsd[5*ix(i,j,k)+0];
					u21j = tmp * rsd[5*ix(i,j,k)+1];
					u31j = tmp * rsd[5*ix(i,j,k)+2];
					u41j = tmp * rsd[5*ix(i,j,k)+3];
					u51j = tmp * rsd[5*ix(i,j,k)+4];
					tmp = 1.0 / rsd[5*ix(i,j-1,k)+0];
					u21jm1 = tmp * rsd[5*ix(i,j-1,k)+1];
					u31jm1 = tmp * rsd[5*ix(i,j-1,k)+2];
					u41jm1 = tmp * rsd[5*ix(i,j-1,k)+3];
					u51jm1 = tmp * rsd[5*ix(i,j-1,k)+4];
					flux[1][j] = ty3 * ( u21j - u21jm1 );
					flux[2][j] = (4.0/3.0) * ty3 * ( u31j - u31jm1 );
					flux[3][j] = ty3 * ( u41j - u41jm1 );
					flux[4][j] = 0.50 * ( 1.0 - c1*c5 ) * ty3 * ( ( (u21j)*(u21j) + (u31j)*(u31j) + (u41j)*(u41j) ) - ( (u21jm1)*(u21jm1) + (u31jm1)*(u31jm1) + (u41jm1)*(u41jm1) ) ) + (1.0/6.0) * ty3 * ( (u31j)*(u31j) - (u31jm1)*(u31jm1) ) + c1 * c5 * ty3 * ( u51j - u51jm1 );
				}
				for (j = jst; j <= jend; j++){
					frct[5*ix(i,j,k)+0] = frct[5*ix(i,j,k)+0] + dy1 * ty1 * ( rsd[5*ix(i,j-1,k)+0] - 2.0 * rsd[5*ix(i,j,k)+0] + rsd[5*ix(i,j+1,k)+0] );
					frct[5*ix(i,j,k)+1] = frct[5*ix(i,j,k)+1] + ty3 * c3 * c4 * ( flux[1][j+1] - flux[1][j] ) + dy2 * ty1 * ( rsd[5*ix(i,j-1,k)+1] - 2.0 * rsd[5*ix(i,j,k)+1] + rsd[5*ix(i,j+1,k)+1] );
					frct[5*ix(i,j,k)+2] = frct[5*ix(i,j,k)+2] + ty3 * c3 * c4 * ( flux[2][j+1] - flux[2][j] ) + dy3 * ty1 * ( rsd[5*ix(i,j-1,k)+2] - 2.0 * rsd[5*ix(i,j,k)+2] + rsd[5*ix(i,j+1,k)+2] );
					frct[5*ix(i,j,k)+3] = frct[5*ix(i,j,k)+3] + ty3 * c3 * c4 * ( flux[3][j+1] - flux[3][j] ) + dy4 * ty1 * ( rsd[5*ix(i,j-1,k)+3] - 2.0 * rsd[5*ix(i,j,k)+3] + rsd[5*ix(i,j+1,k)+3] );
					frct[5*ix(i,j,k)+4] = frct[5*ix(i,j,k)+4] + ty3 * c3 * c4 * ( flux[4][j+1] - flux[4][j] ) + dy5 * ty1 * ( rsd[5*ix(i,j-1,k)+4] - 2.0 * rsd[5*ix(i,j,k)+4] + rsd[5*ix(i,j+1,k)+4] );
				}
				for (m = 0; m <= 4; m++){
					frct[5*ix(i,2,k)+m] = frct[5*ix(i,2,k)+m] - dssp * ( + 5.0 * rsd[5*ix(i,2,k)+m] - 4.0 * rsd[5*ix(i,3,k)+m] + rsd[5*ix(i,4,k)+m] );
					frct[5*ix(i,3,k)+m] = frct[5*ix(i,3,k)+m] - dssp * ( - 4.0 * rsd[5*ix(i,2,k)+m] + 6.0 * rsd[5*ix(i,3,k)+m] - 4.0 * rsd[5*ix(i,4,k)+m] + rsd[5*ix(i,5,k)+m] );
				}
				for (j = 4; j <= ny - 3; j++){
					for (m = 0; m <= 4; m++){
						frct[5*ix(i,j,k)+m] = frct[5*ix(i,j,k)+m] - dssp * ( rsd[5*ix(i,j-2,k)+m] - 4.0 * rsd[5*ix(i,j-1,k)+m] + 6.0 * rsd[5*ix(i,j,k)+m] - 4.0 * rsd[5*ix(i,j+1,k)+m] + rsd[5*ix(i,j+2,k)+m] );
					}
				}
				for (m = 0; m <= 4; m++){
					frct[5*ix(i,ny-2,k)+m] = frct[5*ix(i,ny-2,k)+m] - dssp * ( rsd[5*ix(i,ny-4,k)+m] - 4.0 * rsd[5*ix(i,ny-3,k)+m] + 6.0 * rsd[5*ix(i,ny-2,k)+m] - 4.0 * rsd[5*ix(i,ny-1,k)+m] );
					frct[5*ix(i,ny-1,k)+m] = frct[5*ix(i,ny-1,k)+m] - dssp * ( rsd[5*ix(i,ny-3,k)+m] - 4.0 * rsd[5*ix(i,ny-2,k)+m] + 5.0 * rsd[5*ix(i,ny-1,k)+m] );
				}
			}
		}
		for (j = jst; j <= jend; j++){
			for (i = ist; i <= iend; i++){
				for (k = 1; k <= nz; k++){
					flux[0][k] = rsd[5*ix(i,j,k)+3];
					u41 = rsd[5*ix(i,j,k)+3] / rsd[5*ix(i,j,k)+0];
					q = 0.50 * ( rsd[5*ix(i,j,k)+1] * rsd[5*ix(i,j,k)+1] + rsd[5*ix(i,j,k)+2] * rsd[5*ix(i,j,k)+2] + rsd[5*ix(i,j,k)+3] * rsd[5*ix(i,j,k)+3] ) / rsd[5*ix(i,j,k)+0];
					flux[1][k] = rsd[5*ix(i,j,k)+1] * u41;
					flux[2][k] = rsd[5*ix(i,j,k)+2] * u41;
					flux[3][k] = rsd[5*ix(i,j,k)+3] * u41 + c2 * ( rsd[5*ix(i,j,k)+4] - q );
					flux[4][k] = ( c1 * rsd[5*ix(i,j,k)+4] - c2 * q ) * u41;
				}
				for (k = 2; k <= nz - 1; k++){
					for (m = 0; m <= 4; m++){
						frct[5*ix(i,j,k)+m] = frct[5*ix(i,j,k)+m] - tz2 * ( flux[m][k+1] - flux[m][k-1] );
					}
				}
				for (k = 2; k <= nz; k++){
					tmp = 1.0 / rsd[5*ix(i,j,k)+0];
					u21k = tmp * rsd[5*ix(i,j,k)+1];
					u31k = tmp * rsd[5*ix(i,j,k)+2];
					u41k = tmp * rsd[5*ix(i,j,k)+3];
					u51k = tmp * rsd[5*ix(i,j,k)+4];
					tmp = 1.0 / rsd[5*ix(i,j,k-1)+0];
					u21km1 = tmp * rsd[5*ix(i,j,k-1)+1];
					u31km1 = tmp * rsd[5*ix(i,j,k-1)+2];
					u41km1 = tmp * rsd[5*ix(i,j,k-1)+3];
					u51km1 = tmp * rsd[5*ix(i,j,k-1)+4];
					flux[1][k] = tz3 * ( u21k - u21km1 );
					flux[2][k] = tz3 * ( u31k - u31km1 );
					flux[3][k] = (4.0/3.0) * tz3 * ( u41k - u41km1 );
					flux[4][k] = 0.50 * ( 1.0 - c1*c5 ) * tz3 * ( ( (u21k)*(u21k) + (u31k)*(u31k) + (u41k)*(u41k) ) - ( (u21km1)*(u21km1) + (u31km1)*(u31km1) + (u41km1)*(u41km1) ) ) + (1.0/6.0) * tz3 * ( (u41k)*(u41k) - (u41km1)*(u41km1) ) + c1 * c5 * tz3 * ( u51k - u51km1 );
				}
				for (k = 2; k <= nz - 1; k++){
					frct[5*ix(i,j,k)+0] = frct[5*ix(i,j,k)+0] + dz1 * tz1 * ( rsd[5*ix(i,j,k+1)+0] - 2.0 * rsd[5*ix(i,j,k)+0] + rsd[5*ix(i,j,k-1)+0] );
					frct[5*ix(i,j,k)+1] = frct[5*ix(i,j,k)+1] + tz3 * c3 * c4 * ( flux[1][k+1] - flux[1][k] ) + dz2 * tz1 * ( rsd[5*ix(i,j,k+1)+1] - 2.0 * rsd[5*ix(i,j,k)+1] + rsd[5*ix(i,j,k-1)+1] );
					frct[5*ix(i,j,k)+2] = frct[5*ix(i,j,k)+2] + tz3 * c3 * c4 * ( flux[2][k+1] - flux[2][k] ) + dz3 * tz1 * ( rsd[5*ix(i,j,k+1)+2] - 2.0 * rsd[5*ix(i,j,k)+2] + rsd[5*ix(i,j,k-1)+2] );
					frct[5*ix(i,j,k)+3] = frct[5*ix(i,j,k)+3] + tz3 * c3 * c4 * ( flux[3][k+1] - flux[3][k] ) + dz4 * tz1 * ( rsd[5*ix(i,j,k+1)+3] - 2.0 * rsd[5*ix(i,j,k)+3] + rsd[5*ix(i,j,k-1)+3] );
					frct[5*ix(i,j,k)+4] = frct[5*ix(i,j,k)+4] + tz3 * c3 * c4 * ( flux[4][k+1] - flux[4][k] ) + dz5 * tz1 * ( rsd[5*ix(i,j,k+1)+4] - 2.0 * rsd[5*ix(i,j,k)+4] + rsd[5*ix(i,j,k-1)+4] );
				}
				for (m = 0; m <= 4; m++){
					frct[5*ix(i,j,2)+m] = frct[5*ix(i,j,2)+m] - dssp * ( + 5.0 * rsd[5*ix(i,j,2)+m] - 4.0 * rsd[5*ix(i,j,3)+m] + rsd[5*ix(i,j,4)+m] );
					frct[5*ix(i,j,3)+m] = frct[5*ix(i,j,3)+m] - dssp * (- 4.0 * rsd[5*ix(i,j,2)+m] + 6.0 * rsd[5*ix(i,j,3)+m] - 4.0 * rsd[5*ix(i,j,4)+m] + rsd[5*ix(i,j,5)+m] );
				}
				for (k = 4; k <= nz - 3; k++){
					for (m = 0; m <= 4; m++){
						frct[5*ix(i,j,k)+m] = frct[5*ix(i,j,k)+m] - dssp * ( rsd[5*ix(i,j,k-2)+m] - 4.0 * rsd[5*ix(i,j,k-1)+m] + 6.0 * rsd[5*ix(i,j,k)+m] - 4.0 * rsd[5*ix(i,j,k+1)+m] + rsd[5*ix(i,j,k+2)+m] );
					}
				}
				for (m = 0; m <= 4; m++){
					frct[5*ix(i,j,nz-2)+m] = frct[5*ix(i,j,nz-2)+m] - dssp * ( rsd[5*ix(i,j,nz-4)+m] - 4.0 * rsd[5*ix(i,j,nz-3)+m] + 6.0 * rsd[5*ix(i,j,nz-2)+m] - 4.0 * rsd[5*ix(i,j,nz-1)+m] );
					frct[5*ix(i,j,nz-1)+m] = frct[5*ix(i,j,nz-1)+m] - dssp * ( rsd[5*ix(i,j,nz-3)+m] - 4.0 * rsd[5*ix(i,j,nz-2)+m] + 5.0 * rsd[5*ix(i,j,nz-1)+m] );
				}
			}
		}
	}

	private void ssor(int niter){

		int i, j, k, m;
		double tmp, tmp2;
		double delunm[] = new double[5];

		/* begin pseudo-time stepping iterations */
		tmp = 1.0 / ( omega * ( 2.0 - omega ) );

		for(i = 1; i<=T_LAST; i++)
			timer.reset(i);

		/* compute the steady-state residuals */
		rhs();

		/* compute the L2 norms of newton iteration residuals */
		l2norm(rsdnm);

		for(i = 1; i<=T_LAST; i++)
			timer.reset(i);

		timer.start(1);

		/* the timestep loop */
		for(int istep = 1; istep<=niter; istep++){

			if(istep%20 == 0 || istep == itmax || istep == 1){
				if(niter > 1)
					System.out.println(" Time step "+istep);
			}

			/* perform SSOR iteration */
			if (detailedTiming) timer.start(T_RHS);
			tmp2 = dt;
			for(k = 2; k<=nz-1; k++){
				for(j = jst; j<=jend; j++){
					for(i = ist; i<=iend; i++){
						int p = 5*ix(i,j,k);
						for(m = 0; m<5; m++){
							rsd[p+m] = tmp2 * rsd[p+m];
						}
					}
				}
			}
			if (detailedTiming) timer.stop(T_RHS);

			/* the lower triangular part of the SSOR */
			if (detailedTiming) timer.start(T_BLTS);
			for(k = 2; k<=nz-1; k++){
				for(j = jst; j<=jend; j++){
					/* form the lower triangular part of the jacobian matrix */
					jacld(j, k);
					/* perform the lower triangular solution */
					blts(a, b, c, d, j, k);
				}
			}
			if (detailedTiming) timer.stop(T_BLTS);

			/* the upper triangular part of the SSOR */
			if (detailedTiming) timer.start(T_BUTS);
			for(k = nz-1; k>=2; k--){
				for(j = jend; j>=jst; j--){
					/* form the strictly upper triangular part of the jacobian matrix */
					jacu(j, k);
					/* perform the upper triangular solution */
					buts(d, a, b, c, j, k);
				}
			}
			if (detailedTiming) timer.stop(T_BUTS);

			/* update the variables */
			if (detailedTiming) timer.start(T_ADD);
			tmp2 = tmp;
			for(k = 2; k<=nz-1; k++){
				for(j = jst; j<=jend; j++){
					for(i = ist; i<=iend; i++){
						int p = 5*ix(i,j,k);
						for(m = 0; m<5; m++){
							u[p+m] = u[p+m]
									+ tmp2 * rsd[p+m];
						}
					}
				}
			}
			if (detailedTiming) timer.stop(T_ADD);

			/* compute the max-norms of newton iteration corrections */
			if(istep%inorm == 0){
				if (detailedTiming) timer.start(T_L2NORM);
				l2norm(delunm);
				if (detailedTiming) timer.stop(T_L2NORM);
			}

			/* compute the steady-state residuals */
			rhs();

			/* compute the max-norms of newton iteration residuals */
			if(istep%inorm == 0 || istep == itmax){
				if (detailedTiming) timer.start(T_L2NORM);
				l2norm(rsdnm);
				if (detailedTiming) timer.stop(T_L2NORM);
			}

			/* check the newton-iteration residuals against the tolerance levels */
			if(rsdnm[0] < tolrsd[0] &&
					rsdnm[1] < tolrsd[1] &&
					rsdnm[2] < tolrsd[2] &&
					rsdnm[3] < tolrsd[3] &&
					rsdnm[4] < tolrsd[4]){
				System.out.println(" convergence was achieved after "+istep+" pseudo-time steps");
				break;
			}
		}

		timer.stop(1);
		maxtime = timer.read(1);
	}

	/* RMS-norm of rsd over the interior points */
	private void l2norm(double sum[]){
		int i, j, k, m;
		for (m = 0; m <= 4; m++){
			sum[m] = 0.0;
		}
		for (k = 2; k <= nz0-1; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					for (m = 0; m <= 4; m++){
						sum[m] = sum[m] + rsd[5*ix(i,j,k)+m]*rsd[5*ix(i,j,k)+m];
					}
				}
			}
		}
		for (m = 0; m <= 4; m++){
			sum[m] = Math.sqrt( sum[m] / ( (double)(nx0-2)*(ny0-2)*(nz0-2) ) );
		}
	}

	/* Steady-state residual of the current u (in rsd) */
	private void rhs(){
		int i, j, k;
		if (detailedTiming) timer.start(T_RHS);
		rhsInit();
		if (detailedTiming) timer.start(T_RHSX);
		for (k = 2; k <= nz - 1; k++){
			for (j = jst; j <= jend; j++){
				rhsX(j, k);
			}
		}
		if (detailedTiming) timer.stop(T_RHSX);
		if (detailedTiming) timer.start(T_RHSY);
		for (k = 2; k <= nz - 1; k++){
			for (i = ist; i <= iend; i++){
				rhsY(i, k);
			}
		}
		rhsYDiss();
		if (detailedTiming) timer.stop(T_RHSY);
		if (detailedTiming) timer.start(T_RHSZ);
		for (j = jst; j <= jend; j++){
			for (i = ist; i <= iend; i++){
				rhsZ(i, j);
			}
		}
		if (detailedTiming) timer.stop(T_RHSZ);
		if (detailedTiming) timer.stop(T_RHS);
	}

	/* rsd = -frct, rhoI y qs */
	private void rhsInit(){
		int i, j, k, m;
		double q, tmp, u21, u21i, u21im1, u21j, u21jm1, u21k, u21km1, u31, u31i, u31im1, u31j, u31jm1, u31k, u31km1, u41, u41i, u41im1, u41j, u41jm1, u41k, u41km1, u51i, u51im1, u51j, u51jm1, u51k, u51km1;
		for (k = 1; k <= nz; k++){
			for (j = 1; j <= ny; j++){
				for (i = 1; i <= nx; i++){
					for (m = 0; m <= 4; m++){
						rsd[5*ix(i,j,k)+m] = - frct[5*ix(i,j,k)+m];
					}
					tmp = 1.0 / u[5*ix(i,j,k)+0];
					rhoI[ix(i,j,k)] = tmp;
					qs[ix(i,j,k)] = 0.50 * ( u[5*ix(i,j,k)+1] * u[5*ix(i,j,k)+1] + u[5*ix(i,j,k)+2] * u[5*ix(i,j,k)+2] + u[5*ix(i,j,k)+3] * u[5*ix(i,j,k)+3] ) * tmp;
				}
			}
		}
	}

	/* Fluxes in the xi direction of line (j,k) */
	private void rhsX(int j, int k){
		int i, m;
		double q, tmp, u21, u21i, u21im1, u21j, u21jm1, u21k, u21km1, u31, u31i, u31im1, u31j, u31jm1, u31k, u31km1, u41, u41i, u41im1, u41j, u41jm1, u41k, u41km1, u51i, u51im1, u51j, u51jm1, u51k, u51km1;
		for (i = 1; i <= nx; i++){
			flux[0][i] = u[5*ix(i,j,k)+1];
			u21 = u[5*ix(i,j,k)+1] * rhoI[ix(i,j,k)];
			q = qs[ix(i,j,k)];
			flux[1][i] = u[5*ix(i,j,k)+1] * u21 + c2 * ( u[5*ix(i,j,k)+4] - q );
			flux[2][i] = u[5*ix(i,j,k)+2] * u21;
			flux[3][i] = u[5*ix(i,j,k)+3] * u21;
			flux[4][i] = ( c1 * u[5*ix(i,j,k)+4] - c2 * q ) * u21;
		}
		for (i = ist; i <= iend; i++){
			for (m = 0; m <= 4; m++){
				rsd[5*ix(i,j,k)+m] = rsd[5*ix(i,j,k)+m] - tx2 * ( flux[m][i+1] - flux[m][i-1] );
			}
		}
		for (i = ist; i <= nx; i++){
			tmp = rhoI[ix(i,j,k)];
			u21i = tmp * u[5*ix(i,j,k)+1];
			u31i = tmp * u[5*ix(i,j,k)+2];
			u41i = tmp * u[5*ix(i,j,k)+3];
			u51i = tmp * u[5*ix(i,j,k)+4];
			tmp = rhoI[ix(i-1,j,k)];
			u21im1 = tmp * u[5*ix(i-1,j,k)+1];
			u31im1 = tmp * u[5*ix(i-1,j,k)+2];
			u41im1 = tmp * u[5*ix(i-1,j,k)+3];
			u51im1 = tmp * u[5*ix(i-1,j,k)+4];
			flux[1][i] = (4.0/3.0) * tx3 * (u21i-u21im1);
			flux[2][i] = tx3 * ( u31i - u31im1 );
			flux[3][i] = tx3 * ( u41i - u41im1 );
			flux[4][i] = 0.50 * ( 1.0 - c1*c5 ) * tx3 * ( ( (u21i)*(u21i) + (u31i)*(u31i) + (u41i)*(u41i) ) - ( (u21im1)*(u21im1) + (u31im1)*(u31im1) + (u41im1)*(u41im1) ) ) + (1.0/6.0) * tx3 * ( (u21i)*(u21i) - (u21im1)*(u21im1) ) + c1 * c5 * tx3 * ( u51i - u51im1 );
		}
		for (i = ist; i <= iend; i++){
			rsd[5*ix(i,j,k)+0] = rsd[5*ix(i,j,k)+0] + dx1 * tx1 * ( u[5*ix(i-1,j,k)+0] - 2.0 * u[5*ix(i,j,k)+0] + u[5*ix(i+1,j,k)+0] );
			rsd[5*ix(i,j,k)+1] = rsd[5*ix(i,j,k)+1] + tx3 * c3 * c4 * ( flux[1][i+1] - flux[1][i] ) + dx2 * tx1 * ( u[5*ix(i-1,j,k)+1] - 2.0 * u[5*ix(i,j,k)+1] + u[5*ix(i+1,j,k)+1] );
			rsd[5*ix(i,j,k)+2] = rsd[5*ix(i,j,k)+2] + tx3 * c3 * c4 * ( flux[2][i+1] - flux[2][i] ) + dx3 * tx1 * ( u[5*ix(i-1,j,k)+2] - 2.0 * u[5*ix(i,j,k)+2] + u[5*ix(i+1,j,k)+2] );
			rsd[5*ix(i,j,k)+3] = rsd[5*ix(i,j,k)+3] + tx3 * c3 * c4 * ( flux[3][i+1] - flux[3][i] ) + dx4 * tx1 * ( u[5*ix(i-1,j,k)+3] - 2.0 * u[5*ix(i,j,k)+3] + u[5*ix(i+1,j,k)+3] );
			rsd[5*ix(i,j,k)+4] = rsd[5*ix(i,j,k)+4] + tx3 * c3 * c4 * ( flux[4][i+1] - flux[4][i] ) + dx5 * tx1 * ( u[5*ix(i-1,j,k)+4] - 2.0 * u[5*ix(i,j,k)+4] + u[5*ix(i+1,j,k)+4] );
		}
		for (m = 0; m <= 4; m++){
			rsd[5*ix(2,j,k)+m] = rsd[5*ix(2,j,k)+m] - dssp * ( + 5.0 * u[5*ix(2,j,k)+m] - 4.0 * u[5*ix(3,j,k)+m] + u[5*ix(4,j,k)+m] );
			rsd[5*ix(3,j,k)+m] = rsd[5*ix(3,j,k)+m] - dssp * ( - 4.0 * u[5*ix(2,j,k)+m] + 6.0 * u[5*ix(3,j,k)+m] - 4.0 * u[5*ix(4,j,k)+m] + u[5*ix(5,j,k)+m] );
		}
		for (i = 4; i <= nx - 3; i++){
			for (m = 0; m <= 4; m++){
				rsd[5*ix(i,j,k)+m] = rsd[5*ix(i,j,k)+m] - dssp * ( u[5*ix(i-2,j,k)+m] - 4.0 * u[5*ix(i-1,j,k)+m] + 6.0 * u[5*ix(i,j,k)+m] - 4.0 * u[5*ix(i+1,j,k)+m] + u[5*ix(i+2,j,k)+m] );
			}
		}
		for (m = 0; m <= 4; m++){
			rsd[5*ix(nx-2,j,k)+m] = rsd[5*ix(nx-2,j,k)+m] - dssp * ( u[5*ix(nx-4,j,k)+m] - 4.0 * u[5*ix(nx-3,j,k)+m] + 6.0 * u[5*ix(nx-2,j,k)+m] - 4.0 * u[5*ix(nx-1,j,k)+m] );
			rsd[5*ix(nx-1,j,k)+m] = rsd[5*ix(nx-1,j,k)+m] - dssp * ( u[5*ix(nx-3,j,k)+m] - 4.0 * u[5*ix(nx-2,j,k)+m] + 5.0 * u[5*ix(nx-1,j,k)+m] );
		}
	}

	/* Fluxes in the eta direction of line (i,k) */
	private void rhsY(int i, int k){
		int j, m;
		double q, tmp, u21, u21i, u21im1, u21j, u21jm1, u21k, u21km1, u31, u31i, u31im1, u31j, u31jm1, u31k, u31km1, u41, u41i, u41im1, u41j, u41jm1, u41k, u41km1, u51i, u51im1, u51j, u51jm1, u51k, u51km1;
		for (j = 1; j <= ny; j++){
			flux[0][j] = u[5*ix(i,j,k)+2];
			u31 = u[5*ix(i,j,k)+2] * rhoI[ix(i,j,k)];
			q = qs[ix(i,j,k)];
			flux[1][j] = u[5*ix(i,j,k)+1] * u31;
			flux[2][j] = u[5*ix(i,j,k)+2] * u31 + c2 * (u[5*ix(i,j,k)+4]-q);
			flux[3][j] = u[5*ix(i,j,k)+3] * u31;
			flux[4][j] = ( c1 * u[5*ix(i,j,k)+4] - c2 * q ) * u31;
		}
		for (j = jst; j <= jend; j++){
			for (m = 0; m <= 4; m++){
				rsd[5*ix(i,j,k)+m] = rsd[5*ix(i,j,k)+m] - ty2 * ( flux[m][j+1] - flux[m][j-1] );
			}
		}
		for (j = jst; j <= ny; j++){
			tmp = rhoI[ix(i,j,k)];
			u21j = tmp * u[5*ix(i,j,k)+1];
			u31j = tmp * u[5*ix(i,j,k)+2];
			u41j = tmp * u[5*ix(i,j,k)+3];
			u51j = tmp * u[5*ix(i,j,k)+4];
			tmp = rhoI[ix(i,j-1,k)];
			u21jm1 = tmp * u[5*ix(i,j-1,k)+1];
			u31jm1 = tmp * u[5*ix(i,j-1,k)+2];
			u41jm1 = tmp * u[5*ix(i,j-1,k)+3];
			u51jm1 = tmp * u[5*ix(i,j-1,k)+4];
			flux[1][j] = ty3 * ( u21j - u21jm1 );
			flux[2][j] = (4.0/3.0) * ty3 * (u31j-u31jm1);
			flux[3][j] = ty3 * ( u41j - u41jm1 );
			flux[4][j] = 0.50 * ( 1.0 - c1*c5 ) * ty3 * ( ( (u21j)*(u21j) + (u31j)*(u31j) + (u41j)*(u41j) ) - ( (u21jm1)*(u21jm1) + (u31jm1)*(u31jm1) + (u41jm1)*(u41jm1) ) ) + (1.0/6.0) * ty3 * ( (u31j)*(u31j) - (u31jm1)*(u31jm1) ) + c1 * c5 * ty3 * ( u51j - u51jm1 );
		}
		for (j = jst; j <= jend; j++){
			rsd[5*ix(i,j,k)+0] = rsd[5*ix(i,j,k)+0] + dy1 * ty1 * ( u[5*ix(i,j-1,k)+0] - 2.0 * u[5*ix(i,j,k)+0] + u[5*ix(i,j+1,k)+0] );
			rsd[5*ix(i,j,k)+1] = rsd[5*ix(i,j,k)+1] + ty3 * c3 * c4 * ( flux[1][j+1] - flux[1][j] ) + dy2 * ty1 * ( u[5*ix(i,j-1,k)+1] - 2.0 * u[5*ix(i,j,k)+1] + u[5*ix(i,j+1,k)+1] );
			rsd[5*ix(i,j,k)+2] = rsd[5*ix(i,j,k)+2] + ty3 * c3 * c4 * ( flux[2][j+1] - flux[2][j] ) + dy3 * ty1 * ( u[5*ix(i,j-1,k)+2] - 2.0 * u[5*ix(i,j,k)+2] + u[5*ix(i,j+1,k)+2] );
			rsd[5*ix(i,j,k)+3] = rsd[5*ix(i,j,k)+3] + ty3 * c3 * c4 * ( flux[3][j+1] - flux[3][j] ) + dy4 * ty1 * ( u[5*ix(i,j-1,k)+3] - 2.0 * u[5*ix(i,j,k)+3] + u[5*ix(i,j+1,k)+3] );
			rsd[5*ix(i,j,k)+4] = rsd[5*ix(i,j,k)+4] + ty3 * c3 * c4 * ( flux[4][j+1] - flux[4][j] ) + dy5 * ty1 * ( u[5*ix(i,j-1,k)+4] - 2.0 * u[5*ix(i,j,k)+4] + u[5*ix(i,j+1,k)+4] );
		}
	}

	/* Fourth-order dissipation in the eta direction */
	private void rhsYDiss(){
		int i, j, k, m;
		double q, tmp, u21, u21i, u21im1, u21j, u21jm1, u21k, u21km1, u31, u31i, u31im1, u31j, u31jm1, u31k, u31km1, u41, u41i, u41im1, u41j, u41jm1, u41k, u41km1, u51i, u51im1, u51j, u51jm1, u51k, u51km1;
		for (k = 2; k <= nz - 1; k++){
			for (j = jst; j <= jend; j++){
				if (j == 2){
					for (i = ist; i <= iend; i++){
						for (m = 0; m <= 4; m++){
							rsd[5*ix(i,2,k)+m] = rsd[5*ix(i,2,k)+m] - dssp * ( + 5.0 * u[5*ix(i,2,k)+m] - 4.0 * u[5*ix(i,3,k)+m] + u[5*ix(i,4,k)+m] );
						}
					}
				} else if (j == 3){
					for (i = ist; i <= iend; i++){
						for (m = 0; m <= 4; m++){
							rsd[5*ix(i,3,k)+m] = rsd[5*ix(i,3,k)+m] - dssp * ( - 4.0 * u[5*ix(i,2,k)+m] + 6.0 * u[5*ix(i,3,k)+m] - 4.0 * u[5*ix(i,4,k)+m] + u[5*ix(i,5,k)+m] );
						}
					}
				} else if (j == ny-2){
					for (i = ist; i <= iend; i++){
						for (m = 0; m <= 4; m++){
							rsd[5*ix(i,ny-2,k)+m] = rsd[5*ix(i,ny-2,k)+m] - dssp * ( u[5*ix(i,ny-4,k)+m] - 4.0 * u[5*ix(i,ny-3,k)+m] + 6.0 * u[5*ix(i,ny-2,k)+m] - 4.0 * u[5*ix(i,ny-1,k)+m] );
						}
					}
				} else if (j == ny-1){
					for (i = ist; i <= iend; i++){
						for (m = 0; m <= 4; m++){
							rsd[5*ix(i,ny-1,k)+m] = rsd[5*ix(i,ny-1,k)+m] - dssp * ( u[5*ix(i,ny-3,k)+m] - 4.0 * u[5*ix(i,ny-2,k)+m] + 5.0 * u[5*ix(i,ny-1,k)+m] );
						}
					}
				} else {
					for (i = ist; i <= iend; i++){
						for (m = 0; m <= 4; m++){
							rsd[5*ix(i,j,k)+m] = rsd[5*ix(i,j,k)+m] - dssp * ( u[5*ix(i,j-2,k)+m] - 4.0 * u[5*ix(i,j-1,k)+m] + 6.0 * u[5*ix(i,j,k)+m] - 4.0 * u[5*ix(i,j+1,k)+m] + u[5*ix(i,j+2,k)+m] );
						}
					}
				}
			}
		}
	}

	/* Fluxes in the zeta direction of column (i,j) */
	private void rhsZ(int i, int j){
		int k, m;
		double q, tmp, u21, u21i, u21im1, u21j, u21jm1, u21k, u21km1, u31, u31i, u31im1, u31j, u31jm1, u31k, u31km1, u41, u41i, u41im1, u41j, u41jm1, u41k, u41km1, u51i, u51im1, u51j, u51jm1, u51k, u51km1;
		for (k = 1; k <= nz; k++){
			utmp[0][k] = u[5*ix(i,j,k)+0];
			utmp[1][k] = u[5*ix(i,j,k)+1];
			utmp[2][k] = u[5*ix(i,j,k)+2];
			utmp[3][k] = u[5*ix(i,j,k)+3];
			utmp[4][k] = u[5*ix(i,j,k)+4];
			utmp[5][k] = rhoI[ix(i,j,k)];
		}
		for (k = 1; k <= nz; k++){
			flux[0][k] = utmp[3][k];
			u41 = utmp[3][k] * utmp[5][k];
			q = qs[ix(i,j,k)];
			flux[1][k] = utmp[1][k] * u41;
			flux[2][k] = utmp[2][k] * u41;
			flux[3][k] = utmp[3][k] * u41 + c2 * (utmp[4][k]-q);
			flux[4][k] = ( c1 * utmp[4][k] - c2 * q ) * u41;
		}
		for (k = 2; k <= nz - 1; k++){
			for (m = 0; m <= 4; m++){
				rtmp[m][k] = rsd[5*ix(i,j,k)+m] - tz2 * ( flux[m][k+1] - flux[m][k-1] );
			}
		}
		for (k = 2; k <= nz; k++){
			tmp = utmp[5][k];
			u21k = tmp * utmp[1][k];
			u31k = tmp * utmp[2][k];
			u41k = tmp * utmp[3][k];
			u51k = tmp * utmp[4][k];
			tmp = utmp[5][k-1];
			u21km1 = tmp * utmp[1][k-1];
			u31km1 = tmp * utmp[2][k-1];
			u41km1 = tmp * utmp[3][k-1];
			u51km1 = tmp * utmp[4][k-1];
			flux[1][k] = tz3 * ( u21k - u21km1 );
			flux[2][k] = tz3 * ( u31k - u31km1 );
			flux[3][k] = (4.0/3.0) * tz3 * (u41k-u41km1);
			flux[4][k] = 0.50 * ( 1.0 - c1*c5 ) * tz3 * ( ( (u21k)*(u21k) + (u31k)*(u31k) + (u41k)*(u41k) ) - ( (u21km1)*(u21km1) + (u31km1)*(u31km1) + (u41km1)*(u41km1) ) ) + (1.0/6.0) * tz3 * ( (u41k)*(u41k) - (u41km1)*(u41km1) ) + c1 * c5 * tz3 * ( u51k - u51km1 );
		}
		for (k = 2; k <= nz - 1; k++){
			rtmp[0][k] = rtmp[0][k] + dz1 * tz1 * ( utmp[0][k-1] - 2.0 * utmp[0][k] + utmp[0][k+1] );
			rtmp[1][k] = rtmp[1][k] + tz3 * c3 * c4 * ( flux[1][k+1] - flux[1][k] ) + dz2 * tz1 * ( utmp[1][k-1] - 2.0 * utmp[1][k] + utmp[1][k+1] );
			rtmp[2][k] = rtmp[2][k] + tz3 * c3 * c4 * ( flux[2][k+1] - flux[2][k] ) + dz3 * tz1 * ( utmp[2][k-1] - 2.0 * utmp[2][k] + utmp[2][k+1] );
			rtmp[3][k] = rtmp[3][k] + tz3 * c3 * c4 * ( flux[3][k+1] - flux[3][k] ) + dz4 * tz1 * ( utmp[3][k-1] - 2.0 * utmp[3][k] + utmp[3][k+1] );
			rtmp[4][k] = rtmp[4][k] + tz3 * c3 * c4 * ( flux[4][k+1] - flux[4][k] ) + dz5 * tz1 * ( utmp[4][k-1] - 2.0 * utmp[4][k] + utmp[4][k+1] );
		}
		for (m = 0; m <= 4; m++){
			rsd[5*ix(i,j,2)+m] = rtmp[m][2] - dssp * ( + 5.0 * utmp[m][2] - 4.0 * utmp[m][3] + utmp[m][4] );
			rsd[5*ix(i,j,3)+m] = rtmp[m][3] - dssp * ( - 4.0 * utmp[m][2] + 6.0 * utmp[m][3] - 4.0 * utmp[m][4] + utmp[m][5] );
		}
		for (k = 4; k <= nz - 3; k++){
			for (m = 0; m <= 4; m++){
				rsd[5*ix(i,j,k)+m] = rtmp[m][k] - dssp * ( utmp[m][k-2] - 4.0 * utmp[m][k-1] + 6.0 * utmp[m][k] - 4.0 * utmp[m][k+1] + utmp[m][k+2] );
			}
		}
		for (m = 0; m <= 4; m++){
			rsd[5*ix(i,j,nz-2)+m] = rtmp[m][nz-2] - dssp * ( utmp[m][nz-4] - 4.0 * utmp[m][nz-3] + 6.0 * utmp[m][nz-2] - 4.0 * utmp[m][nz-1] );
			rsd[5*ix(i,j,nz-1)+m] = rtmp[m][nz-1] - dssp * ( utmp[m][nz-3] - 4.0 * utmp[m][nz-2] + 5.0 * utmp[m][nz-1] );
		}
	}

	/* Lower triangular part of the jacobian for the line j of the plane k: blocks d, a, b and c of each point */
	private void jacld(int j, int k){
		int i;
		double tmp1, tmp2, tmp3;
		double r43 = ( 4.0 / 3.0 );
		double c1345 = c1 * c3 * c4 * c5;
		double c34 = c3 * c4;
		/* fields in locals and indices without ix(); the operands of u, qs and rhoI are read once per point (same expressions) */
		final double[] u = this.u, qs = this.qs, rhoI = this.rhoI, a = this.a, b = this.b, c = this.c, d = this.d;
		final double dt = this.dt, tx1 = this.tx1, tx2 = this.tx2, ty1 = this.ty1, ty2 = this.ty2, tz1 = this.tz1, tz2 = this.tz2, dx1 = this.dx1, dx2 = this.dx2, dx3 = this.dx3, dx4 = this.dx4, dx5 = this.dx5, dy1 = this.dy1, dy2 = this.dy2, dy3 = this.dy3, dy4 = this.dy4, dy5 = this.dy5, dz1 = this.dz1, dz2 = this.dz2, dz3 = this.dz3, dz4 = this.dz4, dz5 = this.dz5;
		final int ldx = this.ldx, ldxy = this.ldx*this.ldy, base = ix(0,j,k);
		for (i = ist; i <= iend; i++){
			final int pc = base + i;
			final double rc = rhoI[pc], u1c = u[5*pc+1], u2c = u[5*pc+2], u3c = u[5*pc+3], u4c = u[5*pc+4];
			final int pkm = pc - ldxy;
			final double qkm = qs[pkm], rkm = rhoI[pkm], u1km = u[5*pkm+1], u2km = u[5*pkm+2], u3km = u[5*pkm+3], u4km = u[5*pkm+4];
			final int pjm = pc - ldx;
			final double qjm = qs[pjm], rjm = rhoI[pjm], u1jm = u[5*pjm+1], u2jm = u[5*pjm+2], u3jm = u[5*pjm+3], u4jm = u[5*pjm+4];
			final int pim = pc - 1;
			final double qim = qs[pim], rim = rhoI[pim], u1im = u[5*pim+1], u2im = u[5*pim+2], u3im = u[5*pim+3], u4im = u[5*pim+4];
			tmp1 = rc;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			d[25*i+0+5*0] = 1.0 + dt * 2.0 * ( tx1 * dx1 + ty1 * dy1 + tz1 * dz1 );
			d[25*i+0+5*1] = 0.0;
			d[25*i+0+5*2] = 0.0;
			d[25*i+0+5*3] = 0.0;
			d[25*i+0+5*4] = 0.0;
			d[25*i+1+5*0] = -dt * 2.0 * ( tx1 * r43 + ty1 + tz1 ) * c34 * tmp2 * u1c;
			d[25*i+1+5*1] = 1.0 + dt * 2.0 * c34 * tmp1 * ( tx1 * r43 + ty1 + tz1 ) + dt * 2.0 * ( tx1 * dx2 + ty1 * dy2 + tz1 * dz2 );
			d[25*i+1+5*2] = 0.0;
			d[25*i+1+5*3] = 0.0;
			d[25*i+1+5*4] = 0.0;
			d[25*i+2+5*0] = -dt * 2.0 * ( tx1 + ty1 * r43 + tz1 ) * c34 * tmp2 * u2c;
			d[25*i+2+5*1] = 0.0;
			d[25*i+2+5*2] = 1.0 + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 * r43 + tz1 ) + dt * 2.0 * ( tx1 * dx3 + ty1 * dy3 + tz1 * dz3 );
			d[25*i+2+5*3] = 0.0;
			d[25*i+2+5*4] = 0.0;
			d[25*i+3+5*0] = -dt * 2.0 * ( tx1 + ty1 + tz1 * r43 ) * c34 * tmp2 * u3c;
			d[25*i+3+5*1] = 0.0;
			d[25*i+3+5*2] = 0.0;
			d[25*i+3+5*3] = 1.0 + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 + tz1 * r43 ) + dt * 2.0 * ( tx1 * dx4 + ty1 * dy4 + tz1 * dz4 );
			d[25*i+3+5*4] = 0.0;
			d[25*i+4+5*0] = -dt * 2.0 * ( ( ( tx1 * ( r43*c34 - c1345 ) + ty1 * ( c34 - c1345 ) + tz1 * ( c34 - c1345 ) ) * ( (u1c)*(u1c) ) + ( tx1 * ( c34 - c1345 ) + ty1 * ( r43*c34 - c1345 ) + tz1 * ( c34 - c1345 ) ) * ( (u2c)*(u2c) ) + ( tx1 * ( c34 - c1345 ) + ty1 * ( c34 - c1345 ) + tz1 * ( r43*c34 - c1345 ) ) * ( (u3c)*(u3c) ) ) * tmp3 + ( tx1 + ty1 + tz1 ) * c1345 * tmp2 * u4c );
			d[25*i+4+5*1] = dt * 2.0 * tmp2 * u1c * ( tx1 * ( r43*c34 - c1345 ) + ty1 * ( c34 - c1345 ) + tz1 * ( c34 - c1345 ) );
			d[25*i+4+5*2] = dt * 2.0 * tmp2 * u2c * ( tx1 * ( c34 - c1345 ) + ty1 * ( r43*c34 -c1345 ) + tz1 * ( c34 - c1345 ) );
			d[25*i+4+5*3] = dt * 2.0 * tmp2 * u3c * ( tx1 * ( c34 - c1345 ) + ty1 * ( c34 - c1345 ) + tz1 * ( r43*c34 - c1345 ) );
			d[25*i+4+5*4] = 1.0 + dt * 2.0 * ( tx1 + ty1 + tz1 ) * c1345 * tmp1 + dt * 2.0 * ( tx1 * dx5 + ty1 * dy5 + tz1 * dz5 );
			tmp1 = rkm;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			a[25*i+0+5*0] = - dt * tz1 * dz1;
			a[25*i+0+5*1] = 0.0;
			a[25*i+0+5*2] = 0.0;
			a[25*i+0+5*3] = - dt * tz2;
			a[25*i+0+5*4] = 0.0;
			a[25*i+1+5*0] = - dt * tz2 * ( - ( u1km*u3km ) * tmp2 ) - dt * tz1 * ( - c34 * tmp2 * u1km );
			a[25*i+1+5*1] = - dt * tz2 * ( u3km * tmp1 ) - dt * tz1 * c34 * tmp1 - dt * tz1 * dz2;
			a[25*i+1+5*2] = 0.0;
			a[25*i+1+5*3] = - dt * tz2 * ( u1km * tmp1 );
			a[25*i+1+5*4] = 0.0;
			a[25*i+2+5*0] = - dt * tz2 * ( - ( u2km*u3km ) * tmp2 ) - dt * tz1 * ( - c34 * tmp2 * u2km );
			a[25*i+2+5*1] = 0.0;
			a[25*i+2+5*2] = - dt * tz2 * ( u3km * tmp1 ) - dt * tz1 * ( c34 * tmp1 ) - dt * tz1 * dz3;
			a[25*i+2+5*3] = - dt * tz2 * ( u2km * tmp1 );
			a[25*i+2+5*4] = 0.0;
			a[25*i+3+5*0] = - dt * tz2 * ( - (( u3km * tmp1 ))*(( u3km * tmp1 )) + c2 * qkm * tmp1 ) - dt * tz1 * ( - r43 * c34 * tmp2 * u3km );
			a[25*i+3+5*1] = - dt * tz2 * ( - c2 * ( u1km * tmp1 ) );
			a[25*i+3+5*2] = - dt * tz2 * ( - c2 * ( u2km * tmp1 ) );
			a[25*i+3+5*3] = - dt * tz2 * ( 2.0 - c2 ) * ( u3km * tmp1 ) - dt * tz1 * ( r43 * c34 * tmp1 ) - dt * tz1 * dz4;
			a[25*i+3+5*4] = - dt * tz2 * c2;
			a[25*i+4+5*0] = - dt * tz2 * ( ( c2 * 2.0 * qkm - c1 * u4km ) * u3km * tmp2 ) - dt * tz1 * ( - ( c34 - c1345 ) * tmp3 * ((u1km)*(u1km)) - ( c34 - c1345 ) * tmp3 * ((u2km)*(u2km)) - ( r43*c34 - c1345 )* tmp3 * ((u3km)*(u3km)) - c1345 * tmp2 * u4km );
			a[25*i+4+5*1] = - dt * tz2 * ( - c2 * ( u1km*u3km ) * tmp2 ) - dt * tz1 * ( c34 - c1345 ) * tmp2 * u1km;
			a[25*i+4+5*2] = - dt * tz2 * ( - c2 * ( u2km*u3km ) * tmp2 ) - dt * tz1 * ( c34 - c1345 ) * tmp2 * u2km;
			a[25*i+4+5*3] = - dt * tz2 * ( c1 * ( u4km * tmp1 ) - c2 * ( qkm * tmp1 + u3km*u3km * tmp2 ) ) - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u3km;
			a[25*i+4+5*4] = - dt * tz2 * ( c1 * ( u3km * tmp1 ) ) - dt * tz1 * c1345 * tmp1 - dt * tz1 * dz5;
			tmp1 = rjm;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			b[25*i+0+5*0] = - dt * ty1 * dy1;
			b[25*i+0+5*1] = 0.0;
			b[25*i+0+5*2] = - dt * ty2;
			b[25*i+0+5*3] = 0.0;
			b[25*i+0+5*4] = 0.0;
			b[25*i+1+5*0] = - dt * ty2 * ( - ( u1jm*u2jm ) * tmp2 ) - dt * ty1 * ( - c34 * tmp2 * u1jm );
			b[25*i+1+5*1] = - dt * ty2 * ( u2jm * tmp1 ) - dt * ty1 * ( c34 * tmp1 ) - dt * ty1 * dy2;
			b[25*i+1+5*2] = - dt * ty2 * ( u1jm * tmp1 );
			b[25*i+1+5*3] = 0.0;
			b[25*i+1+5*4] = 0.0;
			b[25*i+2+5*0] = - dt * ty2 * ( - (( u2jm * tmp1 ))*(( u2jm * tmp1 )) + c2 * ( qjm * tmp1 ) ) - dt * ty1 * ( - r43 * c34 * tmp2 * u2jm );
			b[25*i+2+5*1] = - dt * ty2 * ( - c2 * ( u1jm * tmp1 ) );
			b[25*i+2+5*2] = - dt * ty2 * ( ( 2.0 - c2 ) * ( u2jm * tmp1 ) ) - dt * ty1 * ( r43 * c34 * tmp1 ) - dt * ty1 * dy3;
			b[25*i+2+5*3] = - dt * ty2 * ( - c2 * ( u3jm * tmp1 ) );
			b[25*i+2+5*4] = - dt * ty2 * c2;
			b[25*i+3+5*0] = - dt * ty2 * ( - ( u2jm*u3jm ) * tmp2 ) - dt * ty1 * ( - c34 * tmp2 * u3jm );
			b[25*i+3+5*1] = 0.0;
			b[25*i+3+5*2] = - dt * ty2 * ( u3jm * tmp1 );
			b[25*i+3+5*3] = - dt * ty2 * ( u2jm * tmp1 ) - dt * ty1 * ( c34 * tmp1 ) - dt * ty1 * dy4;
			b[25*i+3+5*4] = 0.0;
			b[25*i+4+5*0] = - dt * ty2 * ( ( c2 * 2.0 * qjm - c1 * u4jm ) * ( u2jm * tmp2 ) ) - dt * ty1 * ( - ( c34 - c1345 )*tmp3*((u1jm)*(u1jm)) - ( r43*c34 - c1345 )*tmp3*((u2jm)*(u2jm)) - ( c34 - c1345 )*tmp3*((u3jm)*(u3jm)) - c1345*tmp2*u4jm );
			b[25*i+4+5*1] = - dt * ty2 * ( - c2 * ( u1jm*u2jm ) * tmp2 ) - dt * ty1 * ( c34 - c1345 ) * tmp2 * u1jm;
			b[25*i+4+5*2] = - dt * ty2 * ( c1 * ( u4jm * tmp1 ) - c2 * ( qjm * tmp1 + u2jm*u2jm * tmp2 ) ) - dt * ty1 * ( r43*c34 - c1345 ) * tmp2 * u2jm;
			b[25*i+4+5*3] = - dt * ty2 * ( - c2 * ( u2jm*u3jm ) * tmp2 ) - dt * ty1 * ( c34 - c1345 ) * tmp2 * u3jm;
			b[25*i+4+5*4] = - dt * ty2 * ( c1 * ( u2jm * tmp1 ) ) - dt * ty1 * c1345 * tmp1 - dt * ty1 * dy5;
			tmp1 = rim;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			c[25*i+0+5*0] = - dt * tx1 * dx1;
			c[25*i+0+5*1] = - dt * tx2;
			c[25*i+0+5*2] = 0.0;
			c[25*i+0+5*3] = 0.0;
			c[25*i+0+5*4] = 0.0;
			c[25*i+1+5*0] = - dt * tx2 * ( - (( u1im * tmp1 ))*(( u1im * tmp1 )) + c2 * qim * tmp1 ) - dt * tx1 * ( - r43 * c34 * tmp2 * u1im );
			c[25*i+1+5*1] = - dt * tx2 * ( ( 2.0 - c2 ) * ( u1im * tmp1 ) ) - dt * tx1 * ( r43 * c34 * tmp1 ) - dt * tx1 * dx2;
			c[25*i+1+5*2] = - dt * tx2 * ( - c2 * ( u2im * tmp1 ) );
			c[25*i+1+5*3] = - dt * tx2 * ( - c2 * ( u3im * tmp1 ) );
			c[25*i+1+5*4] = - dt * tx2 * c2;
			c[25*i+2+5*0] = - dt * tx2 * ( - ( u1im * u2im ) * tmp2 ) - dt * tx1 * ( - c34 * tmp2 * u2im );
			c[25*i+2+5*1] = - dt * tx2 * ( u2im * tmp1 );
			c[25*i+2+5*2] = - dt * tx2 * ( u1im * tmp1 ) - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx3;
			c[25*i+2+5*3] = 0.0;
			c[25*i+2+5*4] = 0.0;
			c[25*i+3+5*0] = - dt * tx2 * ( - ( u1im*u3im ) * tmp2 ) - dt * tx1 * ( - c34 * tmp2 * u3im );
			c[25*i+3+5*1] = - dt * tx2 * ( u3im * tmp1 );
			c[25*i+3+5*2] = 0.0;
			c[25*i+3+5*3] = - dt * tx2 * ( u1im * tmp1 ) - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx4;
			c[25*i+3+5*4] = 0.0;
			c[25*i+4+5*0] = - dt * tx2 * ( ( c2 * 2.0 * qim - c1 * u4im ) * u1im * tmp2 ) - dt * tx1 * ( - ( r43*c34 - c1345 ) * tmp3 * ( (u1im)*(u1im) ) - ( c34 - c1345 ) * tmp3 * ( (u2im)*(u2im) ) - ( c34 - c1345 ) * tmp3 * ( (u3im)*(u3im) ) - c1345 * tmp2 * u4im );
			c[25*i+4+5*1] = - dt * tx2 * ( c1 * ( u4im * tmp1 ) - c2 * ( u1im*u1im * tmp2 + qim * tmp1 ) ) - dt * tx1 * ( r43*c34 - c1345 ) * tmp2 * u1im;
			c[25*i+4+5*2] = - dt * tx2 * ( - c2 * ( u2im*u1im ) * tmp2 ) - dt * tx1 * ( c34 - c1345 ) * tmp2 * u2im;
			c[25*i+4+5*3] = - dt * tx2 * ( - c2 * ( u3im*u1im ) * tmp2 ) - dt * tx1 * ( c34 - c1345 ) * tmp2 * u3im;
			c[25*i+4+5*4] = - dt * tx2 * ( c1 * ( u1im * tmp1 ) ) - dt * tx1 * c1345 * tmp1 - dt * tx1 * dx5;
		}
	}

	/* Upper triangular part of the jacobian for the line j of the plane k: blocks d, a, b and c of each point */
	private void jacu(int j, int k){
		int i;
		double tmp1, tmp2, tmp3;
		double r43 = ( 4.0 / 3.0 );
		double c1345 = c1 * c3 * c4 * c5;
		double c34 = c3 * c4;
		/* fields in locals and indices without ix(); the operands of u, qs and rhoI are read once per point (same expressions) */
		final double[] u = this.u, qs = this.qs, rhoI = this.rhoI, a = this.a, b = this.b, c = this.c, d = this.d;
		final double dt = this.dt, tx1 = this.tx1, tx2 = this.tx2, ty1 = this.ty1, ty2 = this.ty2, tz1 = this.tz1, tz2 = this.tz2, dx1 = this.dx1, dx2 = this.dx2, dx3 = this.dx3, dx4 = this.dx4, dx5 = this.dx5, dy1 = this.dy1, dy2 = this.dy2, dy3 = this.dy3, dy4 = this.dy4, dy5 = this.dy5, dz1 = this.dz1, dz2 = this.dz2, dz3 = this.dz3, dz4 = this.dz4, dz5 = this.dz5;
		final int ldx = this.ldx, ldxy = this.ldx*this.ldy, base = ix(0,j,k);
		for (i = iend; i >= ist; i--){
			final int pc = base + i;
			final double rc = rhoI[pc], u1c = u[5*pc+1], u2c = u[5*pc+2], u3c = u[5*pc+3], u4c = u[5*pc+4];
			final int pip = pc + 1;
			final double qip = qs[pip], rip = rhoI[pip], u1ip = u[5*pip+1], u2ip = u[5*pip+2], u3ip = u[5*pip+3], u4ip = u[5*pip+4];
			final int pjp = pc + ldx;
			final double qjp = qs[pjp], rjp = rhoI[pjp], u1jp = u[5*pjp+1], u2jp = u[5*pjp+2], u3jp = u[5*pjp+3], u4jp = u[5*pjp+4];
			final int pkp = pc + ldxy;
			final double qkp = qs[pkp], rkp = rhoI[pkp], u1kp = u[5*pkp+1], u2kp = u[5*pkp+2], u3kp = u[5*pkp+3], u4kp = u[5*pkp+4];
			tmp1 = rc;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			d[25*i+0+5*0] = 1.0 + dt * 2.0 * ( tx1 * dx1 + ty1 * dy1 + tz1 * dz1 );
			d[25*i+0+5*1] = 0.0;
			d[25*i+0+5*2] = 0.0;
			d[25*i+0+5*3] = 0.0;
			d[25*i+0+5*4] = 0.0;
			d[25*i+1+5*0] = dt * 2.0 * ( - tx1 * r43 - ty1 - tz1 ) * ( c34 * tmp2 * u1c );
			d[25*i+1+5*1] = 1.0 + dt * 2.0 * c34 * tmp1 * ( tx1 * r43 + ty1 + tz1 ) + dt * 2.0 * ( tx1 * dx2 + ty1 * dy2 + tz1 * dz2 );
			d[25*i+1+5*2] = 0.0;
			d[25*i+1+5*3] = 0.0;
			d[25*i+1+5*4] = 0.0;
			d[25*i+2+5*0] = dt * 2.0 * ( - tx1 - ty1 * r43 - tz1 ) * ( c34 * tmp2 * u2c );
			d[25*i+2+5*1] = 0.0;
			d[25*i+2+5*2] = 1.0 + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 * r43 + tz1 ) + dt * 2.0 * ( tx1 * dx3 + ty1 * dy3 + tz1 * dz3 );
			d[25*i+2+5*3] = 0.0;
			d[25*i+2+5*4] = 0.0;
			d[25*i+3+5*0] = dt * 2.0 * ( - tx1 - ty1 - tz1 * r43 ) * ( c34 * tmp2 * u3c );
			d[25*i+3+5*1] = 0.0;
			d[25*i+3+5*2] = 0.0;
			d[25*i+3+5*3] = 1.0 + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 + tz1 * r43 ) + dt * 2.0 * ( tx1 * dx4 + ty1 * dy4 + tz1 * dz4 );
			d[25*i+3+5*4] = 0.0;
			d[25*i+4+5*0] = -dt * 2.0 * ( ( ( tx1 * ( r43*c34 - c1345 ) + ty1 * ( c34 - c1345 ) + tz1 * ( c34 - c1345 ) ) * ( (u1c)*(u1c) ) + ( tx1 * ( c34 - c1345 ) + ty1 * ( r43*c34 - c1345 ) + tz1 * ( c34 - c1345 ) ) * ( (u2c)*(u2c) ) + ( tx1 * ( c34 - c1345 ) + ty1 * ( c34 - c1345 ) + tz1 * ( r43*c34 - c1345 ) ) * ( (u3c)*(u3c) ) ) * tmp3 + ( tx1 + ty1 + tz1 ) * c1345 * tmp2 * u4c );
			d[25*i+4+5*1] = dt * 2.0 * ( tx1 * ( r43*c34 - c1345 ) + ty1 * ( c34 - c1345 ) + tz1 * ( c34 - c1345 ) ) * tmp2 * u1c;
			d[25*i+4+5*2] = dt * 2.0 * ( tx1 * ( c34 - c1345 ) + ty1 * ( r43*c34 -c1345 ) + tz1 * ( c34 - c1345 ) ) * tmp2 * u2c;
			d[25*i+4+5*3] = dt * 2.0 * ( tx1 * ( c34 - c1345 ) + ty1 * ( c34 - c1345 ) + tz1 * ( r43*c34 - c1345 ) ) * tmp2 * u3c;
			d[25*i+4+5*4] = 1.0 + dt * 2.0 * ( tx1 + ty1 + tz1 ) * c1345 * tmp1 + dt * 2.0 * ( tx1 * dx5 + ty1 * dy5 + tz1 * dz5 );
			tmp1 = rip;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			a[25*i+0+5*0] = - dt * tx1 * dx1;
			a[25*i+0+5*1] = dt * tx2;
			a[25*i+0+5*2] = 0.0;
			a[25*i+0+5*3] = 0.0;
			a[25*i+0+5*4] = 0.0;
			a[25*i+1+5*0] = dt * tx2 * ( - (( u1ip * tmp1 ))*(( u1ip * tmp1 )) + c2 * qip * tmp1 ) - dt * tx1 * ( - r43 * c34 * tmp2 * u1ip );
			a[25*i+1+5*1] = dt * tx2 * ( ( 2.0 - c2 ) * ( u1ip * tmp1 ) ) - dt * tx1 * ( r43 * c34 * tmp1 ) - dt * tx1 * dx2;
			a[25*i+1+5*2] = dt * tx2 * ( - c2 * ( u2ip * tmp1 ) );
			a[25*i+1+5*3] = dt * tx2 * ( - c2 * ( u3ip * tmp1 ) );
			a[25*i+1+5*4] = dt * tx2 * c2;
			a[25*i+2+5*0] = dt * tx2 * ( - ( u1ip * u2ip ) * tmp2 ) - dt * tx1 * ( - c34 * tmp2 * u2ip );
			a[25*i+2+5*1] = dt * tx2 * ( u2ip * tmp1 );
			a[25*i+2+5*2] = dt * tx2 * ( u1ip * tmp1 ) - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx3;
			a[25*i+2+5*3] = 0.0;
			a[25*i+2+5*4] = 0.0;
			a[25*i+3+5*0] = dt * tx2 * ( - ( u1ip*u3ip ) * tmp2 ) - dt * tx1 * ( - c34 * tmp2 * u3ip );
			a[25*i+3+5*1] = dt * tx2 * ( u3ip * tmp1 );
			a[25*i+3+5*2] = 0.0;
			a[25*i+3+5*3] = dt * tx2 * ( u1ip * tmp1 ) - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx4;
			a[25*i+3+5*4] = 0.0;
			a[25*i+4+5*0] = dt * tx2 * ( ( c2 * 2.0 * qip - c1 * u4ip ) * ( u1ip * tmp2 ) ) - dt * tx1 * ( - ( r43*c34 - c1345 ) * tmp3 * ( (u1ip)*(u1ip) ) - ( c34 - c1345 ) * tmp3 * ( (u2ip)*(u2ip) ) - ( c34 - c1345 ) * tmp3 * ( (u3ip)*(u3ip) ) - c1345 * tmp2 * u4ip );
			a[25*i+4+5*1] = dt * tx2 * ( c1 * ( u4ip * tmp1 ) - c2 * ( u1ip*u1ip * tmp2 + qip * tmp1 ) ) - dt * tx1 * ( r43*c34 - c1345 ) * tmp2 * u1ip;
			a[25*i+4+5*2] = dt * tx2 * ( - c2 * ( u2ip*u1ip ) * tmp2 ) - dt * tx1 * ( c34 - c1345 ) * tmp2 * u2ip;
			a[25*i+4+5*3] = dt * tx2 * ( - c2 * ( u3ip*u1ip ) * tmp2 ) - dt * tx1 * ( c34 - c1345 ) * tmp2 * u3ip;
			a[25*i+4+5*4] = dt * tx2 * ( c1 * ( u1ip * tmp1 ) ) - dt * tx1 * c1345 * tmp1 - dt * tx1 * dx5;
			tmp1 = rjp;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			b[25*i+0+5*0] = - dt * ty1 * dy1;
			b[25*i+0+5*1] = 0.0;
			b[25*i+0+5*2] = dt * ty2;
			b[25*i+0+5*3] = 0.0;
			b[25*i+0+5*4] = 0.0;
			b[25*i+1+5*0] = dt * ty2 * ( - ( u1jp*u2jp ) * tmp2 ) - dt * ty1 * ( - c34 * tmp2 * u1jp );
			b[25*i+1+5*1] = dt * ty2 * ( u2jp * tmp1 ) - dt * ty1 * ( c34 * tmp1 ) - dt * ty1 * dy2;
			b[25*i+1+5*2] = dt * ty2 * ( u1jp * tmp1 );
			b[25*i+1+5*3] = 0.0;
			b[25*i+1+5*4] = 0.0;
			b[25*i+2+5*0] = dt * ty2 * ( - (( u2jp * tmp1 ))*(( u2jp * tmp1 )) + c2 * ( qjp * tmp1 ) ) - dt * ty1 * ( - r43 * c34 * tmp2 * u2jp );
			b[25*i+2+5*1] = dt * ty2 * ( - c2 * ( u1jp * tmp1 ) );
			b[25*i+2+5*2] = dt * ty2 * ( ( 2.0 - c2 ) * ( u2jp * tmp1 ) ) - dt * ty1 * ( r43 * c34 * tmp1 ) - dt * ty1 * dy3;
			b[25*i+2+5*3] = dt * ty2 * ( - c2 * ( u3jp * tmp1 ) );
			b[25*i+2+5*4] = dt * ty2 * c2;
			b[25*i+3+5*0] = dt * ty2 * ( - ( u2jp*u3jp ) * tmp2 ) - dt * ty1 * ( - c34 * tmp2 * u3jp );
			b[25*i+3+5*1] = 0.0;
			b[25*i+3+5*2] = dt * ty2 * ( u3jp * tmp1 );
			b[25*i+3+5*3] = dt * ty2 * ( u2jp * tmp1 ) - dt * ty1 * ( c34 * tmp1 ) - dt * ty1 * dy4;
			b[25*i+3+5*4] = 0.0;
			b[25*i+4+5*0] = dt * ty2 * ( ( c2 * 2.0 * qjp - c1 * u4jp ) * ( u2jp * tmp2 ) ) - dt * ty1 * ( - ( c34 - c1345 )*tmp3*((u1jp)*(u1jp)) - ( r43*c34 - c1345 )*tmp3*((u2jp)*(u2jp)) - ( c34 - c1345 )*tmp3*((u3jp)*(u3jp)) - c1345*tmp2*u4jp );
			b[25*i+4+5*1] = dt * ty2 * ( - c2 * ( u1jp*u2jp ) * tmp2 ) - dt * ty1 * ( c34 - c1345 ) * tmp2 * u1jp;
			b[25*i+4+5*2] = dt * ty2 * ( c1 * ( u4jp * tmp1 ) - c2 * ( qjp * tmp1 + u2jp*u2jp * tmp2 ) ) - dt * ty1 * ( r43*c34 - c1345 ) * tmp2 * u2jp;
			b[25*i+4+5*3] = dt * ty2 * ( - c2 * ( u2jp*u3jp ) * tmp2 ) - dt * ty1 * ( c34 - c1345 ) * tmp2 * u3jp;
			b[25*i+4+5*4] = dt * ty2 * ( c1 * ( u2jp * tmp1 ) ) - dt * ty1 * c1345 * tmp1 - dt * ty1 * dy5;
			tmp1 = rkp;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			c[25*i+0+5*0] = - dt * tz1 * dz1;
			c[25*i+0+5*1] = 0.0;
			c[25*i+0+5*2] = 0.0;
			c[25*i+0+5*3] = dt * tz2;
			c[25*i+0+5*4] = 0.0;
			c[25*i+1+5*0] = dt * tz2 * ( - ( u1kp*u3kp ) * tmp2 ) - dt * tz1 * ( - c34 * tmp2 * u1kp );
			c[25*i+1+5*1] = dt * tz2 * ( u3kp * tmp1 ) - dt * tz1 * c34 * tmp1 - dt * tz1 * dz2;
			c[25*i+1+5*2] = 0.0;
			c[25*i+1+5*3] = dt * tz2 * ( u1kp * tmp1 );
			c[25*i+1+5*4] = 0.0;
			c[25*i+2+5*0] = dt * tz2 * ( - ( u2kp*u3kp ) * tmp2 ) - dt * tz1 * ( - c34 * tmp2 * u2kp );
			c[25*i+2+5*1] = 0.0;
			c[25*i+2+5*2] = dt * tz2 * ( u3kp * tmp1 ) - dt * tz1 * ( c34 * tmp1 ) - dt * tz1 * dz3;
			c[25*i+2+5*3] = dt * tz2 * ( u2kp * tmp1 );
			c[25*i+2+5*4] = 0.0;
			c[25*i+3+5*0] = dt * tz2 * ( - (( u3kp * tmp1 ))*(( u3kp * tmp1 )) + c2 * ( qkp * tmp1 ) ) - dt * tz1 * ( - r43 * c34 * tmp2 * u3kp );
			c[25*i+3+5*1] = dt * tz2 * ( - c2 * ( u1kp * tmp1 ) );
			c[25*i+3+5*2] = dt * tz2 * ( - c2 * ( u2kp * tmp1 ) );
			c[25*i+3+5*3] = dt * tz2 * ( 2.0 - c2 ) * ( u3kp * tmp1 ) - dt * tz1 * ( r43 * c34 * tmp1 ) - dt * tz1 * dz4;
			c[25*i+3+5*4] = dt * tz2 * c2;
			c[25*i+4+5*0] = dt * tz2 * ( ( c2 * 2.0 * qkp - c1 * u4kp ) * ( u3kp * tmp2 ) ) - dt * tz1 * ( - ( c34 - c1345 ) * tmp3 * ((u1kp)*(u1kp)) - ( c34 - c1345 ) * tmp3 * ((u2kp)*(u2kp)) - ( r43*c34 - c1345 )* tmp3 * ((u3kp)*(u3kp)) - c1345 * tmp2 * u4kp );
			c[25*i+4+5*1] = dt * tz2 * ( - c2 * ( u1kp*u3kp ) * tmp2 ) - dt * tz1 * ( c34 - c1345 ) * tmp2 * u1kp;
			c[25*i+4+5*2] = dt * tz2 * ( - c2 * ( u2kp*u3kp ) * tmp2 ) - dt * tz1 * ( c34 - c1345 ) * tmp2 * u2kp;
			c[25*i+4+5*3] = dt * tz2 * ( c1 * ( u4kp * tmp1 ) - c2 * ( qkp * tmp1 + u3kp*u3kp * tmp2 ) ) - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u3kp;
			c[25*i+4+5*4] = dt * tz2 * ( c1 * ( u3kp * tmp1 ) ) - dt * tz1 * c1345 * tmp1 - dt * tz1 * dz5;
		}
	}

	/* Block lower triangular solve of the line j of the plane k, in place in rsd */
	private void blts(double ldz[], double ldy[], double ldx[], double d[], int j, int k){
		int i;
		double tmp, tmp1;
		/* loops over m unrolled and tmat/tv in local variables (same operations, same order) */
		final double[] rsd = this.rsd;
		final double omega = this.omega;
		double tv0, tv1, tv2, tv3, tv4;
		double t00, t01, t02, t03, t04, t10, t11, t12, t13, t14, t20, t21, t22, t23, t24, t30, t31, t32, t33, t34, t40, t41, t42, t43, t44;
		for (i = ist; i <= iend; i++){
			tv0 = rsd[5*ix(i,j,k)+0] - omega * ( ldz[25*i+0+5*0] * rsd[5*ix(i,j,k-1)+0] + ldz[25*i+0+5*1] * rsd[5*ix(i,j,k-1)+1] + ldz[25*i+0+5*2] * rsd[5*ix(i,j,k-1)+2] + ldz[25*i+0+5*3] * rsd[5*ix(i,j,k-1)+3] + ldz[25*i+0+5*4] * rsd[5*ix(i,j,k-1)+4] );
			tv0 = tv0 - omega * ( ldy[25*i+0+5*0] * rsd[5*ix(i,j-1,k)+0] + ldx[25*i+0+5*0] * rsd[5*ix(i-1,j,k)+0] + ldy[25*i+0+5*1] * rsd[5*ix(i,j-1,k)+1] + ldx[25*i+0+5*1] * rsd[5*ix(i-1,j,k)+1] + ldy[25*i+0+5*2] * rsd[5*ix(i,j-1,k)+2] + ldx[25*i+0+5*2] * rsd[5*ix(i-1,j,k)+2] + ldy[25*i+0+5*3] * rsd[5*ix(i,j-1,k)+3] + ldx[25*i+0+5*3] * rsd[5*ix(i-1,j,k)+3] + ldy[25*i+0+5*4] * rsd[5*ix(i,j-1,k)+4] + ldx[25*i+0+5*4] * rsd[5*ix(i-1,j,k)+4] );
			tv1 = rsd[5*ix(i,j,k)+1] - omega * ( ldz[25*i+1+5*0] * rsd[5*ix(i,j,k-1)+0] + ldz[25*i+1+5*1] * rsd[5*ix(i,j,k-1)+1] + ldz[25*i+1+5*2] * rsd[5*ix(i,j,k-1)+2] + ldz[25*i+1+5*3] * rsd[5*ix(i,j,k-1)+3] + ldz[25*i+1+5*4] * rsd[5*ix(i,j,k-1)+4] );
			tv1 = tv1 - omega * ( ldy[25*i+1+5*0] * rsd[5*ix(i,j-1,k)+0] + ldx[25*i+1+5*0] * rsd[5*ix(i-1,j,k)+0] + ldy[25*i+1+5*1] * rsd[5*ix(i,j-1,k)+1] + ldx[25*i+1+5*1] * rsd[5*ix(i-1,j,k)+1] + ldy[25*i+1+5*2] * rsd[5*ix(i,j-1,k)+2] + ldx[25*i+1+5*2] * rsd[5*ix(i-1,j,k)+2] + ldy[25*i+1+5*3] * rsd[5*ix(i,j-1,k)+3] + ldx[25*i+1+5*3] * rsd[5*ix(i-1,j,k)+3] + ldy[25*i+1+5*4] * rsd[5*ix(i,j-1,k)+4] + ldx[25*i+1+5*4] * rsd[5*ix(i-1,j,k)+4] );
			tv2 = rsd[5*ix(i,j,k)+2] - omega * ( ldz[25*i+2+5*0] * rsd[5*ix(i,j,k-1)+0] + ldz[25*i+2+5*1] * rsd[5*ix(i,j,k-1)+1] + ldz[25*i+2+5*2] * rsd[5*ix(i,j,k-1)+2] + ldz[25*i+2+5*3] * rsd[5*ix(i,j,k-1)+3] + ldz[25*i+2+5*4] * rsd[5*ix(i,j,k-1)+4] );
			tv2 = tv2 - omega * ( ldy[25*i+2+5*0] * rsd[5*ix(i,j-1,k)+0] + ldx[25*i+2+5*0] * rsd[5*ix(i-1,j,k)+0] + ldy[25*i+2+5*1] * rsd[5*ix(i,j-1,k)+1] + ldx[25*i+2+5*1] * rsd[5*ix(i-1,j,k)+1] + ldy[25*i+2+5*2] * rsd[5*ix(i,j-1,k)+2] + ldx[25*i+2+5*2] * rsd[5*ix(i-1,j,k)+2] + ldy[25*i+2+5*3] * rsd[5*ix(i,j-1,k)+3] + ldx[25*i+2+5*3] * rsd[5*ix(i-1,j,k)+3] + ldy[25*i+2+5*4] * rsd[5*ix(i,j-1,k)+4] + ldx[25*i+2+5*4] * rsd[5*ix(i-1,j,k)+4] );
			tv3 = rsd[5*ix(i,j,k)+3] - omega * ( ldz[25*i+3+5*0] * rsd[5*ix(i,j,k-1)+0] + ldz[25*i+3+5*1] * rsd[5*ix(i,j,k-1)+1] + ldz[25*i+3+5*2] * rsd[5*ix(i,j,k-1)+2] + ldz[25*i+3+5*3] * rsd[5*ix(i,j,k-1)+3] + ldz[25*i+3+5*4] * rsd[5*ix(i,j,k-1)+4] );
			tv3 = tv3 - omega * ( ldy[25*i+3+5*0] * rsd[5*ix(i,j-1,k)+0] + ldx[25*i+3+5*0] * rsd[5*ix(i-1,j,k)+0] + ldy[25*i+3+5*1] * rsd[5*ix(i,j-1,k)+1] + ldx[25*i+3+5*1] * rsd[5*ix(i-1,j,k)+1] + ldy[25*i+3+5*2] * rsd[5*ix(i,j-1,k)+2] + ldx[25*i+3+5*2] * rsd[5*ix(i-1,j,k)+2] + ldy[25*i+3+5*3] * rsd[5*ix(i,j-1,k)+3] + ldx[25*i+3+5*3] * rsd[5*ix(i-1,j,k)+3] + ldy[25*i+3+5*4] * rsd[5*ix(i,j-1,k)+4] + ldx[25*i+3+5*4] * rsd[5*ix(i-1,j,k)+4] );
			tv4 = rsd[5*ix(i,j,k)+4] - omega * ( ldz[25*i+4+5*0] * rsd[5*ix(i,j,k-1)+0] + ldz[25*i+4+5*1] * rsd[5*ix(i,j,k-1)+1] + ldz[25*i+4+5*2] * rsd[5*ix(i,j,k-1)+2] + ldz[25*i+4+5*3] * rsd[5*ix(i,j,k-1)+3] + ldz[25*i+4+5*4] * rsd[5*ix(i,j,k-1)+4] );
			tv4 = tv4 - omega * ( ldy[25*i+4+5*0] * rsd[5*ix(i,j-1,k)+0] + ldx[25*i+4+5*0] * rsd[5*ix(i-1,j,k)+0] + ldy[25*i+4+5*1] * rsd[5*ix(i,j-1,k)+1] + ldx[25*i+4+5*1] * rsd[5*ix(i-1,j,k)+1] + ldy[25*i+4+5*2] * rsd[5*ix(i,j-1,k)+2] + ldx[25*i+4+5*2] * rsd[5*ix(i-1,j,k)+2] + ldy[25*i+4+5*3] * rsd[5*ix(i,j-1,k)+3] + ldx[25*i+4+5*3] * rsd[5*ix(i-1,j,k)+3] + ldy[25*i+4+5*4] * rsd[5*ix(i,j-1,k)+4] + ldx[25*i+4+5*4] * rsd[5*ix(i-1,j,k)+4] );
			t00 = d[25*i+0+5*0];
			t01 = d[25*i+0+5*1];
			t02 = d[25*i+0+5*2];
			t03 = d[25*i+0+5*3];
			t04 = d[25*i+0+5*4];
			t10 = d[25*i+1+5*0];
			t11 = d[25*i+1+5*1];
			t12 = d[25*i+1+5*2];
			t13 = d[25*i+1+5*3];
			t14 = d[25*i+1+5*4];
			t20 = d[25*i+2+5*0];
			t21 = d[25*i+2+5*1];
			t22 = d[25*i+2+5*2];
			t23 = d[25*i+2+5*3];
			t24 = d[25*i+2+5*4];
			t30 = d[25*i+3+5*0];
			t31 = d[25*i+3+5*1];
			t32 = d[25*i+3+5*2];
			t33 = d[25*i+3+5*3];
			t34 = d[25*i+3+5*4];
			t40 = d[25*i+4+5*0];
			t41 = d[25*i+4+5*1];
			t42 = d[25*i+4+5*2];
			t43 = d[25*i+4+5*3];
			t44 = d[25*i+4+5*4];
			tmp1 = 1.0 / t00;
			tmp = tmp1 * t10;
			t11 = t11 - tmp * t01;
			t12 = t12 - tmp * t02;
			t13 = t13 - tmp * t03;
			t14 = t14 - tmp * t04;
			tv1 = tv1 - tv0 * tmp;
			tmp = tmp1 * t20;
			t21 = t21 - tmp * t01;
			t22 = t22 - tmp * t02;
			t23 = t23 - tmp * t03;
			t24 = t24 - tmp * t04;
			tv2 = tv2 - tv0 * tmp;
			tmp = tmp1 * t30;
			t31 = t31 - tmp * t01;
			t32 = t32 - tmp * t02;
			t33 = t33 - tmp * t03;
			t34 = t34 - tmp * t04;
			tv3 = tv3 - tv0 * tmp;
			tmp = tmp1 * t40;
			t41 = t41 - tmp * t01;
			t42 = t42 - tmp * t02;
			t43 = t43 - tmp * t03;
			t44 = t44 - tmp * t04;
			tv4 = tv4 - tv0 * tmp;
			tmp1 = 1.0 / t11;
			tmp = tmp1 * t21;
			t22 = t22 - tmp * t12;
			t23 = t23 - tmp * t13;
			t24 = t24 - tmp * t14;
			tv2 = tv2 - tv1 * tmp;
			tmp = tmp1 * t31;
			t32 = t32 - tmp * t12;
			t33 = t33 - tmp * t13;
			t34 = t34 - tmp * t14;
			tv3 = tv3 - tv1 * tmp;
			tmp = tmp1 * t41;
			t42 = t42 - tmp * t12;
			t43 = t43 - tmp * t13;
			t44 = t44 - tmp * t14;
			tv4 = tv4 - tv1 * tmp;
			tmp1 = 1.0 / t22;
			tmp = tmp1 * t32;
			t33 = t33 - tmp * t23;
			t34 = t34 - tmp * t24;
			tv3 = tv3 - tv2 * tmp;
			tmp = tmp1 * t42;
			t43 = t43 - tmp * t23;
			t44 = t44 - tmp * t24;
			tv4 = tv4 - tv2 * tmp;
			tmp1 = 1.0 / t33;
			tmp = tmp1 * t43;
			t44 = t44 - tmp * t34;
			tv4 = tv4 - tv3 * tmp;
			rsd[5*ix(i,j,k)+4] = tv4 / t44;
			tv3 = tv3 - t34 * rsd[5*ix(i,j,k)+4];
			rsd[5*ix(i,j,k)+3] = tv3 / t33;
			tv2 = tv2 - t23 * rsd[5*ix(i,j,k)+3] - t24 * rsd[5*ix(i,j,k)+4];
			rsd[5*ix(i,j,k)+2] = tv2 / t22;
			tv1 = tv1 - t12 * rsd[5*ix(i,j,k)+2] - t13 * rsd[5*ix(i,j,k)+3] - t14 * rsd[5*ix(i,j,k)+4];
			rsd[5*ix(i,j,k)+1] = tv1 / t11;
			tv0 = tv0 - t01 * rsd[5*ix(i,j,k)+1] - t02 * rsd[5*ix(i,j,k)+2] - t03 * rsd[5*ix(i,j,k)+3] - t04 * rsd[5*ix(i,j,k)+4];
			rsd[5*ix(i,j,k)+0] = tv0 / t00;
		}
	}

	/* Block upper triangular solve of the line j of the plane k, in place in rsd */
	private void buts(double d[], double udx[], double udy[], double udz[], int j, int k){
		int i;
		double tmp, tmp1;
		/* loops over m unrolled and tmat/tv in local variables (same operations, same order) */
		final double[] rsd = this.rsd;
		final double omega = this.omega;
		double tv0, tv1, tv2, tv3, tv4;
		double t00, t01, t02, t03, t04, t10, t11, t12, t13, t14, t20, t21, t22, t23, t24, t30, t31, t32, t33, t34, t40, t41, t42, t43, t44;
		for (i = iend; i >= ist; i--){
			tv0 = omega * ( udz[25*i+0+5*0] * rsd[5*ix(i,j,k+1)+0] + udz[25*i+0+5*1] * rsd[5*ix(i,j,k+1)+1] + udz[25*i+0+5*2] * rsd[5*ix(i,j,k+1)+2] + udz[25*i+0+5*3] * rsd[5*ix(i,j,k+1)+3] + udz[25*i+0+5*4] * rsd[5*ix(i,j,k+1)+4] );
			tv0 = tv0 + omega * ( udy[25*i+0+5*0] * rsd[5*ix(i,j+1,k)+0] + udx[25*i+0+5*0] * rsd[5*ix(i+1,j,k)+0] + udy[25*i+0+5*1] * rsd[5*ix(i,j+1,k)+1] + udx[25*i+0+5*1] * rsd[5*ix(i+1,j,k)+1] + udy[25*i+0+5*2] * rsd[5*ix(i,j+1,k)+2] + udx[25*i+0+5*2] * rsd[5*ix(i+1,j,k)+2] + udy[25*i+0+5*3] * rsd[5*ix(i,j+1,k)+3] + udx[25*i+0+5*3] * rsd[5*ix(i+1,j,k)+3] + udy[25*i+0+5*4] * rsd[5*ix(i,j+1,k)+4] + udx[25*i+0+5*4] * rsd[5*ix(i+1,j,k)+4] );
			tv1 = omega * ( udz[25*i+1+5*0] * rsd[5*ix(i,j,k+1)+0] + udz[25*i+1+5*1] * rsd[5*ix(i,j,k+1)+1] + udz[25*i+1+5*2] * rsd[5*ix(i,j,k+1)+2] + udz[25*i+1+5*3] * rsd[5*ix(i,j,k+1)+3] + udz[25*i+1+5*4] * rsd[5*ix(i,j,k+1)+4] );
			tv1 = tv1 + omega * ( udy[25*i+1+5*0] * rsd[5*ix(i,j+1,k)+0] + udx[25*i+1+5*0] * rsd[5*ix(i+1,j,k)+0] + udy[25*i+1+5*1] * rsd[5*ix(i,j+1,k)+1] + udx[25*i+1+5*1] * rsd[5*ix(i+1,j,k)+1] + udy[25*i+1+5*2] * rsd[5*ix(i,j+1,k)+2] + udx[25*i+1+5*2] * rsd[5*ix(i+1,j,k)+2] + udy[25*i+1+5*3] * rsd[5*ix(i,j+1,k)+3] + udx[25*i+1+5*3] * rsd[5*ix(i+1,j,k)+3] + udy[25*i+1+5*4] * rsd[5*ix(i,j+1,k)+4] + udx[25*i+1+5*4] * rsd[5*ix(i+1,j,k)+4] );
			tv2 = omega * ( udz[25*i+2+5*0] * rsd[5*ix(i,j,k+1)+0] + udz[25*i+2+5*1] * rsd[5*ix(i,j,k+1)+1] + udz[25*i+2+5*2] * rsd[5*ix(i,j,k+1)+2] + udz[25*i+2+5*3] * rsd[5*ix(i,j,k+1)+3] + udz[25*i+2+5*4] * rsd[5*ix(i,j,k+1)+4] );
			tv2 = tv2 + omega * ( udy[25*i+2+5*0] * rsd[5*ix(i,j+1,k)+0] + udx[25*i+2+5*0] * rsd[5*ix(i+1,j,k)+0] + udy[25*i+2+5*1] * rsd[5*ix(i,j+1,k)+1] + udx[25*i+2+5*1] * rsd[5*ix(i+1,j,k)+1] + udy[25*i+2+5*2] * rsd[5*ix(i,j+1,k)+2] + udx[25*i+2+5*2] * rsd[5*ix(i+1,j,k)+2] + udy[25*i+2+5*3] * rsd[5*ix(i,j+1,k)+3] + udx[25*i+2+5*3] * rsd[5*ix(i+1,j,k)+3] + udy[25*i+2+5*4] * rsd[5*ix(i,j+1,k)+4] + udx[25*i+2+5*4] * rsd[5*ix(i+1,j,k)+4] );
			tv3 = omega * ( udz[25*i+3+5*0] * rsd[5*ix(i,j,k+1)+0] + udz[25*i+3+5*1] * rsd[5*ix(i,j,k+1)+1] + udz[25*i+3+5*2] * rsd[5*ix(i,j,k+1)+2] + udz[25*i+3+5*3] * rsd[5*ix(i,j,k+1)+3] + udz[25*i+3+5*4] * rsd[5*ix(i,j,k+1)+4] );
			tv3 = tv3 + omega * ( udy[25*i+3+5*0] * rsd[5*ix(i,j+1,k)+0] + udx[25*i+3+5*0] * rsd[5*ix(i+1,j,k)+0] + udy[25*i+3+5*1] * rsd[5*ix(i,j+1,k)+1] + udx[25*i+3+5*1] * rsd[5*ix(i+1,j,k)+1] + udy[25*i+3+5*2] * rsd[5*ix(i,j+1,k)+2] + udx[25*i+3+5*2] * rsd[5*ix(i+1,j,k)+2] + udy[25*i+3+5*3] * rsd[5*ix(i,j+1,k)+3] + udx[25*i+3+5*3] * rsd[5*ix(i+1,j,k)+3] + udy[25*i+3+5*4] * rsd[5*ix(i,j+1,k)+4] + udx[25*i+3+5*4] * rsd[5*ix(i+1,j,k)+4] );
			tv4 = omega * ( udz[25*i+4+5*0] * rsd[5*ix(i,j,k+1)+0] + udz[25*i+4+5*1] * rsd[5*ix(i,j,k+1)+1] + udz[25*i+4+5*2] * rsd[5*ix(i,j,k+1)+2] + udz[25*i+4+5*3] * rsd[5*ix(i,j,k+1)+3] + udz[25*i+4+5*4] * rsd[5*ix(i,j,k+1)+4] );
			tv4 = tv4 + omega * ( udy[25*i+4+5*0] * rsd[5*ix(i,j+1,k)+0] + udx[25*i+4+5*0] * rsd[5*ix(i+1,j,k)+0] + udy[25*i+4+5*1] * rsd[5*ix(i,j+1,k)+1] + udx[25*i+4+5*1] * rsd[5*ix(i+1,j,k)+1] + udy[25*i+4+5*2] * rsd[5*ix(i,j+1,k)+2] + udx[25*i+4+5*2] * rsd[5*ix(i+1,j,k)+2] + udy[25*i+4+5*3] * rsd[5*ix(i,j+1,k)+3] + udx[25*i+4+5*3] * rsd[5*ix(i+1,j,k)+3] + udy[25*i+4+5*4] * rsd[5*ix(i,j+1,k)+4] + udx[25*i+4+5*4] * rsd[5*ix(i+1,j,k)+4] );
			t00 = d[25*i+0+5*0];
			t01 = d[25*i+0+5*1];
			t02 = d[25*i+0+5*2];
			t03 = d[25*i+0+5*3];
			t04 = d[25*i+0+5*4];
			t10 = d[25*i+1+5*0];
			t11 = d[25*i+1+5*1];
			t12 = d[25*i+1+5*2];
			t13 = d[25*i+1+5*3];
			t14 = d[25*i+1+5*4];
			t20 = d[25*i+2+5*0];
			t21 = d[25*i+2+5*1];
			t22 = d[25*i+2+5*2];
			t23 = d[25*i+2+5*3];
			t24 = d[25*i+2+5*4];
			t30 = d[25*i+3+5*0];
			t31 = d[25*i+3+5*1];
			t32 = d[25*i+3+5*2];
			t33 = d[25*i+3+5*3];
			t34 = d[25*i+3+5*4];
			t40 = d[25*i+4+5*0];
			t41 = d[25*i+4+5*1];
			t42 = d[25*i+4+5*2];
			t43 = d[25*i+4+5*3];
			t44 = d[25*i+4+5*4];
			tmp1 = 1.0 / t00;
			tmp = tmp1 * t10;
			t11 = t11 - tmp * t01;
			t12 = t12 - tmp * t02;
			t13 = t13 - tmp * t03;
			t14 = t14 - tmp * t04;
			tv1 = tv1 - tv0 * tmp;
			tmp = tmp1 * t20;
			t21 = t21 - tmp * t01;
			t22 = t22 - tmp * t02;
			t23 = t23 - tmp * t03;
			t24 = t24 - tmp * t04;
			tv2 = tv2 - tv0 * tmp;
			tmp = tmp1 * t30;
			t31 = t31 - tmp * t01;
			t32 = t32 - tmp * t02;
			t33 = t33 - tmp * t03;
			t34 = t34 - tmp * t04;
			tv3 = tv3 - tv0 * tmp;
			tmp = tmp1 * t40;
			t41 = t41 - tmp * t01;
			t42 = t42 - tmp * t02;
			t43 = t43 - tmp * t03;
			t44 = t44 - tmp * t04;
			tv4 = tv4 - tv0 * tmp;
			tmp1 = 1.0 / t11;
			tmp = tmp1 * t21;
			t22 = t22 - tmp * t12;
			t23 = t23 - tmp * t13;
			t24 = t24 - tmp * t14;
			tv2 = tv2 - tv1 * tmp;
			tmp = tmp1 * t31;
			t32 = t32 - tmp * t12;
			t33 = t33 - tmp * t13;
			t34 = t34 - tmp * t14;
			tv3 = tv3 - tv1 * tmp;
			tmp = tmp1 * t41;
			t42 = t42 - tmp * t12;
			t43 = t43 - tmp * t13;
			t44 = t44 - tmp * t14;
			tv4 = tv4 - tv1 * tmp;
			tmp1 = 1.0 / t22;
			tmp = tmp1 * t32;
			t33 = t33 - tmp * t23;
			t34 = t34 - tmp * t24;
			tv3 = tv3 - tv2 * tmp;
			tmp = tmp1 * t42;
			t43 = t43 - tmp * t23;
			t44 = t44 - tmp * t24;
			tv4 = tv4 - tv2 * tmp;
			tmp1 = 1.0 / t33;
			tmp = tmp1 * t43;
			t44 = t44 - tmp * t34;
			tv4 = tv4 - tv3 * tmp;
			tv4 = tv4 / t44;
			tv3 = tv3 - t34 * tv4;
			tv3 = tv3 / t33;
			tv2 = tv2 - t23 * tv3 - t24 * tv4;
			tv2 = tv2 / t22;
			tv1 = tv1 - t12 * tv2 - t13 * tv3 - t14 * tv4;
			tv1 = tv1 / t11;
			tv0 = tv0 - t01 * tv1 - t02 * tv2 - t03 * tv3 - t04 * tv4;
			tv0 = tv0 / t00;
			rsd[5*ix(i,j,k)+0] = rsd[5*ix(i,j,k)+0] - tv0;
			rsd[5*ix(i,j,k)+1] = rsd[5*ix(i,j,k)+1] - tv1;
			rsd[5*ix(i,j,k)+2] = rsd[5*ix(i,j,k)+2] - tv2;
			rsd[5*ix(i,j,k)+3] = rsd[5*ix(i,j,k)+3] - tv3;
			rsd[5*ix(i,j,k)+4] = rsd[5*ix(i,j,k)+4] - tv4;
		}
	}

	/* RMS-norm of the error of the solution */
	private void error(){
		int i, j, k, m;
		double tmp;
		double u000ijk[] = new double[5];
		for (m = 0; m <= 4; m++){
			errnm[m] = 0.0;
		}
		for (k = 2; k <= nz-1; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					exact( i, j, k, u000ijk );
					for (m = 0; m <= 4; m++){
						tmp = ( u000ijk[m] - u[5*ix(i,j,k)+m] );
						errnm[m] = errnm[m] + tmp * tmp;
					}
				}
			}
		}
		for (m = 0; m <= 4; m++){
			errnm[m] = Math.sqrt( errnm[m] / ( (double)(nx0-2)*(ny0-2)*(nz0-2) ) );
		}
	}

	/* Surface integral of the pressure (frc) */
	private void pintgr(){
		int i, ibeg, ifin, ifin1, j, jbeg, jfin, jfin1, k;
		double frc1, frc2, frc3;
		ibeg = ii1;
		ifin = ii2;
		jbeg = ji1;
		jfin = ji2;
		ifin1 = ifin - 1;
		jfin1 = jfin - 1;
		for (j = jbeg; j <= jfin; j++){
			for (i = ibeg; i <= ifin; i++){
				k = ki1;
				phi1[i][j] = c2*( u[5*ix(i,j,k)+4] - 0.50 * ( (u[5*ix(i,j,k)+1])*(u[5*ix(i,j,k)+1]) + (u[5*ix(i,j,k)+2])*(u[5*ix(i,j,k)+2]) + (u[5*ix(i,j,k)+3])*(u[5*ix(i,j,k)+3]) ) / u[5*ix(i,j,k)+0] );
				k = ki2;
				phi2[i][j] = c2*( u[5*ix(i,j,k)+4] - 0.50 * ( (u[5*ix(i,j,k)+1])*(u[5*ix(i,j,k)+1]) + (u[5*ix(i,j,k)+2])*(u[5*ix(i,j,k)+2]) + (u[5*ix(i,j,k)+3])*(u[5*ix(i,j,k)+3]) ) / u[5*ix(i,j,k)+0] );
			}
		}
		frc1 = 0.0;
		for (j = jbeg; j <= jfin1; j++){
			for (i = ibeg; i <= ifin1; i++){
				frc1 = frc1 + ( phi1[i][j] + phi1[i+1][j] + phi1[i][j+1] + phi1[i+1][j+1] + phi2[i][j] + phi2[i+1][j] + phi2[i][j+1] + phi2[i+1][j+1] );
			}
		}
		frc1 = dxi * deta * frc1;
		for (k = ki1; k <= ki2; k++){
			for (i = ibeg; i <= ifin; i++){
				phi1[i][k] = c2*( u[5*ix(i,jbeg,k)+4] - 0.50 * ( (u[5*ix(i,jbeg,k)+1])*(u[5*ix(i,jbeg,k)+1]) + (u[5*ix(i,jbeg,k)+2])*(u[5*ix(i,jbeg,k)+2]) + (u[5*ix(i,jbeg,k)+3])*(u[5*ix(i,jbeg,k)+3]) ) / u[5*ix(i,jbeg,k)+0] );
			}
		}
		for (k = ki1; k <= ki2; k++){
			for (i = ibeg; i <= ifin; i++){
				phi2[i][k] = c2*( u[5*ix(i,jfin,k)+4] - 0.50 * ( (u[5*ix(i,jfin,k)+1])*(u[5*ix(i,jfin,k)+1]) + (u[5*ix(i,jfin,k)+2])*(u[5*ix(i,jfin,k)+2]) + (u[5*ix(i,jfin,k)+3])*(u[5*ix(i,jfin,k)+3]) ) / u[5*ix(i,jfin,k)+0] );
			}
		}
		frc2 = 0.0;
		for (k = ki1; k <= ki2-1; k++){
			for (i = ibeg; i <= ifin1; i++){
				frc2 = frc2 + ( phi1[i][k] + phi1[i+1][k] + phi1[i][k+1] + phi1[i+1][k+1] + phi2[i][k] + phi2[i+1][k] + phi2[i][k+1] + phi2[i+1][k+1] );
			}
		}
		frc2 = dxi * dzeta * frc2;
		for (k = ki1; k <= ki2; k++){
			for (j = jbeg; j <= jfin; j++){
				phi1[j][k] = c2*( u[5*ix(ibeg,j,k)+4] - 0.50 * ( (u[5*ix(ibeg,j,k)+1])*(u[5*ix(ibeg,j,k)+1]) + (u[5*ix(ibeg,j,k)+2])*(u[5*ix(ibeg,j,k)+2]) + (u[5*ix(ibeg,j,k)+3])*(u[5*ix(ibeg,j,k)+3]) ) / u[5*ix(ibeg,j,k)+0] );
			}
		}
		for (k = ki1; k <= ki2; k++){
			for (j = jbeg; j <= jfin; j++){
				phi2[j][k] = c2*( u[5*ix(ifin,j,k)+4] - 0.50 * ( (u[5*ix(ifin,j,k)+1])*(u[5*ix(ifin,j,k)+1]) + (u[5*ix(ifin,j,k)+2])*(u[5*ix(ifin,j,k)+2]) + (u[5*ix(ifin,j,k)+3])*(u[5*ix(ifin,j,k)+3]) ) / u[5*ix(ifin,j,k)+0] );
			}
		}
		frc3 = 0.0;
		for (k = ki1; k <= ki2-1; k++){
			for (j = jbeg; j <= jfin1; j++){
				frc3 = frc3 + ( phi1[j][k] + phi1[j+1][k] + phi1[j][k+1] + phi1[j+1][k+1] + phi2[j][k] + phi2[j+1][k] + phi2[j][k+1] + phi2[j+1][k+1] );
			}
		}
		frc3 = deta * dzeta * frc3;
		frc = 0.25 * ( frc1 + frc2 + frc3 );
	}

	private int verify(char CLASS){

		int verified = 0;
		double xcrref[] = new double[5];
		double xceref[] = new double[5];
		double xciref;
		double xcrdif[] = new double[5];
		double xcedif[] = new double[5];
		double xcidif;
		double dtref = 0;

		/* Tolerance level */
		double epsilon = 1.0E-8;

		verified = 1;

		for(int m = 0; m<5; m++){
			xcrref[m] = 1.0;
			xceref[m] = 1.0;
		}
		xciref = 1.0;

		/* As the original, the class is the one of the actual problem (grid and number of iterations) */
		if(!(nx0 == problemSize && ny0 == problemSize && nz0 == problemSize && itmax == itmaxDefault))
			CLASS = 'U';

		/* Reference values of RMS-norms of residual and of solution error, and of the surface integral */
		if(CLASS == 'S'){
			dtref = 5.0e-1;
			xcrref[0] = 1.6196343210976702e-02;
			xcrref[1] = 2.1976745164821318e-03;
			xcrref[2] = 1.5179927653399185e-03;
			xcrref[3] = 1.5029584435994323e-03;
			xcrref[4] = 3.4264073155896461e-02;
			xceref[0] = 6.4223319957960924e-04;
			xceref[1] = 8.4144342047347926e-05;
			xceref[2] = 5.8588269616485186e-05;
			xceref[3] = 5.8474222595157350e-05;
			xceref[4] = 1.3103347914111294e-03;
			xciref = 7.8418928865937083e+00;
		}
		else if(CLASS == 'W'){
			dtref = 1.5e-3;
			xcrref[0] = 0.1236511638192e+02;
			xcrref[1] = 0.1317228477799e+01;
			xcrref[2] = 0.2550120713095e+01;
			xcrref[3] = 0.2326187750252e+01;
			xcrref[4] = 0.2826799444189e+02;
			xceref[0] = 0.4867877144216e+00;
			xceref[1] = 0.5064652880982e-01;
			xceref[2] = 0.9281818101960e-01;
			xceref[3] = 0.8570126542733e-01;
			xceref[4] = 0.1084277417792e+01;
			xciref = 0.1161399311023e+02;
		}
		else if(CLASS == 'A'){
			dtref = 2.0e+0;
			xcrref[0] = 7.7902107606689367e+02;
			xcrref[1] = 6.3402765259692870e+01;
			xcrref[2] = 1.9499249727292479e+02;
			xcrref[3] = 1.7845301160418537e+02;
			xcrref[4] = 1.8384760349464247e+03;
			xceref[0] = 2.9964085685471943e+01;
			xceref[1] = 2.8194576365003349e+00;
			xceref[2] = 7.3473412698774742e+00;
			xceref[3] = 6.7139225687777051e+00;
			xceref[4] = 7.0715315688392578e+01;
			xciref = 2.6030925604886277e+01;
		}
		else if(CLASS == 'B'){
			dtref = 2.0e+0;
			xcrref[0] = 3.5532672969982736e+03;
			xcrref[1] = 2.6214750795310692e+02;
			xcrref[2] = 8.8333721850952190e+02;
			xcrref[3] = 7.7812774739425265e+02;
			xcrref[4] = 7.3087969592545314e+03;
			xceref[0] = 1.1401176380212709e+02;
			xceref[1] = 8.1098963655421574e+00;
			xceref[2] = 2.8480597317698308e+01;
			xceref[3] = 2.5905394567832939e+01;
			xceref[4] = 2.6054907504857413e+02;
			xciref = 4.7887162703308227e+01;
		}
		else if(CLASS == 'C'){
			dtref = 2.0e+0;
			xcrref[0] = 1.03766980323537846e+04;
			xcrref[1] = 8.92212458801008552e+02;
			xcrref[2] = 2.56238814582660871e+03;
			xcrref[3] = 2.19194343857831427e+03;
			xcrref[4] = 1.78078057261061185e+04;
			xceref[0] = 2.15986399716949279e+02;
			xceref[1] = 1.55789559239863600e+01;
			xceref[2] = 5.41318863077207766e+01;
			xceref[3] = 4.82262643154045421e+01;
			xceref[4] = 4.55902910043250358e+02;
			xciref = 6.66404553572181300e+01;
		}
		else if(CLASS == 'D'){
			dtref = 1.0e+0;
			xcrref[0] = 0.4868417937025e+05;
			xcrref[1] = 0.4696371050071e+04;
			xcrref[2] = 0.1218114549776e+05;
			xcrref[3] = 0.1033801493461e+05;
			xcrref[4] = 0.7142398413817e+05;
			xceref[0] = 0.3752393004482e+03;
			xceref[1] = 0.3084128893659e+02;
			xceref[2] = 0.9434276905469e+02;
			xceref[3] = 0.8230686681928e+02;
			xceref[4] = 0.7002620636210e+03;
			xciref = 0.8334101392503e+02;
		}
		else{
			CLASS = 'U';
			verified = 0;
		}

		/* Compute the difference of solution values and the known reference values */
		for(int m = 0; m<5; m++){
			xcrdif[m] = Math.abs((rsdnm[m]-xcrref[m])/xcrref[m]);
			xcedif[m] = Math.abs((errnm[m]-xceref[m])/xceref[m]);
		}
		xcidif = Math.abs((frc - xciref)/xciref);

		/* Output the comparison of computed results to known cases */
		if(CLASS != 'U'){
			System.out.println();
			System.out.println(" Verification being performed for class "+CLASS);
			System.out.println(" Accuracy setting for epsilon = "+epsilon);
			if(!(Math.abs(dt-dtref) <= epsilon)){
				verified = 0;
				CLASS = 'U';
				System.out.println(" DT does not match the reference value of "+dtref);
			}
		}
		else{
			System.out.println(" Unknown class");
		}

		if(CLASS != 'U'){
			System.out.println(" Comparison of RMS-norms of residual");
		}
		else{
			System.out.println(" RMS-norms of residual");
		}

		for(int m = 0; m<5; m++){
			if(CLASS == 'U'){
				System.out.println("\t"+(m+1)+"  "+rsdnm[m]);
			}
			else if(xcrdif[m] <= epsilon){
				System.out.println("\t"+(m+1)+"  "+rsdnm[m]+" "+xcrref[m]+" "+xcrdif[m]);
			}
			else{
				verified = 0;
				System.out.println("FAILURE: \t"+(m+1)+"  "+rsdnm[m]+" "+xcrref[m]+" "+xcrdif[m]);
			}
		}

		if(CLASS != 'U'){
			System.out.println(" Comparison of RMS-norms of solution error");
		}
		else{
			System.out.println(" RMS-norms of solution error");
		}

		for(int m = 0; m<5; m++){
			if(CLASS == 'U'){
				System.out.println("\t"+(m+1)+"  "+errnm[m]);
			}
			else if(xcedif[m] <= epsilon){
				System.out.println("\t"+(m+1)+"  "+errnm[m]+" "+xceref[m]+" "+xcedif[m]);
			}
			else{
				verified = 0;
				System.out.println("FAILURE: \t"+(m+1)+"  "+errnm[m]+" "+xceref[m]+" "+xcedif[m]);
			}
		}

		if(CLASS != 'U'){
			System.out.println(" Comparison of surface integral");
		}
		else{
			System.out.println(" Surface integral");
		}

		if(CLASS == 'U'){
			System.out.println("\t    "+frc);
		}
		else if(xcidif <= epsilon){
			System.out.println("\t    "+frc+" "+xciref+" "+xcidif);
		}
		else{
			verified = 0;
			System.out.println("FAILURE: \t    "+frc+" "+xciref+" "+xcidif);
		}

		if(CLASS == 'U'){
			System.out.println(" No reference values provided");
			System.out.println(" No verification performed");
		}
		else if(verified>0){
			System.out.println(" Verification Successful");
		}
		else{
			System.out.println(" Verification failed");
		}

		return verified;
	}

	private void printAdditionalTimers()
	{
		double tmax = maxtime;
		double t;

		if(tmax == 0.0)
			tmax = 1.0;

		System.out.format(java.util.Locale.ENGLISH, "  SECTION     Time (secs)\n");

		String names[] = {"total", "rhsx", "rhsy", "rhsz", "rhs", "jacld", "blts", "jacu", "buts", "add", "l2norm"};
		for(int i = 1; i<=T_LAST; i++){
			t = timer.read(i);
			if(i != T_JACLD && i != T_JACU)
				System.out.format(java.util.Locale.ENGLISH, "  %-8s:%9.3f  (%6.2f%%)\n", names[i-1], t, t*100./tmax);
			if(i == T_RHS){
				t = timer.read(T_RHSX) + timer.read(T_RHSY) + timer.read(T_RHSZ);
				System.out.format(java.util.Locale.ENGLISH, "     --> %-8s:%9.3f  (%6.2f%%)\n", "sub-rhs", t, t*100./tmax);
				t = timer.read(i) - t;
				System.out.format(java.util.Locale.ENGLISH, "     --> %-8s:%9.3f  (%6.2f%%)\n", "rest-rhs", t, t*100./tmax);
			}
		}
	}
}
