/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB 3.4.4 OpenMP (NPB3.4-OMP/SP)
 */
package NPB.SER;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import NPB.Benchmark;
import NPB.common.io.BMArgs;
import NPB.common.io.BMResults;
import NPB.common.timer.Timer;
import NPB.exceptions.BadClassException;
import NPB.exceptions.BadInputException;
import NPB.exceptions.NPBException;

public class SP implements Benchmark{

	private static final String BM_NAME="SP";
	private char CLASS;

	private boolean detailedTiming;

	private int problemSize;
	private int niter;
	private double dt;

	private int gridPoints[] = new int[3];
	private int nx2, ny2, nz2;

	/*
	 * Field arrays. As in the original (u(5,0:IMAXP,0:JMAXP,0:KMAX-1)), the
	 * first two dimensions are padded by 1 for even sizes and the 5 components
	 * of a point are contiguous: u(m,i,j,k) is u[5*ix(i,j,k)+m-1].
	 */
	private int ldx, ldy, sk;	/* ldx = IMAXP+1, ldy = JMAXP+1, sk = ldx*ldy */
	private double u[];
	private double rhs[];
	private double forcing[];
	private double us[];
	private double vs[];
	private double ws[];
	private double qs[];
	private double rhoI[];
	private double speed[];
	private double square[];

	/* work arrays of one grid line (the original lhs, lhsp, lhsm, cv, rhov) */
	/* work arrays of the line solves: lhs (rows 0..4), lhsp (5..9) and lhsm (10..14) of the point i at lh[15*i + row] */
	private double lh[];
	private double cv[];
	private double rhov[];

	private double ce[][] = new double[5][13];

	private double c1,c2,c3,c4,c5,c1c2,c1c5,c3c4;
	private double bt,dnxm1,dnym1,dnzm1,conz1;
	private double tx1,tx2,tx3,ty1,ty2,ty3,tz1,tz2,tz3;
	private double dx1,dx2,dx3,dx4,dx5,dy1,dy2,dy3,dy4,dy5,dz1,dz2,dz3,dz4,dz5;
	private double dxmax,dymax,dzmax,dssp,dttx1,dttx2,dtty1,dtty2,dttz1,dttz2;
	private double c2dttx1,c2dtty1,c2dttz1,dtdssp,comz1,comz4,comz5,comz6;
	private double c3c4tx3,c3c4ty3,c3c4tz3,dx1tx1,dx2tx1,dx3tx1,dx4tx1,dx5tx1;
	private double dy1ty1,dy2ty1,dy3ty1,dy4ty1,dy5ty1;
	private double dz1tz1,dz2tz1,dz3tz1,dz4tz1,dz5tz1;
	private double c2iv, con43, con16;
	private double xxcon1,xxcon2,xxcon3,xxcon4,xxcon5;
	private double yycon1,yycon2,yycon3,yycon4,yycon5;
	private double zzcon1,zzcon2,zzcon3,zzcon4,zzcon5;

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
	private final int T_XSOLVE = 6;
	private final int T_YSOLVE = 7;
	private final int T_ZSOLVE = 8;
	private final int T_TXINVR = 11;
	private final int T_PINVR = 12;
	private final int T_NINVR = 13;
	private final int T_TZETAR = 14;
	private final int T_ADD = 15;
	private final int T_LAST = 15;

	public static void main(String[] args) throws NPBException {
		SP sp = null;

		BMArgs.parseCmdLineArgs(args,BM_NAME);
		try{
			sp = new SP(BMArgs.CLASS,BMArgs.DETAILED_TIMING);
		}catch(OutOfMemoryError e){
			BMArgs.outOfMemoryMessage();
			throw e;
		}
		sp.runBenchmark(args);
	}

	public BMResults getResults(){
		return res;
	}

	public SP(char classSize, boolean dtiming){

		CLASS = classSize;
		detailedTiming = dtiming;
	}

	public void build(String[] args) throws NPBException{

		switch( CLASS )
		{
		case 'S':
			problemSize = 12;
			niter = 100;
			dt = 0.015;
			break;
		case 'W':
			problemSize = 36;
			niter = 400;
			dt = 0.0015;
			break;
		case 'A':
			problemSize = 64;
			niter = 400;
			dt = 0.0015;
			break;
		case 'B':
			problemSize = 102;
			niter = 400;
			dt = 0.001;
			break;
		case 'C':
			problemSize = 162;
			niter = 400;
			dt = 0.00067;
			break;
		case 'D':
			problemSize = 408;
			niter = 500;
			dt = 0.0003;
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

		readData();

		System.out.println(" Size:  "+gridPoints[0]+"x "+gridPoints[1]+"x "+gridPoints[2]);
		System.out.println(" Iterations:  "+niter+"    dt:  "+dt);
		System.out.println();

		if( gridPoints[0] > problemSize ||
				gridPoints[1] > problemSize ||
				gridPoints[2] > problemSize ){
			System.err.println(" Problem size too big for compiled array sizes");
			terminate();
			return;
		}
		nx2 = gridPoints[0] - 2;
		ny2 = gridPoints[1] - 2;
		nz2 = gridPoints[2] - 2;

		for(int i = 1; i<=T_LAST; i++)
			timer.reset(i);

		allocSpace();

		setConstants();

		exactRHS();

		initialize();

		/* Do one time step to touch all code, and reinitialize */
		adi();

		initialize();

		for(int i = 1; i<=T_LAST; i++)
			timer.reset(i);

		timer.start(T_TOTAL);

		for(int step = 1; step<=niter; step++){
			if(step == 1 || step%20 == 0)
				System.out.println(" Time step "+step);
			adi();
		}

		timer.stop(T_TOTAL);

		double times[] = {0,0};
		times[0] = timer.read(T_TOTAL);

		int verified = verify(niter, CLASS);

		double n3 = (double)gridPoints[0]*gridPoints[1]*gridPoints[2];
		double t = (gridPoints[0]+gridPoints[1]+gridPoints[2])/3.0;
		double mflops = 0;
		if(times[0] != 0)
			/* (the constants are REAL(4) literals in the original, as in FT; here double) */
			mflops = 1.0E-6*(double)niter*(881.174*n3
					-4683.91*(t*t)
					+11484.5*t
					-19272.4)/times[0];

		res = new BMResults(BM_NAME,
				CLASS,
				gridPoints[0],
				gridPoints[1],
				gridPoints[2],
				niter,
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
	 * Reads the optional input file inputsp.data (niter, dt and grid
	 * size, as the original) or takes the compiled defaults of the class
	 */
	private void readData(){

		boolean inputData = true;

		BMArgs.bannerSER(BM_NAME, CLASS);

		try{

			BufferedReader input = new BufferedReader(new FileReader("inputsp.data"));

			System.out.println(" Reading from input file inputsp.data");

			String line[] = input.readLine().trim().split("\\s+");
			niter = Integer.valueOf(line[0]);

			line = input.readLine().trim().split("\\s+");
			/* the sample file writes dt in Fortran form (0.0015d0) */
			dt = Double.valueOf(line[0].replace('d', 'e').replace('D', 'E'));

			line = input.readLine().trim().split("\\s+");
			gridPoints[0] = Integer.valueOf(line[0]);
			gridPoints[1] = Integer.valueOf(line[1]);
			gridPoints[2] = Integer.valueOf(line[2]);

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
			System.out.println(" No input file inputsp.data. Using compiled defaults");
			gridPoints[0] = problemSize;
			gridPoints[1] = problemSize;
			gridPoints[2] = problemSize;
		}
	}

	/* Index of the scalar field point (i,j,k) */
	private int ix(int i, int j, int k){
		return i + ldx*(j + ldy*k);
	}

	private void allocSpace() throws NPBException{

		int imaxp = problemSize/2*2;
		int jmaxp = problemSize/2*2;
		int kmax = problemSize;

		ldx = imaxp + 1;
		ldy = jmaxp + 1;
		sk = ldx*ldy;

		long points = (long)ldx*ldy*kmax;
		if(5*points > Integer.MAX_VALUE)
			throw new BadInputException("Problem size too big for Java arrays");

		int n = (int)points;

		u = new double[5*n];
		rhs = new double[5*n];
		forcing = new double[5*n];
		us = new double[n];
		vs = new double[n];
		ws = new double[n];
		qs = new double[n];
		rhoI = new double[n];
		speed = new double[n];
		square = new double[n];

		int nmax = problemSize + 1;
		lh = new double[15*nmax];
		cv = new double[nmax];
		rhov = new double[nmax];
	}

	private void setConstants(){

		ce[0][0]  = 2.0*1.0;
		ce[0][1]  = 0.0;
		ce[0][2]  = 0.0;
		ce[0][3]  = 4.0;
		ce[0][4]  = 5.0;
		ce[0][5]  = 3.0;
		ce[0][6]  = 0.5;
		ce[0][7]  = 0.02;
		ce[0][8]  = 0.01;
		ce[0][9]  = 0.03;
		ce[0][10] = 0.5;
		ce[0][11] = 0.4;
		ce[0][12] = 0.3;

		ce[1][0]  = 1.0;
		ce[1][1]  = 0.0;
		ce[1][2]  = 0.0;
		ce[1][3]  = 0.0;
		ce[1][4]  = 1.0;
		ce[1][5]  = 2.0;
		ce[1][6]  = 3.0;
		ce[1][7]  = 0.01;
		ce[1][8]  = 0.03;
		ce[1][9]  = 0.02;
		ce[1][10] = 0.4;
		ce[1][11] = 0.3;
		ce[1][12] = 0.5;

		ce[2][0]  = 2.0;
		ce[2][1]  = 2.0;
		ce[2][2]  = 0.0;
		ce[2][3]  = 0.0;
		ce[2][4]  = 0.0;
		ce[2][5]  = 2.0;
		ce[2][6]  = 3.0;
		ce[2][7]  = 0.04;
		ce[2][8]  = 0.03;
		ce[2][9]  = 0.05;
		ce[2][10] = 0.3;
		ce[2][11] = 0.5;
		ce[2][12] = 0.4;

		ce[3][0]  = 2.0;
		ce[3][1]  = 2.0;
		ce[3][2]  = 0.0;
		ce[3][3]  = 0.0;
		ce[3][4]  = 0.0;
		ce[3][5]  = 2.0;
		ce[3][6]  = 3.0;
		ce[3][7]  = 0.03;
		ce[3][8]  = 0.05;
		ce[3][9]  = 0.04;
		ce[3][10] = 0.2;
		ce[3][11] = 0.1;
		ce[3][12] = 0.3;

		ce[4][0]  = 5.0;
		ce[4][1]  = 4.0;
		ce[4][2]  = 3.0;
		ce[4][3]  = 2.0;
		ce[4][4]  = 0.1;
		ce[4][5]  = 0.4;
		ce[4][6]  = 0.3;
		ce[4][7]  = 0.05;
		ce[4][8]  = 0.04;
		ce[4][9]  = 0.03;
		ce[4][10] = 0.1;
		ce[4][11] = 0.3;
		ce[4][12] = 0.2;

		c1 = 1.4;
		c2 = 0.4;
		c3 = 0.1;
		c4 = 1.0;
		c5 = 1.4;

		bt = Math.sqrt(0.5);

		dnxm1 = 1.0 / (gridPoints[0]-1);
		dnym1 = 1.0 / (gridPoints[1]-1);
		dnzm1 = 1.0 / (gridPoints[2]-1);

		c1c2 = c1 * c2;
		c1c5 = c1 * c5;
		c3c4 = c3 * c4;

		conz1 = (1.0-c1c5);

		tx1 = 1.0 / (dnxm1 * dnxm1);
		tx2 = 1.0 / (2.0 * dnxm1);
		tx3 = 1.0 / dnxm1;

		ty1 = 1.0 / (dnym1 * dnym1);
		ty2 = 1.0 / (2.0 * dnym1);
		ty3 = 1.0 / dnym1;

		tz1 = 1.0 / (dnzm1 * dnzm1);
		tz2 = 1.0 / (2.0 * dnzm1);
		tz3 = 1.0 / dnzm1;

		dx1 = 0.75;
		dx2 = 0.75;
		dx3 = 0.75;
		dx4 = 0.75;
		dx5 = 0.75;

		dy1 = 0.75;
		dy2 = 0.75;
		dy3 = 0.75;
		dy4 = 0.75;
		dy5 = 0.75;

		dz1 = 1.0;
		dz2 = 1.0;
		dz3 = 1.0;
		dz4 = 1.0;
		dz5 = 1.0;

		dxmax = dmax1(dx3, dx4);
		dymax = dmax1(dy2, dy4);
		dzmax = dmax1(dz2, dz3);

		dssp = 0.25 * dmax1(dx1, dmax1(dy1, dz1) );

		dttx1 = dt*tx1;
		dttx2 = dt*tx2;
		dtty1 = dt*ty1;
		dtty2 = dt*ty2;
		dttz1 = dt*tz1;
		dttz2 = dt*tz2;

		c2dttx1 = 2.0*dttx1;
		c2dtty1 = 2.0*dtty1;
		c2dttz1 = 2.0*dttz1;

		dtdssp = dt*dssp;

		comz1  = dtdssp;
		comz4  = 4.0*dtdssp;
		comz5  = 5.0*dtdssp;
		comz6  = 6.0*dtdssp;

		c3c4tx3 = c3c4*tx3;
		c3c4ty3 = c3c4*ty3;
		c3c4tz3 = c3c4*tz3;

		dx1tx1 = dx1*tx1;
		dx2tx1 = dx2*tx1;
		dx3tx1 = dx3*tx1;
		dx4tx1 = dx4*tx1;
		dx5tx1 = dx5*tx1;

		dy1ty1 = dy1*ty1;
		dy2ty1 = dy2*ty1;
		dy3ty1 = dy3*ty1;
		dy4ty1 = dy4*ty1;
		dy5ty1 = dy5*ty1;

		dz1tz1 = dz1*tz1;
		dz2tz1 = dz2*tz1;
		dz3tz1 = dz3*tz1;
		dz4tz1 = dz4*tz1;
		dz5tz1 = dz5*tz1;

		c2iv  = 2.5;
		con43 = 4.0/3.0;
		con16 = 1.0/6.0;

		xxcon1 = c3c4tx3*con43*tx3;
		xxcon2 = c3c4tx3*tx3;
		xxcon3 = c3c4tx3*conz1*tx3;
		xxcon4 = c3c4tx3*con16*tx3;
		xxcon5 = c3c4tx3*c1c5*tx3;

		yycon1 = c3c4ty3*con43*ty3;
		yycon2 = c3c4ty3*ty3;
		yycon3 = c3c4ty3*conz1*ty3;
		yycon4 = c3c4ty3*con16*ty3;
		yycon5 = c3c4ty3*c1c5*ty3;

		zzcon1 = c3c4tz3*con43*tz3;
		zzcon2 = c3c4tz3*tz3;
		zzcon3 = c3c4tz3*conz1*tz3;
		zzcon4 = c3c4tz3*con16*tz3;
		zzcon5 = c3c4tz3*c1c5*tz3;

	}

	/*
	 * This method returns the exact solution at point xi, eta, zeta
	 */
	private void exactSolution(double xi, double eta, double zeta, double dtemp[]){
		for(int m = 0; m<5; m++){
			dtemp[m] = ce[m][0]+
			xi*(ce[m][1] + xi*(ce[m][4] + xi*(ce[m][7] + xi*ce[m][10]))) +
			eta*(ce[m][2] + eta*(ce[m][5] + eta*(ce[m][8] +eta*ce[m][11]))) +
			zeta*(ce[m][3] + zeta*(ce[m][6] + zeta*(ce[m][9] + zeta*ce[m][12])));
		}
	}

	/*
	 * This method initializes the field variable u using
	 * tri-linear transfinite interpolation of the boundary values
	 */
	private void initialize(){

		int i, j, k, m, fx, fy, fz;
		double xi, eta, zeta, Pxi, Peta, Pzeta;
		double Pface[][][] = new double[2][3][5];	/* Pface(5,3,2) */
		double temp[] = new double[5];

		/* Later (in compute_rhs) we compute 1/u for every element. A few of
		 * the corner elements are not used, but it convenient (and faster)
		 * to compute the whole thing with a simple loop. Make sure those
		 * values are nonzero by initializing the whole thing here. */
		for(k = 0; k<=gridPoints[2]-1; k++){
			for(j = 0; j<=gridPoints[1]-1; j++){
				for(i = 0; i<=gridPoints[0]-1; i++){
					int c = 5*ix(i,j,k);
					u[c]   = 1.0;
					u[c+1] = 0.0;
					u[c+2] = 0.0;
					u[c+3] = 0.0;
					u[c+4] = 1.0;
				}
			}
		}

		/* first store the "interpolated" values everywhere on the grid */
		for(k = 0; k<=gridPoints[2]-1; k++){
			zeta = (double)k * dnzm1;
			for(j = 0; j<=gridPoints[1]-1; j++){
				eta = (double)j * dnym1;
				for(i = 0; i<=gridPoints[0]-1; i++){
					xi = (double)i * dnxm1;

					for(fx = 0; fx<2; fx++){
						Pxi = (double)fx;
						exactSolution(Pxi, eta, zeta, Pface[fx][0]);
					}

					for(fy = 0; fy<2; fy++){
						Peta = (double)fy;
						exactSolution(xi, Peta, zeta, Pface[fy][1]);
					}

					for(fz = 0; fz<2; fz++){
						Pzeta = (double)fz;
						exactSolution(xi, eta, Pzeta, Pface[fz][2]);
					}

					int c = 5*ix(i,j,k);
					for(m = 0; m<5; m++){
						Pxi   = xi   * Pface[1][0][m] +
						(1.0-xi)   * Pface[0][0][m];
						Peta  = eta  * Pface[1][1][m] +
						(1.0-eta)  * Pface[0][1][m];
						Pzeta = zeta * Pface[1][2][m] +
						(1.0-zeta) * Pface[0][2][m];

						u[c+m] = Pxi + Peta + Pzeta -
						Pxi*Peta - Pxi*Pzeta - Peta*Pzeta +
						Pxi*Peta*Pzeta;
					}
				}
			}
		}

		/* now store the exact values on the boundaries */

		/* west face */
		xi = 0.0;
		i  = 0;
		for(k = 0; k<=gridPoints[2]-1; k++){
			zeta = (double)k * dnzm1;
			for(j = 0; j<=gridPoints[1]-1; j++){
				eta = (double)j * dnym1;
				exactSolution(xi, eta, zeta, temp);
				int c = 5*ix(i,j,k);
				for(m = 0; m<5; m++)
					u[c+m] = temp[m];
			}
		}

		/* east face */
		xi = 1.0;
		i  = gridPoints[0]-1;
		for(k = 0; k<=gridPoints[2]-1; k++){
			zeta = (double)k * dnzm1;
			for(j = 0; j<=gridPoints[1]-1; j++){
				eta = (double)j * dnym1;
				exactSolution(xi, eta, zeta, temp);
				int c = 5*ix(i,j,k);
				for(m = 0; m<5; m++)
					u[c+m] = temp[m];
			}
		}

		/* south face */
		eta = 0.0;
		j   = 0;
		for(k = 0; k<=gridPoints[2]-1; k++){
			zeta = (double)k * dnzm1;
			for(i = 0; i<=gridPoints[0]-1; i++){
				xi = (double)i * dnxm1;
				exactSolution(xi, eta, zeta, temp);
				int c = 5*ix(i,j,k);
				for(m = 0; m<5; m++)
					u[c+m] = temp[m];
			}
		}

		/* north face */
		eta = 1.0;
		j   = gridPoints[1]-1;
		for(k = 0; k<=gridPoints[2]-1; k++){
			zeta = (double)k * dnzm1;
			for(i = 0; i<=gridPoints[0]-1; i++){
				xi = (double)i * dnxm1;
				exactSolution(xi, eta, zeta, temp);
				int c = 5*ix(i,j,k);
				for(m = 0; m<5; m++)
					u[c+m] = temp[m];
			}
		}

		/* bottom face */
		zeta = 0.0;
		k    = 0;
		for(j = 0; j<=gridPoints[1]-1; j++){
			eta = (double)j * dnym1;
			for(i = 0; i<=gridPoints[0]-1; i++){
				xi = (double)i * dnxm1;
				exactSolution(xi, eta, zeta, temp);
				int c = 5*ix(i,j,k);
				for(m = 0; m<5; m++)
					u[c+m] = temp[m];
			}
		}

		/* top face */
		zeta = 1.0;
		k    = gridPoints[2]-1;
		for(j = 0; j<=gridPoints[1]-1; j++){
			eta = (double)j * dnym1;
			for(i = 0; i<=gridPoints[0]-1; i++){
				xi = (double)i * dnxm1;
				exactSolution(xi, eta, zeta, temp);
				int c = 5*ix(i,j,k);
				for(m = 0; m<5; m++)
					u[c+m] = temp[m];
			}
		}
	}

	/*
	 * This method computes the forcing function (the original exact_rhs).
	 * The three sweeps (x, y and z) of the original have the same form
	 * with the roles of the velocity components rotated, so they are written
	 * once and parametrized by the direction (0, 1, 2).
	 */
	private void exactRHS(){

		int i, j, k, m;

		for(k = 0; k<=gridPoints[2]-1; k++){
			for(j = 0; j<=gridPoints[1]-1; j++){
				for(i = 0; i<=gridPoints[0]-1; i++){
					int c = 5*ix(i,j,k);
					for(m = 0; m<5; m++)
						forcing[c+m] = 0.0;
				}
			}
		}

		/* xi-direction flux differences */
		for(k = 1; k<=gridPoints[2]-2; k++)
			for(j = 1; j<=gridPoints[1]-2; j++)
				exactRHSLine(0, 0, j, k);

		/* eta-direction flux differences */
		for(k = 1; k<=gridPoints[2]-2; k++)
			for(i = 1; i<=gridPoints[0]-2; i++)
				exactRHSLine(1, i, 0, k);

		/* zeta-direction flux differences */
		for(j = 1; j<=gridPoints[1]-2; j++)
			for(i = 1; i<=gridPoints[0]-2; i++)
				exactRHSLine(2, i, j, 0);

		/* now change the sign of the forcing function */
		for(k = 1; k<=gridPoints[2]-2; k++){
			for(j = 1; j<=gridPoints[1]-2; j++){
				for(i = 1; i<=gridPoints[0]-2; i++){
					int c = 5*ix(i,j,k);
					for(m = 0; m<5; m++)
						forcing[c+m] = -1.0 * forcing[c+m];
				}
			}
		}
	}

	/*
	 * One grid line of the exact_rhs sweep in direction dir. (i0,j0,k0)
	 * are the fixed coordinates; the one of the direction is ignored.
	 */
	private void exactRHSLine(int dir, int i0, int j0, int k0){

		int n = gridPoints[dir];
		int sl = (dir == 0) ? 1 : ((dir == 1) ? ldx : sk);	/* stride of the line */
		int o0 = ix(dir == 0 ? 0 : i0, dir == 1 ? 0 : j0, dir == 2 ? 0 : k0);

		int vd = dir + 1;				/* velocity component along dir */
		int o1 = (dir == 0) ? 2 : 1;		/* the other two, in increasing order */
		int o2 = (dir == 2) ? 2 : 3;

		double t2, dd[], con1, con2, con3, con4, con5;
		double dxx[] = {dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1};
		double dyy[] = {dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1};
		double dzz[] = {dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1};
		if(dir == 0){
			t2 = tx2; dd = dxx;
			con1 = xxcon1; con2 = xxcon2; con3 = xxcon3; con4 = xxcon4; con5 = xxcon5;
		}
		else if(dir == 1){
			t2 = ty2; dd = dyy;
			con1 = yycon1; con2 = yycon2; con3 = yycon3; con4 = yycon4; con5 = yycon5;
		}
		else{
			t2 = tz2; dd = dzz;
			con1 = zzcon1; con2 = zzcon2; con3 = zzcon3; con4 = zzcon4; con5 = zzcon5;
		}

		double dtemp[] = new double[5];
		double ue[][] = new double[5][n];
		double buf[][] = new double[5][n];
		double cuf[] = new double[n];
		double q[] = new double[n];
		double xi, eta, zeta, dtpp;
		int l, m;

		for(l = 0; l<=n-1; l++){
			int ci = (dir == 0) ? l : i0;
			int cj = (dir == 1) ? l : j0;
			int ck = (dir == 2) ? l : k0;
			xi   = (double)ci * dnxm1;
			eta  = (double)cj * dnym1;
			zeta = (double)ck * dnzm1;

			exactSolution(xi, eta, zeta, dtemp);

			for(m = 0; m<5; m++)
				ue[m][l] = dtemp[m];

			dtpp = 1.0 / dtemp[0];

			for(m = 1; m<5; m++)
				buf[m][l] = dtpp * dtemp[m];

			cuf[l]    = buf[vd][l] * buf[vd][l];
			buf[0][l] = cuf[l] + buf[o1][l] * buf[o1][l] +
			buf[o2][l] * buf[o2][l];
			q[l] = 0.5*(buf[1][l]*ue[1][l] + buf[2][l]*ue[2][l] +
					buf[3][l]*ue[3][l]);
		}

		for(l = 1; l<=n-2; l++){
			int lm1 = l-1;
			int lp1 = l+1;
			int c = 5*(o0+sl*l);

			forcing[c] = forcing[c] -
			t2*( ue[vd][lp1]-ue[vd][lm1] )+
			dd[0]*(ue[0][lp1]-2.0*ue[0][l]+ue[0][lm1]);

			for(m = 1; m<=3; m++){
				if(m == vd){
					forcing[c+m] = forcing[c+m] - t2 * (
							(ue[m][lp1]*buf[m][lp1]+c2*(ue[4][lp1]-q[lp1]))-
							(ue[m][lm1]*buf[m][lm1]+c2*(ue[4][lm1]-q[lm1])))+
							con1*(buf[m][lp1]-2.0*buf[m][l]+buf[m][lm1])+
							dd[m]*( ue[m][lp1]-2.0* ue[m][l]+ue[m][lm1]);
				}
				else{
					forcing[c+m] = forcing[c+m] - t2 * (
							ue[m][lp1]*buf[vd][lp1]-ue[m][lm1]*buf[vd][lm1])+
							con2*(buf[m][lp1]-2.0*buf[m][l]+buf[m][lm1])+
							dd[m]*( ue[m][lp1]-2.0*ue[m][l] +ue[m][lm1]);
				}
			}

			forcing[c+4] = forcing[c+4] - t2*(
					buf[vd][lp1]*(c1*ue[4][lp1]-c2*q[lp1])-
					buf[vd][lm1]*(c1*ue[4][lm1]-c2*q[lm1]))+
					0.5*con3*(buf[0][lp1]-2.0*buf[0][l]+
							buf[0][lm1])+
							con4*(cuf[lp1]-2.0*cuf[l]+cuf[lm1])+
							con5*(buf[4][lp1]-2.0*buf[4][l]+buf[4][lm1])+
							dd[4]*( ue[4][lp1]-2.0* ue[4][l]+ ue[4][lm1]);
		}

		/* Fourth-order dissipation */
		for(m = 0; m<5; m++){
			l = 1;
			int c = 5*(o0+sl*l);
			forcing[c+m] = forcing[c+m] - dssp *
			(5.0*ue[m][l] - 4.0*ue[m][l+1] +ue[m][l+2]);
			l = 2;
			c = 5*(o0+sl*l);
			forcing[c+m] = forcing[c+m] - dssp *
			(-4.0*ue[m][l-1] + 6.0*ue[m][l] -
					4.0*ue[m][l+1] +       ue[m][l+2]);
		}

		for(m = 0; m<5; m++){
			for(l = 3; l<=n-4; l++){
				int c = 5*(o0+sl*l);
				forcing[c+m] = forcing[c+m] - dssp*
				(ue[m][l-2] - 4.0*ue[m][l-1] +
						6.0*ue[m][l] - 4.0*ue[m][l+1] + ue[m][l+2]);
			}
		}

		for(m = 0; m<5; m++){
			l = n-3;
			int c = 5*(o0+sl*l);
			forcing[c+m] = forcing[c+m] - dssp *
			(ue[m][l-2] - 4.0*ue[m][l-1] +
					6.0*ue[m][l] - 4.0*ue[m][l+1]);
			l = n-2;
			c = 5*(o0+sl*l);
			forcing[c+m] = forcing[c+m] - dssp *
			(ue[m][l-2] - 4.0*ue[m][l-1] + 5.0*ue[m][l]);
		}
	}

	private void adi(){

		computeRHS();

		txinvr();

		lineSolve(0);	/* x_solve (and ninvr) */

		lineSolve(1);	/* y_solve (and pinvr) */

		lineSolve(2);	/* z_solve (and tzetar) */

		add();
	}

	/*
	 * Fourth-order dissipation of one point, in the five variants of the
	 * original (first, second, interior, last but one and last point of the
	 * line). c is the index of the first component of the point and s the
	 * distance (in elements of u) to the next point along the line.
	 */
	private void dissipFirst(int c, int s){
		for(int m = 0; m<5; m++){
			rhs[c+m] = rhs[c+m] - dssp *
			( 5.0*u[c+m] - 4.0*u[c+s+m] +
					u[c+2*s+m]);
		}
	}

	private void dissipSecond(int c, int s){
		for(int m = 0; m<5; m++){
			rhs[c+m] = rhs[c+m] - dssp *
			(-4.0*u[c-s+m] + 6.0*u[c+m] -
					4.0*u[c+s+m] + u[c+2*s+m]);
		}
	}

	private void dissipInterior(int c, int s){
		for(int m = 0; m<5; m++){
			rhs[c+m] = rhs[c+m] - dssp *
			(  u[c-2*s+m] - 4.0*u[c-s+m] +
					6.0*u[c+m] - 4.0*u[c+s+m] +
					u[c+2*s+m] );
		}
	}

	private void dissipLast2(int c, int s){
		for(int m = 0; m<5; m++){
			rhs[c+m] = rhs[c+m] - dssp *
			( u[c-2*s+m] - 4.0*u[c-s+m] +
					6.0*u[c+m] - 4.0*u[c+s+m] );
		}
	}

	private void dissipLast(int c, int s){
		for(int m = 0; m<5; m++){
			rhs[c+m] = rhs[c+m] - dssp *
			( u[c-2*s+m] - 4.0*u[c-s+m] +
					5.0*u[c+m] );
		}
	}

	/*
	 * This method computes the right hand side of the system
	 */
	private void computeRHS(){

		int i, j, k, m;
		double aux, rhoInv;

		if (detailedTiming) timer.start(T_RHS);

		/* compute the reciprocal of density, and the kinetic energy,
		 * and the speed of sound. */
		for(k = 0; k<=gridPoints[2]-1; k++){
			for(j = 0; j<=gridPoints[1]-1; j++){
				for(i = 0; i<=gridPoints[0]-1; i++){
					int p = ix(i,j,k);
					int c = 5*p;
					final double u0 = u[c], u1 = u[c+1], u2 = u[c+2], u3 = u[c+3], u4 = u[c+4];
					rhoInv = 1.0/u0;
					rhoI[p] = rhoInv;
					us[p] = u1 * rhoInv;
					vs[p] = u2 * rhoInv;
					ws[p] = u3 * rhoInv;
					final double sq = 0.5* (
							u1*u1 +
							u2*u2 +
							u3*u3 ) * rhoInv;
					square[p] = sq;
					qs[p] = sq * rhoInv;
					/* (don't need speed and ainx until the lhs computation) */
					aux = c1c2*rhoInv* (u4 - sq);
					speed[p] = Math.sqrt(aux);
				}
			}
		}

		/* copy the exact forcing term to the right hand side; because
		 * this forcing term is known, we can store it on the whole grid
		 * including the boundary */
		for(k = 0; k<=nz2+1; k++){
			for(j = 0; j<=ny2+1; j++){
				for(i = 0; i<=nx2+1; i++){
					int c = 5*ix(i,j,k);
					for(m = 0; m<5; m++){
						rhs[c+m] = forcing[c+m];
					}
				}
			}
		}

		/* compute xi-direction fluxes */
		if (detailedTiming) timer.start(T_RHSX);
		for(k = 1; k<=nz2; k++){
			for(j = 1; j<=ny2; j++){
				rhsX(j, k);
			}
		}
		if (detailedTiming) timer.stop(T_RHSX);

		/* compute eta-direction fluxes */
		if (detailedTiming) timer.start(T_RHSY);
		for(k = 1; k<=nz2; k++){
			for(j = 1; j<=ny2; j++){
				rhsY(j, k);
			}
		}
		if (detailedTiming) timer.stop(T_RHSY);

		/* compute zeta-direction fluxes */
		if (detailedTiming) timer.start(T_RHSZ);
		for(k = 1; k<=nz2; k++){
			for(j = 1; j<=ny2; j++){
				rhsZ(j, k);
			}
		}
		if (detailedTiming) timer.stop(T_RHSZ);

		for(k = 1; k<=nz2; k++){
			for(j = 1; j<=ny2; j++){
				for(i = 1; i<=nx2; i++){
					int c = 5*ix(i,j,k);
					for(m = 0; m<5; m++){
						rhs[c+m] = rhs[c+m] * dt;
					}
				}
			}
		}

		if (detailedTiming) timer.stop(T_RHS);
	}

	/* one line (j,k) of the x sweep of compute_rhs: fluxes and fourth order dissipation */
	private void rhsX(int j, int k){
		int i;
		double uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;
		for(i = 1; i<=nx2; i++){
			int p = ix(i,j,k);
			int c = 5*p;
			int cp = c + 5;
			int cm = c - 5;
			uijk = us[p];
			up1  = us[p+1];
			um1  = us[p-1];
					final double uC0 = u[c], uC1 = u[c+1], uC2 = u[c+2], uC3 = u[c+3], uC4 = u[c+4];
					final double uP0 = u[cp], uP1 = u[cp+1], uP2 = u[cp+2], uP3 = u[cp+3], uP4 = u[cp+4];
					final double uM0 = u[cm], uM1 = u[cm+1], uM2 = u[cm+2], uM3 = u[cm+3], uM4 = u[cm+4];
			double r0 = rhs[c] + dx1tx1 *
			(uP0 - 2.0*uC0 +
					uM0) -
					tx2 * (uP1 - uM1);
			double r1 = rhs[c+1] + dx2tx1 *
			(uP1 - 2.0*uC1 +
					uM1) +
					xxcon2*con43 * (up1 - 2.0*uijk + um1) -
					tx2 * (uP1*up1 -
							uM1*um1 +
							(uP4- square[p+1]-
									uM4+ square[p-1])*
									c2);
			double r2 = rhs[c+2] + dx3tx1 *
			(uP2 - 2.0*uC2 +
					uM2) +
					xxcon2 * (vs[p+1] - 2.0*vs[p] +
							vs[p-1]) -
							tx2 * (uP2*up1 -
									uM2*um1);
			double r3 = rhs[c+3] + dx4tx1 *
			(uP3 - 2.0*uC3 +
					uM3) +
					xxcon2 * (ws[p+1] - 2.0*ws[p] +
							ws[p-1]) -
							tx2 * (uP3*up1 -
									uM3*um1);
			double r4 = rhs[c+4] + dx5tx1 *
			(uP4 - 2.0*uC4 +
					uM4) +
					xxcon3 * (qs[p+1] - 2.0*qs[p] +
							qs[p-1]) +
							xxcon4 * (up1*up1 -       2.0*uijk*uijk +
									um1*um1) +
									xxcon5 * (uP4*rhoI[p+1] -
											2.0*uC4*rhoI[p] +
											uM4*rhoI[p-1]) -
											tx2 * ( (c1*uP4 -
													c2*square[p+1])*up1 -
													(c1*uM4 -
															c2*square[p-1])*um1 );
				if(i >= 3 && i <= nx2-2){
					/* fourth order dissipation of an interior point (dissipInterior), fused */
					final int cpp = cp + (cp - c), cmm = cm - (cp - c);
					r0 = r0 - dssp *
					(  u[cmm] - 4.0*uM0 +
							6.0*uC0 - 4.0*uP0 +
							u[cpp] );
					r1 = r1 - dssp *
					(  u[cmm+1] - 4.0*uM1 +
							6.0*uC1 - 4.0*uP1 +
							u[cpp+1] );
					r2 = r2 - dssp *
					(  u[cmm+2] - 4.0*uM2 +
							6.0*uC2 - 4.0*uP2 +
							u[cpp+2] );
					r3 = r3 - dssp *
					(  u[cmm+3] - 4.0*uM3 +
							6.0*uC3 - 4.0*uP3 +
							u[cpp+3] );
					r4 = r4 - dssp *
					(  u[cmm+4] - 4.0*uM4 +
							6.0*uC4 - 4.0*uP4 +
							u[cpp+4] );
				}
				rhs[c] = r0;
				rhs[c+1] = r1;
				rhs[c+2] = r2;
				rhs[c+3] = r3;
				rhs[c+4] = r4;

		}
		/* add fourth order xi-direction dissipation */
		dissipFirst(5*ix(1,j,k), 5);
		dissipSecond(5*ix(2,j,k), 5);
		dissipLast2(5*ix(nx2-1,j,k), 5);
		dissipLast(5*ix(nx2,j,k), 5);
	}

	/* one line (j,k) of the y sweep of compute_rhs: fluxes and fourth order dissipation */
	private void rhsY(int j, int k){
		int i;
		double uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;
		for(i = 1; i<=nx2; i++){
			int p = ix(i,j,k);
			int c = 5*p;
			int cp = c + 5*ldx;
			int cm = c - 5*ldx;
			int pp = p + ldx;
			int pm = p - ldx;
			vijk = vs[p];
			vp1  = vs[pp];
			vm1  = vs[pm];
			rhs[c] = rhs[c] + dy1ty1 *
			(u[cp] - 2.0*u[c] +
					u[cm]) -
					ty2 * (u[cp+2] - u[cm+2]);
			rhs[c+1] = rhs[c+1] + dy2ty1 *
			(u[cp+1] - 2.0*u[c+1] +
					u[cm+1]) +
					yycon2 * (us[pp] - 2.0*us[p] +
							us[pm]) -
							ty2 * (u[cp+1]*vp1 -
									u[cm+1]*vm1);
			rhs[c+2] = rhs[c+2] + dy3ty1 *
			(u[cp+2] - 2.0*u[c+2] +
					u[cm+2]) +
					yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
					ty2 * (u[cp+2]*vp1 -
							u[cm+2]*vm1 +
							(u[cp+4] - square[pp] -
									u[cm+4] + square[pm])
									*c2);
			rhs[c+3] = rhs[c+3] + dy4ty1 *
			(u[cp+3] - 2.0*u[c+3] +
					u[cm+3]) +
					yycon2 * (ws[pp] - 2.0*ws[p] +
							ws[pm]) -
							ty2 * (u[cp+3]*vp1 -
									u[cm+3]*vm1);
			rhs[c+4] = rhs[c+4] + dy5ty1 *
			(u[cp+4] - 2.0*u[c+4] +
					u[cm+4]) +
					yycon3 * (qs[pp] - 2.0*qs[p] +
							qs[pm]) +
							yycon4 * (vp1*vp1       - 2.0*vijk*vijk +
									vm1*vm1) +
									yycon5 * (u[cp+4]*rhoI[pp] -
											2.0*u[c+4]*rhoI[p] +
											u[cm+4]*rhoI[pm]) -
											ty2 * ((c1*u[cp+4] -
													c2*square[pp]) * vp1 -
													(c1*u[cm+4] -
															c2*square[pm]) * vm1);
		}
		/* add fourth order eta-direction dissipation */
		if(j == 1){
			for(i = 1; i<=nx2; i++)
				dissipFirst(5*ix(i,j,k), 5*ldx);
		}
		else if(j == 2){
			for(i = 1; i<=nx2; i++)
				dissipSecond(5*ix(i,j,k), 5*ldx);
		}
		else if(j == ny2-1){
			for(i = 1; i<=nx2; i++)
				dissipLast2(5*ix(i,j,k), 5*ldx);
		}
		else if(j == ny2){
			for(i = 1; i<=nx2; i++)
				dissipLast(5*ix(i,j,k), 5*ldx);
		}
		else{
			dissipInteriorLine(5*ix(1,j,k), 5*(nx2), 5*ldx);
		}
	}

	/* one line (j,k) of the z sweep of compute_rhs: fluxes and fourth order dissipation */
	private void rhsZ(int j, int k){
		int i;
		double uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;
		for(i = 1; i<=nx2; i++){
			int p = ix(i,j,k);
			int c = 5*p;
			int cp = c + 5*sk;
			int cm = c - 5*sk;
			int pp = p + sk;
			int pm = p - sk;
			wijk = ws[p];
			wp1  = ws[pp];
			wm1  = ws[pm];
			rhs[c] = rhs[c] + dz1tz1 *
			(u[cp] - 2.0*u[c] +
					u[cm]) -
					tz2 * (u[cp+3] - u[cm+3]);
			rhs[c+1] = rhs[c+1] + dz2tz1 *
			(u[cp+1] - 2.0*u[c+1] +
					u[cm+1]) +
					zzcon2 * (us[pp] - 2.0*us[p] +
							us[pm]) -
							tz2 * (u[cp+1]*wp1 -
									u[cm+1]*wm1);
			rhs[c+2] = rhs[c+2] + dz3tz1 *
			(u[cp+2] - 2.0*u[c+2] +
					u[cm+2]) +
					zzcon2 * (vs[pp] - 2.0*vs[p] +
							vs[pm]) -
							tz2 * (u[cp+2]*wp1 -
									u[cm+2]*wm1);
			rhs[c+3] = rhs[c+3] + dz4tz1 *
			(u[cp+3] - 2.0*u[c+3] +
					u[cm+3]) +
					zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
					tz2 * (u[cp+3]*wp1 -
							u[cm+3]*wm1 +
							(u[cp+4] - square[pp] -
									u[cm+4] + square[pm])
									*c2);
			rhs[c+4] = rhs[c+4] + dz5tz1 *
			(u[cp+4] - 2.0*u[c+4] +
					u[cm+4]) +
					zzcon3 * (qs[pp] - 2.0*qs[p] +
							qs[pm]) +
							zzcon4 * (wp1*wp1 - 2.0*wijk*wijk +
									wm1*wm1) +
									zzcon5 * (u[cp+4]*rhoI[pp] -
											2.0*u[c+4]*rhoI[p] +
											u[cm+4]*rhoI[pm]) -
											tz2 * ( (c1*u[cp+4] -
													c2*square[pp])*wp1 -
													(c1*u[cm+4] -
															c2*square[pm])*wm1);
		}
		/* add fourth order zeta-direction dissipation */
		if(k == 1){
			for(i = 1; i<=gridPoints[0]-2; i++)
				dissipFirst(5*ix(i,j,k), 5*sk);
		}
		else if(k == 2){
			for(i = 1; i<=gridPoints[0]-2; i++)
				dissipSecond(5*ix(i,j,k), 5*sk);
		}
		else if(k == gridPoints[2]-3){
			for(i = 1; i<=gridPoints[0]-2; i++)
				dissipLast2(5*ix(i,j,k), 5*sk);
		}
		else if(k == gridPoints[2]-2){
			for(i = 1; i<=gridPoints[0]-2; i++)
				dissipLast(5*ix(i,j,k), 5*sk);
		}
		else{
			dissipInteriorLine(5*ix(1,j,k), 5*(gridPoints[0]-2), 5*sk);
		}
	}

	/* dissipInterior for the len contiguous elements that start at c (a whole line of points along x) */
	private void dissipInteriorLine(int c, int len, int s){
		final int s2 = 2*s;
		for(int q = c, qe = c+len; q<qe; q++){
			rhs[q] = rhs[q] - dssp *
			(  u[q-s2] - 4.0*u[q-s] +
					6.0*u[q] - 4.0*u[q+s] +
					u[q+s2] );
		}
	}

	/*
	 * This method block-diagonalizes the x-direction (block-diagonal
	 * matrix multiplication by T_x^-1)
	 */
	private void txinvr(){

		int i, j, k;
		double t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv;

		if (detailedTiming) timer.start(T_TXINVR);
		for(k = 1; k<=nz2; k++){
			for(j = 1; j<=ny2; j++){
				for(i = 1; i<=nx2; i++){
					int p = ix(i,j,k);
					int c = 5*p;

					ru1 = rhoI[p];
					uu = us[p];
					vv = vs[p];
					ww = ws[p];
					ac = speed[p];
					ac2inv = ac*ac;

					r1 = rhs[c];
					r2 = rhs[c+1];
					r3 = rhs[c+2];
					r4 = rhs[c+3];
					r5 = rhs[c+4];

					t1 = c2 / ac2inv * ( qs[p]*r1 - uu*r2  -
							vv*r3 - ww*r4 + r5 );
					t2 = bt * ru1 * ( uu * r1 - r2 );
					t3 = ( bt * ru1 * ac ) * t1;

					rhs[c]   = r1 - t1;
					rhs[c+1] = - ru1 * ( ww*r1 - r4 );
					rhs[c+2] =   ru1 * ( vv*r1 - r3 );
					rhs[c+3] = - t2 + t3;
					rhs[c+4] =   t2 + t3;
				}
			}
		}
		if (detailedTiming) timer.stop(T_TXINVR);
	}

	/*
	 * This method performs the solution of the approximate factorization
	 * step in the direction dir (0: x, 1: y, 2: z) for all five matrix
	 * components simultaneously. The Thomas algorithm is employed to solve
	 * the systems for the lines. Boundary conditions are non-periodic.
	 *
	 * The three routines x_solve, y_solve and z_solve of the original are
	 * the same code with the coordinates permuted, so it is written once:
	 * the line of the direction is walked with the stride sl of the scalar
	 * fields, starting at the index o0 (the coordinate of the direction is 0).
	 * The only difference in the arithmetic is the choice of the constants of
	 * rhov (rv1..rv4) and the direction constants dtt2, dtt1 and c2dtt1,
	 * which are passed as the original uses them (dx2/dx5/dxmax/dx1,
	 * dy3/dy5/dymax/dy1 and dz4/dz5/dzmax/dz1).
	 * After the solve the original calls ninvr (x), pinvr (y) and tzetar (z).
	 */
	private void lineSolve(int dir){

		final int n = gridPoints[dir];
		final int n2 = n - 2;
		final int sl;
		final double cvSrc[];
		final double dtt2, dtt1, c2dtt1, rv1, rv2, rv3, rv4;
		final int timerId;

		if(dir == 0){
			sl = 1;
			cvSrc = us;
			dtt2 = dttx2; dtt1 = dttx1; c2dtt1 = c2dttx1;
			rv1 = dx2; rv2 = dx5; rv3 = dxmax; rv4 = dx1;
			timerId = T_XSOLVE;
		}
		else if(dir == 1){
			sl = ldx;
			cvSrc = vs;
			dtt2 = dtty2; dtt1 = dtty1; c2dtt1 = c2dtty1;
			rv1 = dy3; rv2 = dy5; rv3 = dymax; rv4 = dy1;
			timerId = T_YSOLVE;
		}
		else{
			sl = sk;
			cvSrc = ws;
			dtt2 = dttz2; dtt1 = dttz1; c2dtt1 = c2dttz1;
			rv1 = dz4; rv2 = dz5; rv3 = dzmax; rv4 = dz1;
			timerId = T_ZSOLVE;
		}

		/* the two coordinates that are fixed in each line */
		final int a0, a1;	/* limits of the outer loops (1..a0, 1..a1) */
		if(dir == 0){ a0 = ny2; a1 = nz2; }		/* (j, k) */
		else if(dir == 1){ a0 = nx2; a1 = nz2; }	/* (i, k) */
		else{ a0 = nx2; a1 = ny2; }			/* (i, j) */

		final double[] lh = this.lh;

		if (detailedTiming) timer.start(timerId);

		for(int b = 1; b<=a1; b++){
			for(int a = 1; a<=a0; a++){

				final int o0;
				if(dir == 0)
					o0 = ix(0,a,b);
				else if(dir == 1)
					o0 = ix(a,0,b);
				else
					o0 = ix(a,b,0);

				int i;
				double ru1;

				lhsInit(n2+1);

				/* Computes the left hand side for the three factors.
				 * First fill the lhs for the u-eigenvalue */
				for(i = 0; i<=n-1; i++){
					ru1 = c3c4*rhoI[o0+sl*i];
					cv[i] = cvSrc[o0+sl*i];
					/* dmax1 of the original with comparisons: the four values are positive and not NaN, so it is
					 * the same value as Math.max, whose NaN and signed zero handling costs ~15 instructions each */
					final double m1 = rv1+con43*ru1, m2 = rv2+c1c5*ru1, m3 = rv3+ru1;
					final double m12 = m1 > m2 ? m1 : m2;
					final double m34 = m3 > rv4 ? m3 : rv4;
					rhov[i] = m12 > m34 ? m12 : m34;
				}

				for(i = 1; i<=n2; i++){
					lh[15*i] =  0.0;
					lh[15*i+1] = -dtt2 * cv[i-1] - dtt1 * rhov[i-1];
					lh[15*i+2] =  1.0 + c2dtt1 * rhov[i];
					lh[15*i+3] =  dtt2 * cv[i+1] - dtt1 * rhov[i+1];
					lh[15*i+4] =  0.0;
				}

				/* add fourth order dissipation */
				i = 1;
				lh[15*i+2] = lh[15*i+2] + comz5;
				lh[15*i+3] = lh[15*i+3] - comz4;
				lh[15*i+4] = lh[15*i+4] + comz1;

				lh[15*(i+1)+1] = lh[15*(i+1)+1] - comz4;
				lh[15*(i+1)+2] = lh[15*(i+1)+2] + comz6;
				lh[15*(i+1)+3] = lh[15*(i+1)+3] - comz4;
				lh[15*(i+1)+4] = lh[15*(i+1)+4] + comz1;

				for(i = 3; i<=n-4; i++){
					lh[15*i] = lh[15*i] + comz1;
					lh[15*i+1] = lh[15*i+1] - comz4;
					lh[15*i+2] = lh[15*i+2] + comz6;
					lh[15*i+3] = lh[15*i+3] - comz4;
					lh[15*i+4] = lh[15*i+4] + comz1;
				}

				i = n-3;
				lh[15*i] = lh[15*i] + comz1;
				lh[15*i+1] = lh[15*i+1] - comz4;
				lh[15*i+2] = lh[15*i+2] + comz6;
				lh[15*i+3] = lh[15*i+3] - comz4;

				lh[15*(i+1)] = lh[15*(i+1)] + comz1;
				lh[15*(i+1)+1] = lh[15*(i+1)+1] - comz4;
				lh[15*(i+1)+2] = lh[15*(i+1)+2] + comz5;

				/* subsequently, fill the other factors (u+c), (u-c): only the rows 1 and 3
				 * differ from those of lhs; the rows 0, 2 and 4 are read from lhs in the elimination */
				for(i = 1; i<=n2; i++){
					lh[15*i+6] = lh[15*i+1] -
					dtt2 * speed[o0+sl*(i-1)];
					lh[15*i+8] = lh[15*i+3] +
					dtt2 * speed[o0+sl*(i+1)];
					lh[15*i+11] = lh[15*i+1] +
					dtt2 * speed[o0+sl*(i-1)];
					lh[15*i+13] = lh[15*i+3] -
					dtt2 * speed[o0+sl*(i+1)];
				}

				/* perform the Thomas algorithm; first, FORWARD ELIMINATION. The three factors (lhs for the
				 * components 0..2, lhsp for 3 and lhsm for 4) are independent and go in the same loop. The rows i
				 * (d, c3, c4 and q of rhs) and i+1 (b1, b2, b3 and s of rhs) being eliminated travel in locals; row
				 * i+2 is read when it comes into play. Only what is read later is stored: lhs*[3], lhs*[4] (back
				 * substitution) and rhs. Same operations and order as the original for every element */
				int r = 5*o0;
				int r1 = 5*(o0+sl);
				double dL = lh[15*0+2], c3L = lh[15*0+3], c4L = lh[15*0+4], b1L = lh[15*1+1], b2L = lh[15*1+2], b3L = lh[15*1+3];
				double dP = lh[15*0+2], c3P = lh[15*0+8], c4P = lh[15*0+4], b1P = lh[15*1+6], b2P = lh[15*1+2], b3P = lh[15*1+8];
				double dM = lh[15*0+2], c3M = lh[15*0+13], c4M = lh[15*0+4], b1M = lh[15*1+11], b2M = lh[15*1+2], b3M = lh[15*1+13];
				double q0 = rhs[r], q1 = rhs[r+1], q2 = rhs[r+2], q3 = rhs[r+3], q4 = rhs[r+4];
				double s0 = rhs[r1], s1 = rhs[r1+1], s2 = rhs[r1+2], s3 = rhs[r1+3], s4 = rhs[r1+4];
				double fac1, fac2, e0;
				for(i = 0; i<=n-3; i++){
					final int r2 = r1 + 5*sl;
					/* lhs */
					fac1 = 1.0/dL;
					c3L = fac1*c3L;
					c4L = fac1*c4L;
					lh[15*i+3] = c3L;
					lh[15*i+4] = c4L;
					q0 = fac1*q0;
					rhs[r] = q0;
					q1 = fac1*q1;
					rhs[r+1] = q1;
					q2 = fac1*q2;
					rhs[r+2] = q2;
					b2L = b2L - b1L*c3L;
					b3L = b3L - b1L*c4L;
					s0 = s0 - b1L*q0;
					s1 = s1 - b1L*q1;
					s2 = s2 - b1L*q2;
					e0 = lh[15*(i+2)];
					final double e1L = lh[15*(i+2)+1] - e0*c3L;
					final double e2L = lh[15*(i+2)+2] - e0*c4L;
					final double t0 = rhs[r2] - e0*q0;
					final double t1 = rhs[r2+1] - e0*q1;
					final double t2 = rhs[r2+2] - e0*q2;
					dL = b2L; c3L = b3L; c4L = lh[15*(i+1)+4];
					b1L = e1L; b2L = e2L; b3L = lh[15*(i+2)+3];
					q0 = s0; s0 = t0;
					q1 = s1; s1 = t1;
					q2 = s2; s2 = t2;
					/* lhsp */
					fac1 = 1.0/dP;
					c3P = fac1*c3P;
					c4P = fac1*c4P;
					lh[15*i+8] = c3P;
					lh[15*i+9] = c4P;
					q3 = fac1*q3;
					rhs[r+3] = q3;
					b2P = b2P - b1P*c3P;
					b3P = b3P - b1P*c4P;
					s3 = s3 - b1P*q3;
					e0 = lh[15*(i+2)];
					final double e1P = lh[15*(i+2)+6] - e0*c3P;
					final double e2P = lh[15*(i+2)+2] - e0*c4P;
					final double t3 = rhs[r2+3] - e0*q3;
					dP = b2P; c3P = b3P; c4P = lh[15*(i+1)+4];
					b1P = e1P; b2P = e2P; b3P = lh[15*(i+2)+8];
					q3 = s3; s3 = t3;
					/* lhsm */
					fac1 = 1.0/dM;
					c3M = fac1*c3M;
					c4M = fac1*c4M;
					lh[15*i+13] = c3M;
					lh[15*i+14] = c4M;
					q4 = fac1*q4;
					rhs[r+4] = q4;
					b2M = b2M - b1M*c3M;
					b3M = b3M - b1M*c4M;
					s4 = s4 - b1M*q4;
					e0 = lh[15*(i+2)];
					final double e1M = lh[15*(i+2)+11] - e0*c3M;
					final double e2M = lh[15*(i+2)+2] - e0*c4M;
					final double t4 = rhs[r2+4] - e0*q4;
					dM = b2M; c3M = b3M; c4M = lh[15*(i+1)+4];
					b1M = e1M; b2M = e2M; b3M = lh[15*(i+2)+13];
					q4 = s4; s4 = t4;
					r = r1;
					r1 = r2;
				}
				
				/* The last two rows in this grid block are a bit different,
				 * since they do not have two more rows available for the
				 * elimination of off-diagonal entries (now i = n-2, r and r1 point to the rows n-2 and n-1) */
				fac1 = 1.0/dL;
				c3L = fac1*c3L;
				c4L = fac1*c4L;
				lh[15*i+3] = c3L;
				lh[15*i+4] = c4L;
				q0 = fac1*q0;
				rhs[r] = q0;
				q1 = fac1*q1;
				rhs[r+1] = q1;
				q2 = fac1*q2;
				rhs[r+2] = q2;
				b2L = b2L - b1L*c3L;
				b3L = b3L - b1L*c4L;
				s0 = s0 - b1L*q0;
				s1 = s1 - b1L*q1;
				s2 = s2 - b1L*q2;
				/* scale the last row immediately */
				fac2 = 1.0/b2L;
				s0 = fac2*s0;
				s1 = fac2*s1;
				s2 = fac2*s2;
				rhs[r1] = s0;
				rhs[r1+1] = s1;
				rhs[r1+2] = s2;
				fac1 = 1.0/dP;
				c3P = fac1*c3P;
				c4P = fac1*c4P;
				lh[15*i+8] = c3P;
				lh[15*i+9] = c4P;
				q3 = fac1*q3;
				rhs[r+3] = q3;
				b2P = b2P - b1P*c3P;
				b3P = b3P - b1P*c4P;
				s3 = s3 - b1P*q3;
				s3 = s3/b2P;
				rhs[r1+3] = s3;
				fac1 = 1.0/dM;
				c3M = fac1*c3M;
				c4M = fac1*c4M;
				lh[15*i+13] = c3M;
				lh[15*i+14] = c4M;
				q4 = fac1*q4;
				rhs[r+4] = q4;
				b2M = b2M - b1M*c3M;
				b3M = b3M - b1M*c4M;
				s4 = s4 - b1M*q4;
				s4 = s4/b2M;
				rhs[r1+4] = s4;
				
				/* BACKSUBSTITUTION: s is the solution of the row i+2 and q that of the row i+1 */
				q0 = q0 - c3L*s0;
				rhs[r] = q0;
				q1 = q1 - c3L*s1;
				rhs[r+1] = q1;
				q2 = q2 - c3L*s2;
				rhs[r+2] = q2;
				q3 = q3 - c3P*s3;
				rhs[r+3] = q3;
				q4 = q4 - c3M*s4;
				rhs[r+4] = q4;
				for(i = n-3; i>=0; i--){
					r -= 5*sl;
					final double v0 = rhs[r] - lh[15*i+3]*q0 - lh[15*i+4]*s0;
					rhs[r] = v0;
					s0 = q0; q0 = v0;
					final double v1 = rhs[r+1] - lh[15*i+3]*q1 - lh[15*i+4]*s1;
					rhs[r+1] = v1;
					s1 = q1; q1 = v1;
					final double v2 = rhs[r+2] - lh[15*i+3]*q2 - lh[15*i+4]*s2;
					rhs[r+2] = v2;
					s2 = q2; q2 = v2;
					final double v3 = rhs[r+3] - lh[15*i+8]*q3 - lh[15*i+9]*s3;
					rhs[r+3] = v3;
					s3 = q3; q3 = v3;
					final double v4 = rhs[r+4] - lh[15*i+13]*q4 - lh[15*i+14]*s4;
					rhs[r+4] = v4;
					s4 = q4; q4 = v4;
				}
			}
		}

		if (detailedTiming) timer.stop(timerId);

		/* the block-diagonal multiplication that follows each solve */
		if(dir == 0)
			ninvr();
		else if(dir == 1)
			pinvr();
		else
			tzetar();
	}

	/*
	 * The boundary entries of the work arrays of one line: row 0 and row ni
	 * are the identity (the original lhsinit)
	 */
	private void lhsInit(int ni){

		for(int m = 0; m<5; m++){
			lh[15*0+m] = 0.0;
			lh[15*0+5+m] = 0.0;
			lh[15*0+10+m] = 0.0;
			lh[15*ni+m] = 0.0;
			lh[15*ni+5+m] = 0.0;
			lh[15*ni+10+m] = 0.0;
		}

		lh[15*0+2] = 1.0;
		lh[15*0+7] = 1.0;
		lh[15*0+12] = 1.0;
		lh[15*ni+2] = 1.0;
		lh[15*ni+7] = 1.0;
		lh[15*ni+12] = 1.0;
	}

	/*
	 * This method block-diagonalizes the y-direction (block-diagonal
	 * matrix multiplication by N^-1), done after the x solve
	 */
	private void ninvr(){

		int i, j, k;
		double r1, r2, r3, r4, r5, t1, t2;

		if (detailedTiming) timer.start(T_NINVR);
		for(k = 1; k<=nz2; k++){
			for(j = 1; j<=ny2; j++){
				for(i = 1; i<=nx2; i++){
					int c = 5*ix(i,j,k);

					r1 = rhs[c];
					r2 = rhs[c+1];
					r3 = rhs[c+2];
					r4 = rhs[c+3];
					r5 = rhs[c+4];

					t1 = bt * r3;
					t2 = 0.5 * ( r4 + r5 );

					rhs[c]   = -r2;
					rhs[c+1] =  r1;
					rhs[c+2] = bt * ( r4 - r5 );
					rhs[c+3] = -t1 + t2;
					rhs[c+4] =  t1 + t2;
				}
			}
		}
		if (detailedTiming) timer.stop(T_NINVR);
	}

	/*
	 * This method block-diagonalizes the z-direction (block-diagonal
	 * matrix multiplication by P^-1), done after the y solve
	 */
	private void pinvr(){

		int i, j, k;
		double r1, r2, r3, r4, r5, t1, t2;

		if (detailedTiming) timer.start(T_PINVR);
		for(k = 1; k<=nz2; k++){
			for(j = 1; j<=ny2; j++){
				for(i = 1; i<=nx2; i++){
					int c = 5*ix(i,j,k);

					r1 = rhs[c];
					r2 = rhs[c+1];
					r3 = rhs[c+2];
					r4 = rhs[c+3];
					r5 = rhs[c+4];

					t1 = bt * r1;
					t2 = 0.5 * ( r4 + r5 );

					rhs[c]   =  bt * ( r4 - r5 );
					rhs[c+1] = -r3;
					rhs[c+2] =  r2;
					rhs[c+3] = -t1 + t2;
					rhs[c+4] =  t1 + t2;
				}
			}
		}
		if (detailedTiming) timer.stop(T_PINVR);
	}

	/*
	 * This method performs the block-diagonal inversion T_z^-1 after the
	 * z solve (the original tzetar)
	 */
	private void tzetar(){

		int i, j, k;
		double t1, t2, t3, ac, xvel, yvel, zvel, r1, r2, r3, r4, r5, btuz, ac2u, uzik1;

		if (detailedTiming) timer.start(T_TZETAR);
		for(k = 1; k<=nz2; k++){
			for(j = 1; j<=ny2; j++){
				for(i = 1; i<=nx2; i++){
					int p = ix(i,j,k);
					int c = 5*p;

					xvel = us[p];
					yvel = vs[p];
					zvel = ws[p];
					ac   = speed[p];

					ac2u = ac*ac;

					r1 = rhs[c];
					r2 = rhs[c+1];
					r3 = rhs[c+2];
					r4 = rhs[c+3];
					r5 = rhs[c+4];

					uzik1 = u[c];
					btuz  = bt * uzik1;

					t1 = btuz/ac * (r4 + r5);
					t2 = r3 + t1;
					t3 = btuz * (r4 - r5);

					rhs[c]   = t2;
					rhs[c+1] = -uzik1*r2 + xvel*t2;
					rhs[c+2] =  uzik1*r1 + yvel*t2;
					rhs[c+3] =  zvel*t2  + t3;
					rhs[c+4] =  uzik1*(-xvel*r2 + yvel*r1) +
					qs[p]*t2 + c2iv*ac2u*t1 + zvel*t3;
				}
			}
		}
		if (detailedTiming) timer.stop(T_TZETAR);
	}

	/*
	 * This method adds the update to the solution
	 */
	private void add(){

		int i, j, k, m;

		if (detailedTiming) timer.start(T_ADD);
		for(k = 1; k<=nz2; k++){
			for(j = 1; j<=ny2; j++){
				for(i = 1; i<=nx2; i++){
					int c = 5*ix(i,j,k);
					for(m = 0; m<5; m++){
						u[c+m] = u[c+m] + rhs[c+m];
					}
				}
			}
		}
		if (detailedTiming) timer.stop(T_ADD);
	}

	private double dmax1(double a, double b){
		return Math.max(a, b);
	}

	private double dmax1(double a, double b, double c, double d){
		return Math.max(Math.max(a, b), Math.max(c, d));
	}

	/*
	 * This method verifies the solution: the RMS norms of the residual and of
	 * the solution error are compared against the reference values of the class
	 */
	private int verify(int noTimeSteps, char CLASS){

		int verified = 0;
		double xce[] = new double[5];
		double xcr[] = new double[5];
		double xceref[] = new double[5];
		double xcrref[] = new double[5];
		double xcedif[] = new double[5];
		double xcrdif[] = new double[5];
		double dtref = 0;

		/* Tolerance level */
		double epsilon = 1.0E-8;

		/* Compute the error norm and the residual norm, and exit if not printing */
		errorNorm(xce);

		computeRHS();

		rhsNorm(xcr);

		for(int m = 0; m<5; m++){
			xcr[m] = xcr[m]/dt;
		}

		verified = 1;

		for(int m = 0; m<5; m++){
			xcrref[m] = 1.0;
			xceref[m] = 1.0;
		}

		/* Reference values of RMS-norms of residual and of solution error */
		if(CLASS == 'S'){
			dtref = 1.5e-2;
			xcrref[0] = 2.7470315451339479e-02;
			xcrref[1] = 1.0360746705285417e-02;
			xcrref[2] = 1.6235745065095532e-02;
			xcrref[3] = 1.5840557224455615e-02;
			xcrref[4] = 3.4849040609362460e-02;
			xceref[0] = 2.7289258557377227e-05;
			xceref[1] = 1.0364446640837285e-05;
			xceref[2] = 1.6154798287166471e-05;
			xceref[3] = 1.5750704994480102e-05;
			xceref[4] = 3.4177666183390531e-05;
		}
		else if(CLASS == 'W'){
			dtref = 1.5e-3;
			xcrref[0] = 0.1893253733584e-02;
			xcrref[1] = 0.1717075447775e-03;
			xcrref[2] = 0.2778153350936e-03;
			xcrref[3] = 0.2887475409984e-03;
			xcrref[4] = 0.3143611161242e-02;
			xceref[0] = 0.7542088599534e-04;
			xceref[1] = 0.6512852253086e-05;
			xceref[2] = 0.1049092285688e-04;
			xceref[3] = 0.1128838671535e-04;
			xceref[4] = 0.1212845639773e-03;
		}
		else if(CLASS == 'A'){
			dtref = 1.5e-3;
			xcrref[0] = 2.4799822399300195e0;
			xcrref[1] = 1.1276337964368832e0;
			xcrref[2] = 1.5028977888770491e0;
			xcrref[3] = 1.4217816211695179e0;
			xcrref[4] = 2.1292113035138280e0;
			xceref[0] = 1.0900140297820550e-04;
			xceref[1] = 3.7343951769282091e-05;
			xceref[2] = 5.0092785406541633e-05;
			xceref[3] = 4.7671093939528255e-05;
			xceref[4] = 1.3621613399213001e-04;
		}
		else if(CLASS == 'B'){
			dtref = 1.0e-3;
			xcrref[0] = 0.6903293579998e+02;
			xcrref[1] = 0.3095134488084e+02;
			xcrref[2] = 0.4103336647017e+02;
			xcrref[3] = 0.3864769009604e+02;
			xcrref[4] = 0.5643482272596e+02;
			xceref[0] = 0.9810006190188e-02;
			xceref[1] = 0.1022827905670e-02;
			xceref[2] = 0.1720597911692e-02;
			xceref[3] = 0.1694479428231e-02;
			xceref[4] = 0.1847456263981e-01;
		}
		else if(CLASS == 'C'){
			dtref = 0.67e-3;
			xcrref[0] = 0.5881691581829e+03;
			xcrref[1] = 0.2454417603569e+03;
			xcrref[2] = 0.3293829191851e+03;
			xcrref[3] = 0.3081924971891e+03;
			xcrref[4] = 0.4597223799176e+03;
			xceref[0] = 0.2598120500183e+00;
			xceref[1] = 0.2590888922315e-01;
			xceref[2] = 0.5132886416320e-01;
			xceref[3] = 0.4806073419454e-01;
			xceref[4] = 0.5483377491301e+00;
		}
		else if(CLASS == 'D'){
			dtref = 0.30e-3;
			xcrref[0] = 0.1044696216887e+05;
			xcrref[1] = 0.3204427762578e+04;
			xcrref[2] = 0.4648680733032e+04;
			xcrref[3] = 0.4238923283697e+04;
			xcrref[4] = 0.7588412036136e+04;
			xceref[0] = 0.5089471423669e+01;
			xceref[1] = 0.5323514855894e+00;
			xceref[2] = 0.1187051008971e+01;
			xceref[3] = 0.1083734951938e+01;
			xceref[4] = 0.1164108338568e+02;
		}
		else{
			CLASS = 'U';
			verified = 0;
		}

		/* Compute the difference of solution values and the known reference values */
		for(int m = 0; m<5; m++){
			xcrdif[m] = Math.abs((xcr[m]-xcrref[m])/xcrref[m]);
			xcedif[m] = Math.abs((xce[m]-xceref[m])/xceref[m]);
		}

		/* Output the comparison of computed results to known cases */
		if(CLASS != 'U'){
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
				System.out.println("\t"+(m+1)+" "+xcr[m]);
			}
			else if(xcrdif[m] <= epsilon){
				System.out.println("\t"+(m+1)+" "+xcr[m]+" "+xcrref[m]+" "+xcrdif[m]);
			}
			else{
				verified = 0;
				System.out.println("FAILURE: \t"+(m+1)+" "+xcr[m]+" "+xcrref[m]+" "+xcrdif[m]);
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
				System.out.println("\t"+(m+1)+" "+xce[m]);
			}
			else if(xcedif[m] <= epsilon){
				System.out.println("\t"+(m+1)+" "+xce[m]+" "+xceref[m]+" "+xcedif[m]);
			}
			else{
				verified = 0;
				System.out.println("FAILURE: \t"+(m+1)+" "+xce[m]+" "+xceref[m]+" "+xcedif[m]);
			}
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

	/*
	 * This method computes the norm of the difference between the
	 * computed solution and the exact solution
	 */
	private void errorNorm(double rms[]){

		int i, j, k, m, d;
		double xi, eta, zeta, add;
		double uExact[] = new double[5];

		for(m = 0; m<5; m++){
			rms[m] = 0.0;
		}

		for(k = 0; k<=gridPoints[2]-1; k++){
			zeta = (double)k * dnzm1;
			for(j = 0; j<=gridPoints[1]-1; j++){
				eta = (double)j * dnym1;
				for(i = 0; i<=gridPoints[0]-1; i++){
					xi = (double)i * dnxm1;
					exactSolution(xi, eta, zeta, uExact);

					int c = 5*ix(i,j,k);
					for(m = 0; m<5; m++){
						add = u[c+m]-uExact[m];
						rms[m] = rms[m] + add*add;
					}
				}
			}
		}

		for(m = 0; m<5; m++){
			for(d = 0; d<3; d++){
				rms[m] = rms[m] / (double)(gridPoints[d]-2);
			}
			rms[m] = Math.sqrt(rms[m]);
		}
	}

	/*
	 * This method computes the norm of the right hand side
	 */
	private void rhsNorm(double rms[]){

		int i, j, k, d, m;
		double add;

		for(m = 0; m<5; m++){
			rms[m] = 0.0;
		}

		for(k = 1; k<=nz2; k++){
			for(j = 1; j<=ny2; j++){
				for(i = 1; i<=nx2; i++){
					int c = 5*ix(i,j,k);
					for(m = 0; m<5; m++){
						add = rhs[c+m];
						rms[m] = rms[m] + add*add;
					}
				}
			}
		}

		for(m = 0; m<5; m++){
			for(d = 0; d<3; d++){
				rms[m] = rms[m] / (double)(gridPoints[d]-2);
			}
			rms[m] = Math.sqrt(rms[m]);
		}
	}

	private void printAdditionalTimers()
	{
		double tmax = timer.read(T_TOTAL);
		double t;

		if(tmax == 0.0)
			tmax = 1.0;

		System.out.format(java.util.Locale.ENGLISH, "  SECTION   Time (secs)\n");

		String names[] = {"total", "rhsx", "rhsy", "rhsz", "rhs", "xsolve", "ysolve", "zsolve",
				"", "", "txinvr", "pinvr", "ninvr", "tzetar", "add"};
		for(int i = 1; i<=T_LAST; i++){
			if(names[i-1].length() == 0)
				continue;
			t = timer.read(i);
			System.out.format(java.util.Locale.ENGLISH, "  %-8s:%9.3f  (%6.2f%%)\n", names[i-1], t, t*100./tmax);
			if(i == T_RHS){
				t = timer.read(T_RHSX) + timer.read(T_RHSY) + timer.read(T_RHSZ);
				System.out.format(java.util.Locale.ENGLISH, "    --> %-8s:%9.3f  (%6.2f%%)\n", "sub-rhs", t, t*100./tmax);
				t = timer.read(T_RHS) - t;
				System.out.format(java.util.Locale.ENGLISH, "    --> %-8s:%9.3f  (%6.2f%%)\n", "rest-rhs", t, t*100./tmax);
			}
		}
	}
}
