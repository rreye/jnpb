/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB 3.4.4 OpenMP (NPB3.4-OMP/BT)
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

public class BT implements Benchmark{

	private static final String BM_NAME="BT";
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
	private double square[];

	/*
	 * Work arrays of one grid line. fjac and njac are the original
	 * fjac(5,5,0:N) and njac(5,5,0:N) and lhs is lhs(5,5,3,0:N) (blocks
	 * aa, bb, cc), all with the first index running fastest: element (m,n)
	 * of the block t is [25*t+m+5*n] (and [75*t+25*b+m+5*n] in lhs).
	 */
	private double fjac[];
	private double njac[];
	private double lhs[];

	/* work arrays of exact_rhs: cuf(0:N), q(0:N), ue(0:N,5) and buf(0:N,5) */
	private double cuf[];
	private double q[];
	private double ue[][];
	private double buf[][];

	private double ce[][] = new double[5][13];

	private double c1,c2,c3,c4,c5,c1c2,c1c5,c3c4,c1345;
	private double dnxm1,dnym1,dnzm1,conz1;
	private double tx1,tx2,tx3,ty1,ty2,ty3,tz1,tz2,tz3;
	private double dx1,dx2,dx3,dx4,dx5,dy1,dy2,dy3,dy4,dy5,dz1,dz2,dz3,dz4,dz5;
	private double dssp;
	private double dx1tx1,dx2tx1,dx3tx1,dx4tx1,dx5tx1;
	private double dy1ty1,dy2ty1,dy3ty1,dy4ty1,dy5ty1;
	private double dz1tz1,dz2tz1,dz3tz1,dz4tz1,dz5tz1;
	private double c3c4tx3,c3c4ty3,c3c4tz3;
	private double con43, con16;
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
	private final int T_RDIS1 = 9;
	private final int T_SOLSUB = 10;
	private final int T_ADD = 11;
	private final int T_LAST = 11;

	public static void main(String[] args) throws NPBException {
		BT bt = null;

		BMArgs.parseCmdLineArgs(args,BM_NAME);
		try{
			bt = new BT(BMArgs.CLASS,BMArgs.DETAILED_TIMING);
		}catch(OutOfMemoryError e){
			BMArgs.outOfMemoryMessage();
			throw e;
		}
		bt.runBenchmark(args);
	}

	public BMResults getResults(){
		return res;
	}

	public BT(char classSize, boolean dtiming){

		CLASS = classSize;
		detailedTiming = dtiming;
	}

	public void build(String[] args) throws NPBException{

		switch( CLASS )
		{
		case 'S':
			problemSize = 12;
			niter = 60;
			dt = 0.010;
			break;
		case 'W':
			problemSize = 24;
			niter = 200;
			dt = 0.0008;
			break;
		case 'A':
			problemSize = 64;
			niter = 200;
			dt = 0.0008;
			break;
		case 'B':
			problemSize = 102;
			niter = 200;
			dt = 0.0003;
			break;
		case 'C':
			problemSize = 162;
			niter = 200;
			dt = 0.0001;
			break;
		case 'D':
			problemSize = 408;
			niter = 250;
			dt = 0.00002;
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

		initialize();

		exactRHS();

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
			mflops = 1.0E-6*(double)niter*(3478.8*n3
					-17655.7*(t*t)
					+28023.7*t)/times[0];

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
	 * Reads the optional input file inputbt.data (niter, dt and grid
	 * size, as the original) or takes the compiled defaults of the class
	 */
	private void readData(){

		boolean inputData = true;

		BMArgs.bannerSER(BM_NAME, CLASS);

		try{

			BufferedReader input = new BufferedReader(new FileReader("inputbt.data"));

			System.out.println(" Reading from input file inputbt.data");

			String line[] = input.readLine().trim().split("\\s+");
			niter = Integer.valueOf(line[0]);

			line = input.readLine().trim().split("\\s+");
			/* the sample file writes dt in Fortran form (0.01d0) */
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
			System.out.println(" No input file inputbt.data. Using compiled defaults");
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
		square = new double[n];

		int nmax = problemSize + 1;
		fjac = new double[25*nmax];
		njac = new double[25*nmax];
		lhs = new double[75*nmax];
		cuf = new double[nmax];
		q = new double[nmax];
		ue = new double[5][nmax];
		buf = new double[5][nmax];
	}

	private void setConstants(){

		ce[0][0]  = 2.0;
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

		dnxm1 = 1.0 / (double)(gridPoints[0]-1);
		dnym1 = 1.0 / (double)(gridPoints[1]-1);
		dnzm1 = 1.0 / (double)(gridPoints[2]-1);

		c1c2 = c1 * c2;
		c1c5 = c1 * c5;
		c3c4 = c3 * c4;
		c1345 = c1c5 * c3c4;

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

		dssp = 0.25 * dmax1(dx1, dmax1(dy1, dz1) );

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

	private static double dmax1(double a, double b){
		return (a >= b) ? a : b;
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
					for(m = 0; m<5; m++)
						u[c+m] = 1.0;
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
	 * It is a literal translation: the three sweeps (x, y and z) differ in
	 * the order of the operands, so they are kept separate.
	 */
	private void exactRHS(){

		double dtemp[] = new double[5];
		double xi, eta, zeta, dtpp;
		int m, i, j, k, ip1, im1, jp1, jm1, km1, kp1;

		for (k = 0; k <= gridPoints[2]-1; k++){
			for (j = 0; j <= gridPoints[1]-1; j++){
				for (i = 0; i <= gridPoints[0]-1; i++){
					for (m = 0; m <= 4; m++){
						forcing[5*ix(i,j,k)+m] = 0.0;
					}
				}
			}
		}
		for (k = 1; k <= gridPoints[2]-2; k++){
			for (j = 1; j <= gridPoints[1]-2; j++){
				zeta = (double)(k) * dnzm1;
				eta = (double)(j) * dnym1;
				for (i = 0; i <= gridPoints[0]-1; i++){
					xi = (double)(i) * dnxm1;
					exactSolution(xi, eta, zeta, dtemp);
					for (m = 0; m <= 4; m++){
						ue[m][i] = dtemp[m];
					}
					dtpp = 1.0 / dtemp[0];
					for (m = 1; m <= 4; m++){
						buf[m][i] = dtpp * dtemp[m];
					}
					cuf[i] = buf[1][i] * buf[1][i];
					buf[0][i] = cuf[i] + buf[2][i] * buf[2][i] + buf[3][i] * buf[3][i];
					q[i] = 0.5*(buf[1][i]*ue[1][i] + buf[2][i]*ue[2][i] + buf[3][i]*ue[3][i]);
				}
				for (i = 1; i <= gridPoints[0]-2; i++){
					im1 = i-1;
					ip1 = i+1;
					forcing[5*ix(i,j,k)+0] = forcing[5*ix(i,j,k)+0] - tx2*( ue[1][ip1]-ue[1][im1] )+ dx1tx1*(ue[0][ip1]-2.0*ue[0][i]+ue[0][im1]);
					forcing[5*ix(i,j,k)+1] = forcing[5*ix(i,j,k)+1] - tx2 * ( (ue[1][ip1]*buf[1][ip1]+c2*(ue[4][ip1]-q[ip1]))- (ue[1][im1]*buf[1][im1]+c2*(ue[4][im1]-q[im1])))+ xxcon1*(buf[1][ip1]-2.0*buf[1][i]+buf[1][im1])+ dx2tx1*( ue[1][ip1]-2.0* ue[1][i]+ue[1][im1]);
					forcing[5*ix(i,j,k)+2] = forcing[5*ix(i,j,k)+2] - tx2 * ( ue[2][ip1]*buf[1][ip1]-ue[2][im1]*buf[1][im1])+ xxcon2*(buf[2][ip1]-2.0*buf[2][i]+buf[2][im1])+ dx3tx1*( ue[2][ip1]-2.0*ue[2][i] +ue[2][im1]);
					forcing[5*ix(i,j,k)+3] = forcing[5*ix(i,j,k)+3] - tx2*( ue[3][ip1]*buf[1][ip1]-ue[3][im1]*buf[1][im1])+ xxcon2*(buf[3][ip1]-2.0*buf[3][i]+buf[3][im1])+ dx4tx1*( ue[3][ip1]-2.0* ue[3][i]+ ue[3][im1]);
					forcing[5*ix(i,j,k)+4] = forcing[5*ix(i,j,k)+4] - tx2*( buf[1][ip1]*(c1*ue[4][ip1]-c2*q[ip1])- buf[1][im1]*(c1*ue[4][im1]-c2*q[im1]))+ 0.5*xxcon3*(buf[0][ip1]-2.0*buf[0][i]+ buf[0][im1])+ xxcon4*(cuf[ip1]-2.0*cuf[i]+cuf[im1])+ xxcon5*(buf[4][ip1]-2.0*buf[4][i]+buf[4][im1])+ dx5tx1*( ue[4][ip1]-2.0* ue[4][i]+ ue[4][im1]);
				}
				for (m = 0; m <= 4; m++){
					i = 1;
					forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp * (5.0*ue[m][i] - 4.0*ue[m][i+1] +ue[m][i+2]);
					i = 2;
					forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp * (-4.0*ue[m][i-1] + 6.0*ue[m][i] - 4.0*ue[m][i+1] + ue[m][i+2]);
				}
				for (i = 3; i <= gridPoints[0]-4; i++){
					for (m = 0; m <= 4; m++){
						forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp* (ue[m][i-2] - 4.0*ue[m][i-1] + 6.0*ue[m][i] - 4.0*ue[m][i+1] + ue[m][i+2]);
					}
				}
				for (m = 0; m <= 4; m++){
					i = gridPoints[0]-3;
					forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp * (ue[m][i-2] - 4.0*ue[m][i-1] + 6.0*ue[m][i] - 4.0*ue[m][i+1]);
					i = gridPoints[0]-2;
					forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp * (ue[m][i-2] - 4.0*ue[m][i-1] + 5.0*ue[m][i]);
				}
			}
		}
		for (k = 1; k <= gridPoints[2]-2; k++){
			for (i = 1; i <= gridPoints[0]-2; i++){
				zeta = (double)(k) * dnzm1;
				xi = (double)(i) * dnxm1;
				for (j = 0; j <= gridPoints[1]-1; j++){
					eta = (double)(j) * dnym1;
					exactSolution(xi, eta, zeta, dtemp);
					for (m = 0; m <= 4; m++){
						ue[m][j] = dtemp[m];
					}
					dtpp = 1.0/dtemp[0];
					for (m = 1; m <= 4; m++){
						buf[m][j] = dtpp * dtemp[m];
					}
					cuf[j] = buf[2][j] * buf[2][j];
					buf[0][j] = cuf[j] + buf[1][j] * buf[1][j] + buf[3][j] * buf[3][j];
					q[j] = 0.5*(buf[1][j]*ue[1][j] + buf[2][j]*ue[2][j] + buf[3][j]*ue[3][j]);
				}
				for (j = 1; j <= gridPoints[1]-2; j++){
					jm1 = j-1;
					jp1 = j+1;
					forcing[5*ix(i,j,k)+0] = forcing[5*ix(i,j,k)+0] - ty2*( ue[2][jp1]-ue[2][jm1] )+ dy1ty1*(ue[0][jp1]-2.0*ue[0][j]+ue[0][jm1]);
					forcing[5*ix(i,j,k)+1] = forcing[5*ix(i,j,k)+1] - ty2*( ue[1][jp1]*buf[2][jp1]-ue[1][jm1]*buf[2][jm1])+ yycon2*(buf[1][jp1]-2.0*buf[1][j]+buf[1][jm1])+ dy2ty1*( ue[1][jp1]-2.0* ue[1][j]+ ue[1][jm1]);
					forcing[5*ix(i,j,k)+2] = forcing[5*ix(i,j,k)+2] - ty2*( (ue[2][jp1]*buf[2][jp1]+c2*(ue[4][jp1]-q[jp1]))- (ue[2][jm1]*buf[2][jm1]+c2*(ue[4][jm1]-q[jm1])))+ yycon1*(buf[2][jp1]-2.0*buf[2][j]+buf[2][jm1])+ dy3ty1*( ue[2][jp1]-2.0*ue[2][j] +ue[2][jm1]);
					forcing[5*ix(i,j,k)+3] = forcing[5*ix(i,j,k)+3] - ty2*( ue[3][jp1]*buf[2][jp1]-ue[3][jm1]*buf[2][jm1])+ yycon2*(buf[3][jp1]-2.0*buf[3][j]+buf[3][jm1])+ dy4ty1*( ue[3][jp1]-2.0*ue[3][j]+ ue[3][jm1]);
					forcing[5*ix(i,j,k)+4] = forcing[5*ix(i,j,k)+4] - ty2*( buf[2][jp1]*(c1*ue[4][jp1]-c2*q[jp1])- buf[2][jm1]*(c1*ue[4][jm1]-c2*q[jm1]))+ 0.5*yycon3*(buf[0][jp1]-2.0*buf[0][j]+ buf[0][jm1])+ yycon4*(cuf[jp1]-2.0*cuf[j]+cuf[jm1])+ yycon5*(buf[4][jp1]-2.0*buf[4][j]+buf[4][jm1])+ dy5ty1*(ue[4][jp1]-2.0*ue[4][j]+ue[4][jm1]);
				}
				for (m = 0; m <= 4; m++){
					j = 1;
					forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp * (5.0*ue[m][j] - 4.0*ue[m][j+1] +ue[m][j+2]);
					j = 2;
					forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp * (-4.0*ue[m][j-1] + 6.0*ue[m][j] - 4.0*ue[m][j+1] + ue[m][j+2]);
				}
				for (j = 3; j <= gridPoints[1]-4; j++){
					for (m = 0; m <= 4; m++){
						forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp* (ue[m][j-2] - 4.0*ue[m][j-1] + 6.0*ue[m][j] - 4.0*ue[m][j+1] + ue[m][j+2]);
					}
				}
				for (m = 0; m <= 4; m++){
					j = gridPoints[1]-3;
					forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp * (ue[m][j-2] - 4.0*ue[m][j-1] + 6.0*ue[m][j] - 4.0*ue[m][j+1]);
					j = gridPoints[1]-2;
					forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp * (ue[m][j-2] - 4.0*ue[m][j-1] + 5.0*ue[m][j]);
				}
			}
		}
		for (j = 1; j <= gridPoints[1]-2; j++){
			for (i = 1; i <= gridPoints[0]-2; i++){
				eta = (double)(j) * dnym1;
				xi = (double)(i) * dnxm1;
				for (k = 0; k <= gridPoints[2]-1; k++){
					zeta = (double)(k) * dnzm1;
					exactSolution(xi, eta, zeta, dtemp);
					for (m = 0; m <= 4; m++){
						ue[m][k] = dtemp[m];
					}
					dtpp = 1.0/dtemp[0];
					for (m = 1; m <= 4; m++){
						buf[m][k] = dtpp * dtemp[m];
					}
					cuf[k] = buf[3][k] * buf[3][k];
					buf[0][k] = cuf[k] + buf[1][k] * buf[1][k] + buf[2][k] * buf[2][k];
					q[k] = 0.5*(buf[1][k]*ue[1][k] + buf[2][k]*ue[2][k] + buf[3][k]*ue[3][k]);
				}
				for (k = 1; k <= gridPoints[2]-2; k++){
					km1 = k-1;
					kp1 = k+1;
					forcing[5*ix(i,j,k)+0] = forcing[5*ix(i,j,k)+0] - tz2*( ue[3][kp1]-ue[3][km1] )+ dz1tz1*(ue[0][kp1]-2.0*ue[0][k]+ue[0][km1]);
					forcing[5*ix(i,j,k)+1] = forcing[5*ix(i,j,k)+1] - tz2 * ( ue[1][kp1]*buf[3][kp1]-ue[1][km1]*buf[3][km1])+ zzcon2*(buf[1][kp1]-2.0*buf[1][k]+buf[1][km1])+ dz2tz1*( ue[1][kp1]-2.0* ue[1][k]+ ue[1][km1]);
					forcing[5*ix(i,j,k)+2] = forcing[5*ix(i,j,k)+2] - tz2 * ( ue[2][kp1]*buf[3][kp1]-ue[2][km1]*buf[3][km1])+ zzcon2*(buf[2][kp1]-2.0*buf[2][k]+buf[2][km1])+ dz3tz1*(ue[2][kp1]-2.0*ue[2][k]+ue[2][km1]);
					forcing[5*ix(i,j,k)+3] = forcing[5*ix(i,j,k)+3] - tz2 * ( (ue[3][kp1]*buf[3][kp1]+c2*(ue[4][kp1]-q[kp1]))- (ue[3][km1]*buf[3][km1]+c2*(ue[4][km1]-q[km1])))+ zzcon1*(buf[3][kp1]-2.0*buf[3][k]+buf[3][km1])+ dz4tz1*( ue[3][kp1]-2.0*ue[3][k] +ue[3][km1]);
					forcing[5*ix(i,j,k)+4] = forcing[5*ix(i,j,k)+4] - tz2 * ( buf[3][kp1]*(c1*ue[4][kp1]-c2*q[kp1])- buf[3][km1]*(c1*ue[4][km1]-c2*q[km1]))+ 0.5*zzcon3*(buf[0][kp1]-2.0*buf[0][k] +buf[0][km1])+ zzcon4*(cuf[kp1]-2.0*cuf[k]+cuf[km1])+ zzcon5*(buf[4][kp1]-2.0*buf[4][k]+buf[4][km1])+ dz5tz1*( ue[4][kp1]-2.0*ue[4][k]+ ue[4][km1]);
				}
				for (m = 0; m <= 4; m++){
					k = 1;
					forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp * (5.0*ue[m][k] - 4.0*ue[m][k+1] +ue[m][k+2]);
					k = 2;
					forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp * (-4.0*ue[m][k-1] + 6.0*ue[m][k] - 4.0*ue[m][k+1] + ue[m][k+2]);
				}
				for (k = 3; k <= gridPoints[2]-4; k++){
					for (m = 0; m <= 4; m++){
						forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp* (ue[m][k-2] - 4.0*ue[m][k-1] + 6.0*ue[m][k] - 4.0*ue[m][k+1] + ue[m][k+2]);
					}
				}
				for (m = 0; m <= 4; m++){
					k = gridPoints[2]-3;
					forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp * (ue[m][k-2] - 4.0*ue[m][k-1] + 6.0*ue[m][k] - 4.0*ue[m][k+1]);
					k = gridPoints[2]-2;
					forcing[5*ix(i,j,k)+m] = forcing[5*ix(i,j,k)+m] - dssp * (ue[m][k-2] - 4.0*ue[m][k-1] + 5.0*ue[m][k]);
				}
			}
		}
		for (k = 1; k <= gridPoints[2]-2; k++){
			for (j = 1; j <= gridPoints[1]-2; j++){
				for (i = 1; i <= gridPoints[0]-2; i++){
					for (m = 0; m <= 4; m++){
						forcing[5*ix(i,j,k)+m] = -1.0 * forcing[5*ix(i,j,k)+m];
					}
				}
			}
		}
	}

	/*
	 * Adds the update to the vector u
	 */
	private void add(){

		if (detailedTiming) timer.start(T_ADD);
		for(int k = 1; k<=nz2; k++){
			for(int j = 1; j<=ny2; j++){
				for(int i = 1; i<=nx2; i++){
					int c = 5*ix(i,j,k);
					for(int m = 0; m<5; m++){
						u[c+m] = u[c+m] + rhs[c+m];
					}
				}
			}
		}
		if (detailedTiming) timer.stop(T_ADD);
	}

	private void adi(){

		computeRHS();

		lineSolve(0);

		lineSolve(1);

		lineSolve(2);

		add();
	}

	/*
	 * Fourth-order dissipation of the direction dir (0 = x, 1 = y, 2 = z):
	 * the original does it at the end of each sweep of compute_rhs, with the
	 * five boundary-dependent forms (points 1, 2, 3..n-4, n-3 and n-2 of
	 * the line). Every point is independent of the others, so it is
	 * applied here to the whole grid, once the terms of that direction
	 * have been added.
	 */
	private void dissipation(int dir){

		int n = gridPoints[dir];
		int d = 5*(dir == 0 ? 1 : dir == 1 ? ldx : sk);

		for(int k = 1; k<=nz2; k++){
			for(int j = 1; j<=ny2; j++){
				for(int i = 1; i<=nx2; i++){
					int t = dir == 0 ? i : dir == 1 ? j : k;
					int p = 5*ix(i,j,k);
					int pp = p + d, pm = p - d, pp2 = p + 2*d, pm2 = p - 2*d;

					if(t == 1){
						for(int m = 0; m<5; m++){
							rhs[p+m] = rhs[p+m] - dssp *
									( 5.0*u[p+m] - 4.0*u[pp+m] + u[pp2+m]);
						}
					}
					else if(t == 2){
						for(int m = 0; m<5; m++){
							rhs[p+m] = rhs[p+m] - dssp *
									(-4.0*u[pm+m] + 6.0*u[p+m] - 4.0*u[pp+m] + u[pp2+m]);
						}
					}
					else if(t == n-3){
						for(int m = 0; m<5; m++){
							rhs[p+m] = rhs[p+m] - dssp *
									( u[pm2+m] - 4.0*u[pm+m] + 6.0*u[p+m] - 4.0*u[pp+m] );
						}
					}
					else if(t == n-2){
						for(int m = 0; m<5; m++){
							rhs[p+m] = rhs[p+m] - dssp *
									( u[pm2+m] - 4.0*u[pm+m] + 5.0*u[p+m] );
						}
					}
					else{
						for(int m = 0; m<5; m++){
							rhs[p+m] = rhs[p+m] - dssp *
									( u[pm2+m] - 4.0*u[pm+m] + 6.0*u[p+m] - 4.0*u[pp+m] + u[pp2+m] );
						}
					}
				}
			}
		}
	}

	private void computeRHS(){

		if (detailedTiming) timer.start(T_RHS);

		/* compute the reciprocal of density, and the kinetic energy,
		 * and the speed of sound */
		for(int k = 0; k<gridPoints[2]; k++){
			for(int j = 0; j<gridPoints[1]; j++){
				for(int i = 0; i<gridPoints[0]; i++){
					int s = ix(i,j,k);
					int p = 5*s;
					double rhoInv = 1.0/u[p];
					rhoI[s] = rhoInv;
					us[s] = u[p+1] * rhoInv;
					vs[s] = u[p+2] * rhoInv;
					ws[s] = u[p+3] * rhoInv;
					square[s] = 0.5 * (
							u[p+1]*u[p+1] +
							u[p+2]*u[p+2] +
							u[p+3]*u[p+3] ) * rhoInv;
					qs[s] = square[s] * rhoInv;
				}
			}
		}

		/* copy the exact forcing term to the right hand side */
		for(int k = 0; k<gridPoints[2]; k++){
			for(int j = 0; j<gridPoints[1]; j++){
				int c = 5*ix(0,j,k);
				System.arraycopy(forcing, c, rhs, c, 5*gridPoints[0]);
			}
		}

		/* xi-direction flux differences */
		if (detailedTiming) timer.start(T_RHSX);
		for(int k = 1; k<=nz2; k++){
			for(int j = 1; j<=ny2; j++){
				for(int i = 1; i<=nx2; i++){
					int s = ix(i,j,k), sp = s + 1, sm = s - 1;
					int p = 5*s, pp = p + 5, pm = p - 5;
					double uijk = us[s];
					double up1 = us[sp];
					double um1 = us[sm];
					rhs[p+0] = rhs[p+0] + dx1tx1 * (u[pp+0] - 2.0*u[p+0] + u[pm+0]) - tx2 * (u[pp+1] - u[pm+1]);
					rhs[p+1] = rhs[p+1] + dx2tx1 * (u[pp+1] - 2.0*u[p+1] + u[pm+1]) + xxcon2*con43 * (up1 - 2.0*uijk + um1) - tx2 * (u[pp+1]*up1 - u[pm+1]*um1 + (u[pp+4]- square[sp]- u[pm+4]+ square[sm])* c2);
					rhs[p+2] = rhs[p+2] + dx3tx1 * (u[pp+2] - 2.0*u[p+2] + u[pm+2]) + xxcon2 * (vs[sp] - 2.0*vs[s] + vs[sm]) - tx2 * (u[pp+2]*up1 - u[pm+2]*um1);
					rhs[p+3] = rhs[p+3] + dx4tx1 * (u[pp+3] - 2.0*u[p+3] + u[pm+3]) + xxcon2 * (ws[sp] - 2.0*ws[s] + ws[sm]) - tx2 * (u[pp+3]*up1 - u[pm+3]*um1);
					rhs[p+4] = rhs[p+4] + dx5tx1 * (u[pp+4] - 2.0*u[p+4] + u[pm+4]) + xxcon3 * (qs[sp] - 2.0*qs[s] + qs[sm]) + xxcon4 * (up1*up1 - 2.0*uijk*uijk + um1*um1) + xxcon5 * (u[pp+4]*rhoI[sp] - 2.0*u[p+4]*rhoI[s] + u[pm+4]*rhoI[sm]) - tx2 * ( (c1*u[pp+4] - c2*square[sp])*up1 - (c1*u[pm+4] - c2*square[sm])*um1 );
				}
			}
		}
		dissipation(0);
		if (detailedTiming) timer.stop(T_RHSX);

		/* eta-direction flux differences */
		if (detailedTiming) timer.start(T_RHSY);
		for(int k = 1; k<=nz2; k++){
			for(int j = 1; j<=ny2; j++){
				for(int i = 1; i<=nx2; i++){
					int s = ix(i,j,k), sp = s + ldx, sm = s - ldx;
					int p = 5*s, pp = p + 5*ldx, pm = p - 5*ldx;
					double vijk = vs[s];
					double vp1 = vs[sp];
					double vm1 = vs[sm];
					rhs[p+0] = rhs[p+0] + dy1ty1 * (u[pp+0] - 2.0*u[p+0] + u[pm+0]) - ty2 * (u[pp+2] - u[pm+2]);
					rhs[p+1] = rhs[p+1] + dy2ty1 * (u[pp+1] - 2.0*u[p+1] + u[pm+1]) + yycon2 * (us[sp] - 2.0*us[s] + us[sm]) - ty2 * (u[pp+1]*vp1 - u[pm+1]*vm1);
					rhs[p+2] = rhs[p+2] + dy3ty1 * (u[pp+2] - 2.0*u[p+2] + u[pm+2]) + yycon2*con43 * (vp1 - 2.0*vijk + vm1) - ty2 * (u[pp+2]*vp1 - u[pm+2]*vm1 + (u[pp+4] - square[sp] - u[pm+4] + square[sm]) *c2);
					rhs[p+3] = rhs[p+3] + dy4ty1 * (u[pp+3] - 2.0*u[p+3] + u[pm+3]) + yycon2 * (ws[sp] - 2.0*ws[s] + ws[sm]) - ty2 * (u[pp+3]*vp1 - u[pm+3]*vm1);
					rhs[p+4] = rhs[p+4] + dy5ty1 * (u[pp+4] - 2.0*u[p+4] + u[pm+4]) + yycon3 * (qs[sp] - 2.0*qs[s] + qs[sm]) + yycon4 * (vp1*vp1 - 2.0*vijk*vijk + vm1*vm1) + yycon5 * (u[pp+4]*rhoI[sp] - 2.0*u[p+4]*rhoI[s] + u[pm+4]*rhoI[sm]) - ty2 * ((c1*u[pp+4] - c2*square[sp]) * vp1 - (c1*u[pm+4] - c2*square[sm]) * vm1);
				}
			}
		}
		dissipation(1);
		if (detailedTiming) timer.stop(T_RHSY);

		/* zeta-direction flux differences */
		if (detailedTiming) timer.start(T_RHSZ);
		for(int k = 1; k<=nz2; k++){
			for(int j = 1; j<=ny2; j++){
				for(int i = 1; i<=nx2; i++){
					int s = ix(i,j,k), sp = s + sk, sm = s - sk;
					int p = 5*s, pp = p + 5*sk, pm = p - 5*sk;
					double wijk = ws[s];
					double wp1 = ws[sp];
					double wm1 = ws[sm];
					rhs[p+0] = rhs[p+0] + dz1tz1 * (u[pp+0] - 2.0*u[p+0] + u[pm+0]) - tz2 * (u[pp+3] - u[pm+3]);
					rhs[p+1] = rhs[p+1] + dz2tz1 * (u[pp+1] - 2.0*u[p+1] + u[pm+1]) + zzcon2 * (us[sp] - 2.0*us[s] + us[sm]) - tz2 * (u[pp+1]*wp1 - u[pm+1]*wm1);
					rhs[p+2] = rhs[p+2] + dz3tz1 * (u[pp+2] - 2.0*u[p+2] + u[pm+2]) + zzcon2 * (vs[sp] - 2.0*vs[s] + vs[sm]) - tz2 * (u[pp+2]*wp1 - u[pm+2]*wm1);
					rhs[p+3] = rhs[p+3] + dz4tz1 * (u[pp+3] - 2.0*u[p+3] + u[pm+3]) + zzcon2*con43 * (wp1 - 2.0*wijk + wm1) - tz2 * (u[pp+3]*wp1 - u[pm+3]*wm1 + (u[pp+4] - square[sp] - u[pm+4] + square[sm]) *c2);
					rhs[p+4] = rhs[p+4] + dz5tz1 * (u[pp+4] - 2.0*u[p+4] + u[pm+4]) + zzcon3 * (qs[sp] - 2.0*qs[s] + qs[sm]) + zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + wm1*wm1) + zzcon5 * (u[pp+4]*rhoI[sp] - 2.0*u[p+4]*rhoI[s] + u[pm+4]*rhoI[sm]) - tz2 * ( (c1*u[pp+4] - c2*square[sp])*wp1 - (c1*u[pm+4] - c2*square[sm])*wm1);
				}
			}
		}
		dissipation(2);
		if (detailedTiming) timer.stop(T_RHSZ);

		for(int k = 1; k<=nz2; k++){
			for(int j = 1; j<=ny2; j++){
				for(int i = 1; i<=nx2; i++){
					int p = 5*ix(i,j,k);
					rhs[p]   = rhs[p]   * dt;
					rhs[p+1] = rhs[p+1] * dt;
					rhs[p+2] = rhs[p+2] * dt;
					rhs[p+3] = rhs[p+3] * dt;
					rhs[p+4] = rhs[p+4] * dt;
				}
			}
		}

		if (detailedTiming) timer.stop(T_RHS);
	}

	/*
	 * Jacobians fjac and njac of the point t of a line, in the original
	 * order of the operations of each direction (x_solve, y_solve and
	 * z_solve differ in the association of several products). The values
	 * of the point (index s) are u1..u5, qsv and sq; tmp1..tmp3 are the
	 * powers of the reciprocal density.
	 */
	private void jacobiansX(int t, int s){

		int f = 25*t;
		int p = 5*s;
		double u1 = u[p], u2 = u[p+1], u3 = u[p+2], u4 = u[p+3], u5 = u[p+4];
		double qsv = qs[s], sq = square[s];
		double tmp1 = rhoI[s];
		double tmp2 = tmp1 * tmp1;
		double tmp3 = tmp1 * tmp2;
		fjac[f+0] = 0.0;
		fjac[f+5] = 1.0;
		fjac[f+10] = 0.0;
		fjac[f+15] = 0.0;
		fjac[f+20] = 0.0;
		fjac[f+1] = -(u2 * tmp2 * u2) + c2 * qsv;
		fjac[f+6] = ( 2.0 - c2 ) * ( u2 / u1 );
		fjac[f+11] = - c2 * ( u3 * tmp1 );
		fjac[f+16] = - c2 * ( u4 * tmp1 );
		fjac[f+21] = c2;
		fjac[f+2] = - ( u2*u3 ) * tmp2;
		fjac[f+7] = u3 * tmp1;
		fjac[f+12] = u2 * tmp1;
		fjac[f+17] = 0.0;
		fjac[f+22] = 0.0;
		fjac[f+3] = - ( u2*u4 ) * tmp2;
		fjac[f+8] = u4 * tmp1;
		fjac[f+13] = 0.0;
		fjac[f+18] = u2 * tmp1;
		fjac[f+23] = 0.0;
		fjac[f+4] = ( c2 * 2.0 * sq - c1 * u5 ) * ( u2 * tmp2 );
		fjac[f+9] = c1 * u5 * tmp1 - c2 * ( u2*u2 * tmp2 + qsv );
		fjac[f+14] = - c2 * ( u3*u2 ) * tmp2;
		fjac[f+19] = - c2 * ( u4*u2 ) * tmp2;
		fjac[f+24] = c1 * ( u2 * tmp1 );
		njac[f+0] = 0.0;
		njac[f+5] = 0.0;
		njac[f+10] = 0.0;
		njac[f+15] = 0.0;
		njac[f+20] = 0.0;
		njac[f+1] = - con43 * c3c4 * tmp2 * u2;
		njac[f+6] = con43 * c3c4 * tmp1;
		njac[f+11] = 0.0;
		njac[f+16] = 0.0;
		njac[f+21] = 0.0;
		njac[f+2] = - c3c4 * tmp2 * u3;
		njac[f+7] = 0.0;
		njac[f+12] = c3c4 * tmp1;
		njac[f+17] = 0.0;
		njac[f+22] = 0.0;
		njac[f+3] = - c3c4 * tmp2 * u4;
		njac[f+8] = 0.0;
		njac[f+13] = 0.0;
		njac[f+18] = c3c4 * tmp1;
		njac[f+23] = 0.0;
		njac[f+4] = - ( con43 * c3c4 - c1345 ) * tmp3 * (u2*u2) - ( c3c4 - c1345 ) * tmp3 * (u3*u3) - ( c3c4 - c1345 ) * tmp3 * (u4*u4) - c1345 * tmp2 * u5;
		njac[f+9] = ( con43 * c3c4 - c1345 ) * tmp2 * u2;
		njac[f+14] = ( c3c4 - c1345 ) * tmp2 * u3;
		njac[f+19] = ( c3c4 - c1345 ) * tmp2 * u4;
		njac[f+24] = ( c1345 ) * tmp1;
	}

	private void jacobiansY(int t, int s){

		int f = 25*t;
		int p = 5*s;
		double u1 = u[p], u2 = u[p+1], u3 = u[p+2], u4 = u[p+3], u5 = u[p+4];
		double qsv = qs[s], sq = square[s];
		double tmp1 = rhoI[s];
		double tmp2 = tmp1 * tmp1;
		double tmp3 = tmp1 * tmp2;
		fjac[f+0] = 0.0;
		fjac[f+5] = 0.0;
		fjac[f+10] = 1.0;
		fjac[f+15] = 0.0;
		fjac[f+20] = 0.0;
		fjac[f+1] = - ( u2*u3 ) * tmp2;
		fjac[f+6] = u3 * tmp1;
		fjac[f+11] = u2 * tmp1;
		fjac[f+16] = 0.0;
		fjac[f+21] = 0.0;
		fjac[f+2] = - ( u3*u3*tmp2) + c2 * qsv;
		fjac[f+7] = - c2 * u2 * tmp1;
		fjac[f+12] = ( 2.0 - c2 ) * u3 * tmp1;
		fjac[f+17] = - c2 * u4 * tmp1;
		fjac[f+22] = c2;
		fjac[f+3] = - ( u3*u4 ) * tmp2;
		fjac[f+8] = 0.0;
		fjac[f+13] = u4 * tmp1;
		fjac[f+18] = u3 * tmp1;
		fjac[f+23] = 0.0;
		fjac[f+4] = ( c2 * 2.0 * sq - c1 * u5 ) * u3 * tmp2;
		fjac[f+9] = - c2 * u2*u3 * tmp2;
		fjac[f+14] = c1 * u5 * tmp1 - c2 * ( qsv + u3*u3 * tmp2 );
		fjac[f+19] = - c2 * ( u3*u4 ) * tmp2;
		fjac[f+24] = c1 * u3 * tmp1;
		njac[f+0] = 0.0;
		njac[f+5] = 0.0;
		njac[f+10] = 0.0;
		njac[f+15] = 0.0;
		njac[f+20] = 0.0;
		njac[f+1] = - c3c4 * tmp2 * u2;
		njac[f+6] = c3c4 * tmp1;
		njac[f+11] = 0.0;
		njac[f+16] = 0.0;
		njac[f+21] = 0.0;
		njac[f+2] = - con43 * c3c4 * tmp2 * u3;
		njac[f+7] = 0.0;
		njac[f+12] = con43 * c3c4 * tmp1;
		njac[f+17] = 0.0;
		njac[f+22] = 0.0;
		njac[f+3] = - c3c4 * tmp2 * u4;
		njac[f+8] = 0.0;
		njac[f+13] = 0.0;
		njac[f+18] = c3c4 * tmp1;
		njac[f+23] = 0.0;
		njac[f+4] = - ( c3c4 - c1345 ) * tmp3 * (u2*u2) - ( con43 * c3c4 - c1345 ) * tmp3 * (u3*u3) - ( c3c4 - c1345 ) * tmp3 * (u4*u4) - c1345 * tmp2 * u5;
		njac[f+9] = ( c3c4 - c1345 ) * tmp2 * u2;
		njac[f+14] = ( con43 * c3c4 - c1345 ) * tmp2 * u3;
		njac[f+19] = ( c3c4 - c1345 ) * tmp2 * u4;
		njac[f+24] = ( c1345 ) * tmp1;
	}

	private void jacobiansZ(int t, int s){

		int f = 25*t;
		int p = 5*s;
		double u1 = u[p], u2 = u[p+1], u3 = u[p+2], u4 = u[p+3], u5 = u[p+4];
		double qsv = qs[s], sq = square[s];
		/* (z_solve computes 1/u(1) itself instead of reading rho_i) */
		double tmp1 = 1.0e+00 / u1;
		double tmp2 = tmp1 * tmp1;
		double tmp3 = tmp1 * tmp2;
		fjac[f+0] = 0.0;
		fjac[f+5] = 0.0;
		fjac[f+10] = 0.0;
		fjac[f+15] = 1.0;
		fjac[f+20] = 0.0;
		fjac[f+1] = - ( u2*u4 ) * tmp2;
		fjac[f+6] = u4 * tmp1;
		fjac[f+11] = 0.0;
		fjac[f+16] = u2 * tmp1;
		fjac[f+21] = 0.0;
		fjac[f+2] = - ( u3*u4 ) * tmp2;
		fjac[f+7] = 0.0;
		fjac[f+12] = u4 * tmp1;
		fjac[f+17] = u3 * tmp1;
		fjac[f+22] = 0.0;
		fjac[f+3] = - (u4*u4 * tmp2 ) + c2 * qsv;
		fjac[f+8] = - c2 * u2 * tmp1;
		fjac[f+13] = - c2 * u3 * tmp1;
		fjac[f+18] = ( 2.0 - c2 ) * u4 * tmp1;
		fjac[f+23] = c2;
		fjac[f+4] = ( c2 * 2.0 * sq - c1 * u5 ) * u4 * tmp2;
		fjac[f+9] = - c2 * ( u2*u4 ) * tmp2;
		fjac[f+14] = - c2 * ( u3*u4 ) * tmp2;
		fjac[f+19] = c1 * ( u5 * tmp1 ) - c2 * ( qsv + u4*u4 * tmp2 );
		fjac[f+24] = c1 * u4 * tmp1;
		njac[f+0] = 0.0;
		njac[f+5] = 0.0;
		njac[f+10] = 0.0;
		njac[f+15] = 0.0;
		njac[f+20] = 0.0;
		njac[f+1] = - c3c4 * tmp2 * u2;
		njac[f+6] = c3c4 * tmp1;
		njac[f+11] = 0.0;
		njac[f+16] = 0.0;
		njac[f+21] = 0.0;
		njac[f+2] = - c3c4 * tmp2 * u3;
		njac[f+7] = 0.0;
		njac[f+12] = c3c4 * tmp1;
		njac[f+17] = 0.0;
		njac[f+22] = 0.0;
		njac[f+3] = - con43 * c3c4 * tmp2 * u4;
		njac[f+8] = 0.0;
		njac[f+13] = 0.0;
		njac[f+18] = con43 * c3 * c4 * tmp1;
		njac[f+23] = 0.0;
		njac[f+4] = - ( c3c4 - c1345 ) * tmp3 * (u2*u2) - ( c3c4 - c1345 ) * tmp3 * (u3*u3) - ( con43 * c3c4 - c1345 ) * tmp3 * (u4*u4) - c1345 * tmp2 * u5;
		njac[f+9] = ( c3c4 - c1345 ) * tmp2 * u2;
		njac[f+14] = ( c3c4 - c1345 ) * tmp2 * u3;
		njac[f+19] = ( con43 * c3c4 - c1345 ) * tmp2 * u4;
		njac[f+24] = ( c1345 )* tmp1;
	}

	/*
	 * Block-tridiagonal solve along every grid line of direction dir
	 * (0 = x_solve, 1 = y_solve, 2 = z_solve of the original). The assembly
	 * of the blocks and the elimination are identical in the three
	 * directions (checked against the three Fortran files); only the
	 * jacobians and the constants change.
	 */
	private void lineSolve(int dir){

		int timerId = dir == 0 ? T_XSOLVE : dir == 1 ? T_YSOLVE : T_ZSOLVE;
		if (detailedTiming) timer.start(timerId);

		int isize = gridPoints[dir] - 1;
		int step = dir == 0 ? 1 : dir == 1 ? ldx : sk;	/* distance between the points of a line */
		double tc1 = dir == 0 ? tx1 : dir == 1 ? ty1 : tz1;
		double tc2 = dir == 0 ? tx2 : dir == 1 ? ty2 : tz2;
		double dd[] = dir == 0 ? new double[] {dx1, dx2, dx3, dx4, dx5}
				: dir == 1 ? new double[] {dy1, dy2, dy3, dy4, dy5}
				: new double[] {dz1, dz2, dz3, dz4, dz5};
		double tmp1 = dt * tc1;
		double tmp2 = dt * tc2;

		/* the lines: x and y sweep k and (j or i); z sweeps j and i */
		int bmax = dir == 2 ? ny2 : nz2;
		int amax = dir == 0 ? ny2 : nx2;

		for(int b = 1; b<=bmax; b++){
			for(int a = 1; a<=amax; a++){

				int s0 = dir == 0 ? ix(0,a,b) : dir == 1 ? ix(a,0,b) : ix(a,b,0);

				for(int t = 0; t<=isize; t++){
					int s = s0 + t*step;
					if(dir == 0)
						jacobiansX(t, s);
					else if(dir == 1)
						jacobiansY(t, s);
					else
						jacobiansZ(t, s);
				}

				/* lhsinit */
				lhsInit(0);
				lhsInit(isize);

				/* the three blocks without the if(m == n); the diagonal is completed afterwards (same operations per element) */
				for(int t = 1; t<=isize-1; t++){
					int lo = 75*t, f0 = 25*(t-1), f1 = 25*t, f2 = 25*(t+1);
					for(int e = 0; e<25; e++){
						lhs[lo+e] = - tmp2 * fjac[f0+e]
								- tmp1 * njac[f0+e];
						lhs[lo+25+e] = tmp1 * 2.0 * njac[f1+e];
						lhs[lo+50+e] = tmp2 * fjac[f2+e]
								- tmp1 * njac[f2+e];
					}
					for(int m = 0; m<5; m++){
						int e = 6*m;
						lhs[lo+e] = lhs[lo+e] - tmp1 * dd[m];
						lhs[lo+25+e] = 1.0
								+ tmp1 * 2.0 * njac[f1+e]
								+ tmp1 * 2.0 * dd[m];
						lhs[lo+50+e] = lhs[lo+50+e] - tmp1 * dd[m];
					}
				}

				if (detailedTiming) timer.start(T_SOLSUB);

				/* performs guaussian elimination on this cell; the rhs of
				 * the point t is rhs[5*(s0+t*step)..] */
				binvcrhs(lhs, 75*0+25, lhs, 75*0+50, rhs, 5*s0);

				for(int t = 1; t<=isize-1; t++){
					int pm1 = 5*(s0 + (t-1)*step);
					int p0 = 5*(s0 + t*step);
					matvecSub(lhs, 75*t, rhs, pm1, rhs, p0);
					matmulSub(lhs, 75*t, lhs, 75*(t-1)+50, lhs, 75*t+25);
					binvcrhs(lhs, 75*t+25, lhs, 75*t+50, rhs, p0);
				}

				{
					int pm1 = 5*(s0 + (isize-1)*step);
					int p0 = 5*(s0 + isize*step);
					matvecSub(lhs, 75*isize, rhs, pm1, rhs, p0);
					matmulSub(lhs, 75*isize, lhs, 75*(isize-1)+50, lhs, 75*isize+25);
					binvrhs(lhs, 75*isize+25, rhs, p0);
				}

				if (detailedTiming) timer.stop(T_SOLSUB);

				/* back substitution */
				for(int t = isize-1; t>=0; t--){
					int p0 = 5*(s0 + t*step);
					int p1 = 5*(s0 + (t+1)*step);
					for(int m = 0; m<5; m++){
						for(int n = 0; n<5; n++){
							rhs[p0+m] = rhs[p0+m]
									- lhs[75*t+50+m+5*n]*rhs[p1+n];
						}
					}
				}
			}
		}

		if (detailedTiming) timer.stop(timerId);
	}

	/* The two ends of a line: zero blocks aa and cc, identity as bb */
	private void lhsInit(int t){
		int lo = 75*t;
		for(int m = 0; m<5; m++){
			for(int n = 0; n<5; n++){
				lhs[lo+m+5*n] = 0.0;
				lhs[lo+25+m+5*n] = 0.0;
				lhs[lo+50+m+5*n] = 0.0;
			}
			lhs[lo+25+m+5*m] = 1.0;
		}
	}

	/*
	 * bvec = bvec - ablock*avec
	 * (ablock is a 5x5 block at a[ao], first index fastest; avec and bvec are
	 * 5-vectors at x[xo] and b[bo])
	 */
	private void matvecSub(double a[], int ao, double x[], int xo, double b[], int bo){
		b[bo] = b[bo] - a[ao]*x[xo]
				- a[ao+5]*x[xo+1]
				- a[ao+10]*x[xo+2]
				- a[ao+15]*x[xo+3]
				- a[ao+20]*x[xo+4];
		b[bo+1] = b[bo+1] - a[ao+1]*x[xo]
				- a[ao+6]*x[xo+1]
				- a[ao+11]*x[xo+2]
				- a[ao+16]*x[xo+3]
				- a[ao+21]*x[xo+4];
		b[bo+2] = b[bo+2] - a[ao+2]*x[xo]
				- a[ao+7]*x[xo+1]
				- a[ao+12]*x[xo+2]
				- a[ao+17]*x[xo+3]
				- a[ao+22]*x[xo+4];
		b[bo+3] = b[bo+3] - a[ao+3]*x[xo]
				- a[ao+8]*x[xo+1]
				- a[ao+13]*x[xo+2]
				- a[ao+18]*x[xo+3]
				- a[ao+23]*x[xo+4];
		b[bo+4] = b[bo+4] - a[ao+4]*x[xo]
				- a[ao+9]*x[xo+1]
				- a[ao+14]*x[xo+2]
				- a[ao+19]*x[xo+3]
				- a[ao+24]*x[xo+4];
	}

	/* cblock = cblock - ablock*bblock (5x5 blocks, first index fastest) */
	private void matmulSub(double a[], int ao, double b[], int bo, double c[], int co){
		c[co] = c[co] - a[ao]*b[bo]
				- a[ao+5]*b[bo+1]
				- a[ao+10]*b[bo+2]
				- a[ao+15]*b[bo+3]
				- a[ao+20]*b[bo+4];
		c[co+1] = c[co+1] - a[ao+1]*b[bo]
				- a[ao+6]*b[bo+1]
				- a[ao+11]*b[bo+2]
				- a[ao+16]*b[bo+3]
				- a[ao+21]*b[bo+4];
		c[co+2] = c[co+2] - a[ao+2]*b[bo]
				- a[ao+7]*b[bo+1]
				- a[ao+12]*b[bo+2]
				- a[ao+17]*b[bo+3]
				- a[ao+22]*b[bo+4];
		c[co+3] = c[co+3] - a[ao+3]*b[bo]
				- a[ao+8]*b[bo+1]
				- a[ao+13]*b[bo+2]
				- a[ao+18]*b[bo+3]
				- a[ao+23]*b[bo+4];
		c[co+4] = c[co+4] - a[ao+4]*b[bo]
				- a[ao+9]*b[bo+1]
				- a[ao+14]*b[bo+2]
				- a[ao+19]*b[bo+3]
				- a[ao+24]*b[bo+4];
		c[co+5] = c[co+5] - a[ao]*b[bo+5]
				- a[ao+5]*b[bo+6]
				- a[ao+10]*b[bo+7]
				- a[ao+15]*b[bo+8]
				- a[ao+20]*b[bo+9];
		c[co+6] = c[co+6] - a[ao+1]*b[bo+5]
				- a[ao+6]*b[bo+6]
				- a[ao+11]*b[bo+7]
				- a[ao+16]*b[bo+8]
				- a[ao+21]*b[bo+9];
		c[co+7] = c[co+7] - a[ao+2]*b[bo+5]
				- a[ao+7]*b[bo+6]
				- a[ao+12]*b[bo+7]
				- a[ao+17]*b[bo+8]
				- a[ao+22]*b[bo+9];
		c[co+8] = c[co+8] - a[ao+3]*b[bo+5]
				- a[ao+8]*b[bo+6]
				- a[ao+13]*b[bo+7]
				- a[ao+18]*b[bo+8]
				- a[ao+23]*b[bo+9];
		c[co+9] = c[co+9] - a[ao+4]*b[bo+5]
				- a[ao+9]*b[bo+6]
				- a[ao+14]*b[bo+7]
				- a[ao+19]*b[bo+8]
				- a[ao+24]*b[bo+9];
		c[co+10] = c[co+10] - a[ao]*b[bo+10]
				- a[ao+5]*b[bo+11]
				- a[ao+10]*b[bo+12]
				- a[ao+15]*b[bo+13]
				- a[ao+20]*b[bo+14];
		c[co+11] = c[co+11] - a[ao+1]*b[bo+10]
				- a[ao+6]*b[bo+11]
				- a[ao+11]*b[bo+12]
				- a[ao+16]*b[bo+13]
				- a[ao+21]*b[bo+14];
		c[co+12] = c[co+12] - a[ao+2]*b[bo+10]
				- a[ao+7]*b[bo+11]
				- a[ao+12]*b[bo+12]
				- a[ao+17]*b[bo+13]
				- a[ao+22]*b[bo+14];
		c[co+13] = c[co+13] - a[ao+3]*b[bo+10]
				- a[ao+8]*b[bo+11]
				- a[ao+13]*b[bo+12]
				- a[ao+18]*b[bo+13]
				- a[ao+23]*b[bo+14];
		c[co+14] = c[co+14] - a[ao+4]*b[bo+10]
				- a[ao+9]*b[bo+11]
				- a[ao+14]*b[bo+12]
				- a[ao+19]*b[bo+13]
				- a[ao+24]*b[bo+14];
		c[co+15] = c[co+15] - a[ao]*b[bo+15]
				- a[ao+5]*b[bo+16]
				- a[ao+10]*b[bo+17]
				- a[ao+15]*b[bo+18]
				- a[ao+20]*b[bo+19];
		c[co+16] = c[co+16] - a[ao+1]*b[bo+15]
				- a[ao+6]*b[bo+16]
				- a[ao+11]*b[bo+17]
				- a[ao+16]*b[bo+18]
				- a[ao+21]*b[bo+19];
		c[co+17] = c[co+17] - a[ao+2]*b[bo+15]
				- a[ao+7]*b[bo+16]
				- a[ao+12]*b[bo+17]
				- a[ao+17]*b[bo+18]
				- a[ao+22]*b[bo+19];
		c[co+18] = c[co+18] - a[ao+3]*b[bo+15]
				- a[ao+8]*b[bo+16]
				- a[ao+13]*b[bo+17]
				- a[ao+18]*b[bo+18]
				- a[ao+23]*b[bo+19];
		c[co+19] = c[co+19] - a[ao+4]*b[bo+15]
				- a[ao+9]*b[bo+16]
				- a[ao+14]*b[bo+17]
				- a[ao+19]*b[bo+18]
				- a[ao+24]*b[bo+19];
		c[co+20] = c[co+20] - a[ao]*b[bo+20]
				- a[ao+5]*b[bo+21]
				- a[ao+10]*b[bo+22]
				- a[ao+15]*b[bo+23]
				- a[ao+20]*b[bo+24];
		c[co+21] = c[co+21] - a[ao+1]*b[bo+20]
				- a[ao+6]*b[bo+21]
				- a[ao+11]*b[bo+22]
				- a[ao+16]*b[bo+23]
				- a[ao+21]*b[bo+24];
		c[co+22] = c[co+22] - a[ao+2]*b[bo+20]
				- a[ao+7]*b[bo+21]
				- a[ao+12]*b[bo+22]
				- a[ao+17]*b[bo+23]
				- a[ao+22]*b[bo+24];
		c[co+23] = c[co+23] - a[ao+3]*b[bo+20]
				- a[ao+8]*b[bo+21]
				- a[ao+13]*b[bo+22]
				- a[ao+18]*b[bo+23]
				- a[ao+23]*b[bo+24];
		c[co+24] = c[co+24] - a[ao+4]*b[bo+20]
				- a[ao+9]*b[bo+21]
				- a[ao+14]*b[bo+22]
				- a[ao+19]*b[bo+23]
				- a[ao+24]*b[bo+24];
	}

	/*
	 * Gauss-Jordan elimination of the 5x5 block L (at l[lo]) applied to the 5x5
	 * block C (at c[co]) and to the vector R (at r[ro]): the unrolled
	 * binvcrhs of the original, in the same order of operations (the
	 * elements below and on the diagonal of L are never read again, so
	 * only the columns after the pivot are updated).
	 */
	private void binvcrhs(double l[], int lo, double c[], int co, double r[], int ro){
		double pivot, coeff;
		pivot = 1.00/l[lo];
		l[lo+5] = l[lo+5]*pivot;
		l[lo+10] = l[lo+10]*pivot;
		l[lo+15] = l[lo+15]*pivot;
		l[lo+20] = l[lo+20]*pivot;
		c[co] = c[co]*pivot;
		c[co+5] = c[co+5]*pivot;
		c[co+10] = c[co+10]*pivot;
		c[co+15] = c[co+15]*pivot;
		c[co+20] = c[co+20]*pivot;
		r[ro] = r[ro]*pivot;
		coeff = l[lo+1];
		l[lo+6] = l[lo+6] - coeff*l[lo+5];
		l[lo+11] = l[lo+11] - coeff*l[lo+10];
		l[lo+16] = l[lo+16] - coeff*l[lo+15];
		l[lo+21] = l[lo+21] - coeff*l[lo+20];
		c[co+1] = c[co+1] - coeff*c[co];
		c[co+6] = c[co+6] - coeff*c[co+5];
		c[co+11] = c[co+11] - coeff*c[co+10];
		c[co+16] = c[co+16] - coeff*c[co+15];
		c[co+21] = c[co+21] - coeff*c[co+20];
		r[ro+1] = r[ro+1] - coeff*r[ro];
		coeff = l[lo+2];
		l[lo+7] = l[lo+7] - coeff*l[lo+5];
		l[lo+12] = l[lo+12] - coeff*l[lo+10];
		l[lo+17] = l[lo+17] - coeff*l[lo+15];
		l[lo+22] = l[lo+22] - coeff*l[lo+20];
		c[co+2] = c[co+2] - coeff*c[co];
		c[co+7] = c[co+7] - coeff*c[co+5];
		c[co+12] = c[co+12] - coeff*c[co+10];
		c[co+17] = c[co+17] - coeff*c[co+15];
		c[co+22] = c[co+22] - coeff*c[co+20];
		r[ro+2] = r[ro+2] - coeff*r[ro];
		coeff = l[lo+3];
		l[lo+8] = l[lo+8] - coeff*l[lo+5];
		l[lo+13] = l[lo+13] - coeff*l[lo+10];
		l[lo+18] = l[lo+18] - coeff*l[lo+15];
		l[lo+23] = l[lo+23] - coeff*l[lo+20];
		c[co+3] = c[co+3] - coeff*c[co];
		c[co+8] = c[co+8] - coeff*c[co+5];
		c[co+13] = c[co+13] - coeff*c[co+10];
		c[co+18] = c[co+18] - coeff*c[co+15];
		c[co+23] = c[co+23] - coeff*c[co+20];
		r[ro+3] = r[ro+3] - coeff*r[ro];
		coeff = l[lo+4];
		l[lo+9] = l[lo+9] - coeff*l[lo+5];
		l[lo+14] = l[lo+14] - coeff*l[lo+10];
		l[lo+19] = l[lo+19] - coeff*l[lo+15];
		l[lo+24] = l[lo+24] - coeff*l[lo+20];
		c[co+4] = c[co+4] - coeff*c[co];
		c[co+9] = c[co+9] - coeff*c[co+5];
		c[co+14] = c[co+14] - coeff*c[co+10];
		c[co+19] = c[co+19] - coeff*c[co+15];
		c[co+24] = c[co+24] - coeff*c[co+20];
		r[ro+4] = r[ro+4] - coeff*r[ro];
		pivot = 1.00/l[lo+6];
		l[lo+11] = l[lo+11]*pivot;
		l[lo+16] = l[lo+16]*pivot;
		l[lo+21] = l[lo+21]*pivot;
		c[co+1] = c[co+1]*pivot;
		c[co+6] = c[co+6]*pivot;
		c[co+11] = c[co+11]*pivot;
		c[co+16] = c[co+16]*pivot;
		c[co+21] = c[co+21]*pivot;
		r[ro+1] = r[ro+1]*pivot;
		coeff = l[lo+5];
		l[lo+10] = l[lo+10] - coeff*l[lo+11];
		l[lo+15] = l[lo+15] - coeff*l[lo+16];
		l[lo+20] = l[lo+20] - coeff*l[lo+21];
		c[co] = c[co] - coeff*c[co+1];
		c[co+5] = c[co+5] - coeff*c[co+6];
		c[co+10] = c[co+10] - coeff*c[co+11];
		c[co+15] = c[co+15] - coeff*c[co+16];
		c[co+20] = c[co+20] - coeff*c[co+21];
		r[ro] = r[ro] - coeff*r[ro+1];
		coeff = l[lo+7];
		l[lo+12] = l[lo+12] - coeff*l[lo+11];
		l[lo+17] = l[lo+17] - coeff*l[lo+16];
		l[lo+22] = l[lo+22] - coeff*l[lo+21];
		c[co+2] = c[co+2] - coeff*c[co+1];
		c[co+7] = c[co+7] - coeff*c[co+6];
		c[co+12] = c[co+12] - coeff*c[co+11];
		c[co+17] = c[co+17] - coeff*c[co+16];
		c[co+22] = c[co+22] - coeff*c[co+21];
		r[ro+2] = r[ro+2] - coeff*r[ro+1];
		coeff = l[lo+8];
		l[lo+13] = l[lo+13] - coeff*l[lo+11];
		l[lo+18] = l[lo+18] - coeff*l[lo+16];
		l[lo+23] = l[lo+23] - coeff*l[lo+21];
		c[co+3] = c[co+3] - coeff*c[co+1];
		c[co+8] = c[co+8] - coeff*c[co+6];
		c[co+13] = c[co+13] - coeff*c[co+11];
		c[co+18] = c[co+18] - coeff*c[co+16];
		c[co+23] = c[co+23] - coeff*c[co+21];
		r[ro+3] = r[ro+3] - coeff*r[ro+1];
		coeff = l[lo+9];
		l[lo+14] = l[lo+14] - coeff*l[lo+11];
		l[lo+19] = l[lo+19] - coeff*l[lo+16];
		l[lo+24] = l[lo+24] - coeff*l[lo+21];
		c[co+4] = c[co+4] - coeff*c[co+1];
		c[co+9] = c[co+9] - coeff*c[co+6];
		c[co+14] = c[co+14] - coeff*c[co+11];
		c[co+19] = c[co+19] - coeff*c[co+16];
		c[co+24] = c[co+24] - coeff*c[co+21];
		r[ro+4] = r[ro+4] - coeff*r[ro+1];
		pivot = 1.00/l[lo+12];
		l[lo+17] = l[lo+17]*pivot;
		l[lo+22] = l[lo+22]*pivot;
		c[co+2] = c[co+2]*pivot;
		c[co+7] = c[co+7]*pivot;
		c[co+12] = c[co+12]*pivot;
		c[co+17] = c[co+17]*pivot;
		c[co+22] = c[co+22]*pivot;
		r[ro+2] = r[ro+2]*pivot;
		coeff = l[lo+10];
		l[lo+15] = l[lo+15] - coeff*l[lo+17];
		l[lo+20] = l[lo+20] - coeff*l[lo+22];
		c[co] = c[co] - coeff*c[co+2];
		c[co+5] = c[co+5] - coeff*c[co+7];
		c[co+10] = c[co+10] - coeff*c[co+12];
		c[co+15] = c[co+15] - coeff*c[co+17];
		c[co+20] = c[co+20] - coeff*c[co+22];
		r[ro] = r[ro] - coeff*r[ro+2];
		coeff = l[lo+11];
		l[lo+16] = l[lo+16] - coeff*l[lo+17];
		l[lo+21] = l[lo+21] - coeff*l[lo+22];
		c[co+1] = c[co+1] - coeff*c[co+2];
		c[co+6] = c[co+6] - coeff*c[co+7];
		c[co+11] = c[co+11] - coeff*c[co+12];
		c[co+16] = c[co+16] - coeff*c[co+17];
		c[co+21] = c[co+21] - coeff*c[co+22];
		r[ro+1] = r[ro+1] - coeff*r[ro+2];
		coeff = l[lo+13];
		l[lo+18] = l[lo+18] - coeff*l[lo+17];
		l[lo+23] = l[lo+23] - coeff*l[lo+22];
		c[co+3] = c[co+3] - coeff*c[co+2];
		c[co+8] = c[co+8] - coeff*c[co+7];
		c[co+13] = c[co+13] - coeff*c[co+12];
		c[co+18] = c[co+18] - coeff*c[co+17];
		c[co+23] = c[co+23] - coeff*c[co+22];
		r[ro+3] = r[ro+3] - coeff*r[ro+2];
		coeff = l[lo+14];
		l[lo+19] = l[lo+19] - coeff*l[lo+17];
		l[lo+24] = l[lo+24] - coeff*l[lo+22];
		c[co+4] = c[co+4] - coeff*c[co+2];
		c[co+9] = c[co+9] - coeff*c[co+7];
		c[co+14] = c[co+14] - coeff*c[co+12];
		c[co+19] = c[co+19] - coeff*c[co+17];
		c[co+24] = c[co+24] - coeff*c[co+22];
		r[ro+4] = r[ro+4] - coeff*r[ro+2];
		pivot = 1.00/l[lo+18];
		l[lo+23] = l[lo+23]*pivot;
		c[co+3] = c[co+3]*pivot;
		c[co+8] = c[co+8]*pivot;
		c[co+13] = c[co+13]*pivot;
		c[co+18] = c[co+18]*pivot;
		c[co+23] = c[co+23]*pivot;
		r[ro+3] = r[ro+3]*pivot;
		coeff = l[lo+15];
		l[lo+20] = l[lo+20] - coeff*l[lo+23];
		c[co] = c[co] - coeff*c[co+3];
		c[co+5] = c[co+5] - coeff*c[co+8];
		c[co+10] = c[co+10] - coeff*c[co+13];
		c[co+15] = c[co+15] - coeff*c[co+18];
		c[co+20] = c[co+20] - coeff*c[co+23];
		r[ro] = r[ro] - coeff*r[ro+3];
		coeff = l[lo+16];
		l[lo+21] = l[lo+21] - coeff*l[lo+23];
		c[co+1] = c[co+1] - coeff*c[co+3];
		c[co+6] = c[co+6] - coeff*c[co+8];
		c[co+11] = c[co+11] - coeff*c[co+13];
		c[co+16] = c[co+16] - coeff*c[co+18];
		c[co+21] = c[co+21] - coeff*c[co+23];
		r[ro+1] = r[ro+1] - coeff*r[ro+3];
		coeff = l[lo+17];
		l[lo+22] = l[lo+22] - coeff*l[lo+23];
		c[co+2] = c[co+2] - coeff*c[co+3];
		c[co+7] = c[co+7] - coeff*c[co+8];
		c[co+12] = c[co+12] - coeff*c[co+13];
		c[co+17] = c[co+17] - coeff*c[co+18];
		c[co+22] = c[co+22] - coeff*c[co+23];
		r[ro+2] = r[ro+2] - coeff*r[ro+3];
		coeff = l[lo+19];
		l[lo+24] = l[lo+24] - coeff*l[lo+23];
		c[co+4] = c[co+4] - coeff*c[co+3];
		c[co+9] = c[co+9] - coeff*c[co+8];
		c[co+14] = c[co+14] - coeff*c[co+13];
		c[co+19] = c[co+19] - coeff*c[co+18];
		c[co+24] = c[co+24] - coeff*c[co+23];
		r[ro+4] = r[ro+4] - coeff*r[ro+3];
		pivot = 1.00/l[lo+24];
		c[co+4] = c[co+4]*pivot;
		c[co+9] = c[co+9]*pivot;
		c[co+14] = c[co+14]*pivot;
		c[co+19] = c[co+19]*pivot;
		c[co+24] = c[co+24]*pivot;
		r[ro+4] = r[ro+4]*pivot;
		coeff = l[lo+20];
		c[co] = c[co] - coeff*c[co+4];
		c[co+5] = c[co+5] - coeff*c[co+9];
		c[co+10] = c[co+10] - coeff*c[co+14];
		c[co+15] = c[co+15] - coeff*c[co+19];
		c[co+20] = c[co+20] - coeff*c[co+24];
		r[ro] = r[ro] - coeff*r[ro+4];
		coeff = l[lo+21];
		c[co+1] = c[co+1] - coeff*c[co+4];
		c[co+6] = c[co+6] - coeff*c[co+9];
		c[co+11] = c[co+11] - coeff*c[co+14];
		c[co+16] = c[co+16] - coeff*c[co+19];
		c[co+21] = c[co+21] - coeff*c[co+24];
		r[ro+1] = r[ro+1] - coeff*r[ro+4];
		coeff = l[lo+22];
		c[co+2] = c[co+2] - coeff*c[co+4];
		c[co+7] = c[co+7] - coeff*c[co+9];
		c[co+12] = c[co+12] - coeff*c[co+14];
		c[co+17] = c[co+17] - coeff*c[co+19];
		c[co+22] = c[co+22] - coeff*c[co+24];
		r[ro+2] = r[ro+2] - coeff*r[ro+4];
		coeff = l[lo+23];
		c[co+3] = c[co+3] - coeff*c[co+4];
		c[co+8] = c[co+8] - coeff*c[co+9];
		c[co+13] = c[co+13] - coeff*c[co+14];
		c[co+18] = c[co+18] - coeff*c[co+19];
		c[co+23] = c[co+23] - coeff*c[co+24];
		r[ro+3] = r[ro+3] - coeff*r[ro+4];
	}

	/* The same as binvcrhs, without the matrix C (last cell of a line) */
	private void binvrhs(double l[], int lo, double r[], int ro){
		double pivot, coeff;
		pivot = 1.00/l[lo];
		l[lo+5] = l[lo+5]*pivot;
		l[lo+10] = l[lo+10]*pivot;
		l[lo+15] = l[lo+15]*pivot;
		l[lo+20] = l[lo+20]*pivot;
		r[ro] = r[ro]*pivot;
		coeff = l[lo+1];
		l[lo+6] = l[lo+6] - coeff*l[lo+5];
		l[lo+11] = l[lo+11] - coeff*l[lo+10];
		l[lo+16] = l[lo+16] - coeff*l[lo+15];
		l[lo+21] = l[lo+21] - coeff*l[lo+20];
		r[ro+1] = r[ro+1] - coeff*r[ro];
		coeff = l[lo+2];
		l[lo+7] = l[lo+7] - coeff*l[lo+5];
		l[lo+12] = l[lo+12] - coeff*l[lo+10];
		l[lo+17] = l[lo+17] - coeff*l[lo+15];
		l[lo+22] = l[lo+22] - coeff*l[lo+20];
		r[ro+2] = r[ro+2] - coeff*r[ro];
		coeff = l[lo+3];
		l[lo+8] = l[lo+8] - coeff*l[lo+5];
		l[lo+13] = l[lo+13] - coeff*l[lo+10];
		l[lo+18] = l[lo+18] - coeff*l[lo+15];
		l[lo+23] = l[lo+23] - coeff*l[lo+20];
		r[ro+3] = r[ro+3] - coeff*r[ro];
		coeff = l[lo+4];
		l[lo+9] = l[lo+9] - coeff*l[lo+5];
		l[lo+14] = l[lo+14] - coeff*l[lo+10];
		l[lo+19] = l[lo+19] - coeff*l[lo+15];
		l[lo+24] = l[lo+24] - coeff*l[lo+20];
		r[ro+4] = r[ro+4] - coeff*r[ro];
		pivot = 1.00/l[lo+6];
		l[lo+11] = l[lo+11]*pivot;
		l[lo+16] = l[lo+16]*pivot;
		l[lo+21] = l[lo+21]*pivot;
		r[ro+1] = r[ro+1]*pivot;
		coeff = l[lo+5];
		l[lo+10] = l[lo+10] - coeff*l[lo+11];
		l[lo+15] = l[lo+15] - coeff*l[lo+16];
		l[lo+20] = l[lo+20] - coeff*l[lo+21];
		r[ro] = r[ro] - coeff*r[ro+1];
		coeff = l[lo+7];
		l[lo+12] = l[lo+12] - coeff*l[lo+11];
		l[lo+17] = l[lo+17] - coeff*l[lo+16];
		l[lo+22] = l[lo+22] - coeff*l[lo+21];
		r[ro+2] = r[ro+2] - coeff*r[ro+1];
		coeff = l[lo+8];
		l[lo+13] = l[lo+13] - coeff*l[lo+11];
		l[lo+18] = l[lo+18] - coeff*l[lo+16];
		l[lo+23] = l[lo+23] - coeff*l[lo+21];
		r[ro+3] = r[ro+3] - coeff*r[ro+1];
		coeff = l[lo+9];
		l[lo+14] = l[lo+14] - coeff*l[lo+11];
		l[lo+19] = l[lo+19] - coeff*l[lo+16];
		l[lo+24] = l[lo+24] - coeff*l[lo+21];
		r[ro+4] = r[ro+4] - coeff*r[ro+1];
		pivot = 1.00/l[lo+12];
		l[lo+17] = l[lo+17]*pivot;
		l[lo+22] = l[lo+22]*pivot;
		r[ro+2] = r[ro+2]*pivot;
		coeff = l[lo+10];
		l[lo+15] = l[lo+15] - coeff*l[lo+17];
		l[lo+20] = l[lo+20] - coeff*l[lo+22];
		r[ro] = r[ro] - coeff*r[ro+2];
		coeff = l[lo+11];
		l[lo+16] = l[lo+16] - coeff*l[lo+17];
		l[lo+21] = l[lo+21] - coeff*l[lo+22];
		r[ro+1] = r[ro+1] - coeff*r[ro+2];
		coeff = l[lo+13];
		l[lo+18] = l[lo+18] - coeff*l[lo+17];
		l[lo+23] = l[lo+23] - coeff*l[lo+22];
		r[ro+3] = r[ro+3] - coeff*r[ro+2];
		coeff = l[lo+14];
		l[lo+19] = l[lo+19] - coeff*l[lo+17];
		l[lo+24] = l[lo+24] - coeff*l[lo+22];
		r[ro+4] = r[ro+4] - coeff*r[ro+2];
		pivot = 1.00/l[lo+18];
		l[lo+23] = l[lo+23]*pivot;
		r[ro+3] = r[ro+3]*pivot;
		coeff = l[lo+15];
		l[lo+20] = l[lo+20] - coeff*l[lo+23];
		r[ro] = r[ro] - coeff*r[ro+3];
		coeff = l[lo+16];
		l[lo+21] = l[lo+21] - coeff*l[lo+23];
		r[ro+1] = r[ro+1] - coeff*r[ro+3];
		coeff = l[lo+17];
		l[lo+22] = l[lo+22] - coeff*l[lo+23];
		r[ro+2] = r[ro+2] - coeff*r[ro+3];
		coeff = l[lo+19];
		l[lo+24] = l[lo+24] - coeff*l[lo+23];
		r[ro+4] = r[ro+4] - coeff*r[ro+3];
		pivot = 1.00/l[lo+24];
		r[ro+4] = r[ro+4]*pivot;
		coeff = l[lo+20];
		r[ro] = r[ro] - coeff*r[ro+4];
		coeff = l[lo+21];
		r[ro+1] = r[ro+1] - coeff*r[ro+4];
		coeff = l[lo+22];
		r[ro+2] = r[ro+2] - coeff*r[ro+4];
		coeff = l[lo+23];
		r[ro+3] = r[ro+3] - coeff*r[ro+4];
	}

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
			dtref = 1.0e-2;
			xcrref[0] = 1.7034283709541311e-01;
			xcrref[1] = 1.2975252070034097e-02;
			xcrref[2] = 3.2527926989486055e-02;
			xcrref[3] = 2.6436421275166801e-02;
			xcrref[4] = 1.9211784131744430e-01;
			xceref[0] = 4.9976913345811579e-04;
			xceref[1] = 4.5195666782961927e-05;
			xceref[2] = 7.3973765172921357e-05;
			xceref[3] = 7.3821238632439731e-05;
			xceref[4] = 8.9269630987491446e-04;
		}
		else if(CLASS == 'W'){
			dtref = 0.8e-3;
			xcrref[0] = 0.1125590409344e+03;
			xcrref[1] = 0.1180007595731e+02;
			xcrref[2] = 0.2710329767846e+02;
			xcrref[3] = 0.2469174937669e+02;
			xcrref[4] = 0.2638427874317e+03;
			xceref[0] = 0.4419655736008e+01;
			xceref[1] = 0.4638531260002e+00;
			xceref[2] = 0.1011551749967e+01;
			xceref[3] = 0.9235878729944e+00;
			xceref[4] = 0.1018045837718e+02;
		}
		else if(CLASS == 'A'){
			dtref = 0.8e-3;
			xcrref[0] = 1.0806346714637264e+02;
			xcrref[1] = 1.1319730901220813e+01;
			xcrref[2] = 2.5974354511582465e+01;
			xcrref[3] = 2.3665622544678910e+01;
			xcrref[4] = 2.5278963211748344e+02;
			xceref[0] = 4.2348416040525025e+00;
			xceref[1] = 4.4390282496995698e-01;
			xceref[2] = 9.6692480136345650e-01;
			xceref[3] = 8.8302063039765474e-01;
			xceref[4] = 9.7379901770829278e+00;
		}
		else if(CLASS == 'B'){
			dtref = 3.0e-4;
			xcrref[0] = 1.4233597229287254e+03;
			xcrref[1] = 9.9330522590150238e+01;
			xcrref[2] = 3.5646025644535285e+02;
			xcrref[3] = 3.2485447959084092e+02;
			xcrref[4] = 3.2707541254659363e+03;
			xceref[0] = 5.2969847140936856e+01;
			xceref[1] = 4.4632896115670668e+00;
			xceref[2] = 1.3122573342210174e+01;
			xceref[3] = 1.2006925323559144e+01;
			xceref[4] = 1.2459576151035986e+02;
		}
		else if(CLASS == 'C'){
			dtref = 1.0e-4;
			xcrref[0] = 0.62398116551764615e+04;
			xcrref[1] = 0.50793239190423964e+03;
			xcrref[2] = 0.15423530093013596e+04;
			xcrref[3] = 0.13302387929291190e+04;
			xcrref[4] = 0.11604087428436455e+05;
			xceref[0] = 0.16462008369091265e+03;
			xceref[1] = 0.11497107903824313e+02;
			xceref[2] = 0.41207446207461508e+02;
			xceref[3] = 0.37087651059694167e+02;
			xceref[4] = 0.36211053051841265e+03;
		}
		else if(CLASS == 'D'){
			dtref = 0.2e-4;
			xcrref[0] = 0.2533188551738e+05;
			xcrref[1] = 0.2346393716980e+04;
			xcrref[2] = 0.6294554366904e+04;
			xcrref[3] = 0.5352565376030e+04;
			xcrref[4] = 0.3905864038618e+05;
			xceref[0] = 0.3100009377557e+03;
			xceref[1] = 0.2424086324913e+02;
			xceref[2] = 0.7782212022645e+02;
			xceref[3] = 0.6835623860116e+02;
			xceref[4] = 0.6065737200368e+03;
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
				"redist1", "solsubs", "add"};
		for(int i = 1; i<=T_LAST; i++){
			t = timer.read(i);
			System.out.format(java.util.Locale.ENGLISH, "  %-8s:%9.3f  (%6.2f%%)\n", names[i-1], t, t*100./tmax);
			if(i == T_RHS){
				t = timer.read(T_RHSX) + timer.read(T_RHSY) + timer.read(T_RHSZ);
				System.out.format(java.util.Locale.ENGLISH, "    --> %-8s:%9.3f  (%6.2f%%)\n", "sub-rhs", t, t*100./tmax);
				t = timer.read(T_RHS) - t;
				System.out.format(java.util.Locale.ENGLISH, "    --> %-8s:%9.3f  (%6.2f%%)\n", "rest-rhs", t, t*100./tmax);
			}
			else if(i == T_SOLSUB){
				t = timer.read(T_XSOLVE) + timer.read(T_YSOLVE) + timer.read(T_ZSOLVE)
						- timer.read(T_RDIS1) - timer.read(T_SOLSUB);
				System.out.format(java.util.Locale.ENGLISH, "    --> %-8s:%9.3f  (%6.2f%%)\n", "rest-sol", t, t*100./tmax);
			}
		}
	}
}

