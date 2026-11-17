/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB 3.4.4 MPI (NPB3.4-MPI/LU)
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
import NPB.common.timer.Timer;
import NPB.exceptions.BadClassException;
import NPB.exceptions.BadInputException;
import NPB.exceptions.IncorrectNumProcessException;
import NPB.exceptions.NPBException;

public class LU implements Benchmark{

	private static final String BM_NAME="LU";
	private char CLASS;
	private boolean detailedTiming;

	/* compiled values of the class (isiz01 = isiz02 = isiz03, itmax_default, inorm_default, dt_default) */
	private int problemSize;
	private int itmaxDefault;
	private double dtDefault;

	private int ipr, inorm;
	private int itmax;
	private double dt, omega;
	private double tolrsd[] = new double[5];

	/* full problem, and the sub-domain of this process */
	private int nx0, ny0, nz0;
	private int nx, ny, nz;
	private int ipt, jpt;
	private int ist, iend, jst, jend;
	private int ii1, ii2, ji1, ji2, ki1, ki2;

	/* sizes of the arrays (the same on every process) */
	private int isiz1, isiz2, isiz3;
	private int ldx, ldy;

	/*
	 * Field arrays. As in the original (u(5,-1:isiz1+2,-1:isiz2+2,isiz3)), the 5 components of
	 * a point are contiguous: u(m,i,j,k), with the indices of the original, is
	 * u[ix(i,j,k)+m-1] (ix already includes the factor 5). flux has the same layout as the
	 * others (the original declares it with 0:isiz1+1, 0:isiz2+1).
	 */
	private double u[];
	private double rsd[];
	private double frct[];
	private double flux[];

	/* jacobian blocks of the point i: a(5,5,i) is a[25*(i-1)+(m-1)+5*(n-1)] (first index fastest) */
	private double a[], b[], c[], d[];

	/* working arrays of the surface integral */
	private double phi1[][], phi2[][];

	/* message buffers: buf(5,*) and buf1(5,*), and dum of the exchanges 4, 5 and 6 */
	private double buf[], buf1[], dum[];

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
	private double maxcomm;

	private BMResults res;
	private Timer timer;

	/*
	 * Different timers
	 */
	private final int T_COMP = 0;
	private final int T_COMM = 1;

	/*
	 * MPJ variables
	 */
	private int id; /* rank in the communicator of the active processes */
	private int numProcess; /* active processes (num, no_nodes) */
	private int npTotal; /* processes launched (total_nodes) */
	private boolean active = true; /* false for the excess processes masked out */
	private Intracomm commWork; /* communicator of the active processes */

	/* process grid: xdim x ydim, position (row, col) of this process and its neighbors (-1: none) */
	private int xdim, ydim, row, col;
	private int north, south, east, west;

	private static final int FROM_S = 1, FROM_N = 2, FROM_E = 3, FROM_W = 4;

	public static void main(String[] args) throws NPBException {
		LU lu = null;

		BMArgs.parseCmdLineArgs(args, BM_NAME);
		try {
			lu = new LU(BMArgs.CLASS, BMArgs.DETAILED_TIMING);
		} catch (OutOfMemoryError e) {
			BMArgs.outOfMemoryMessage();
			throw e;
		}
		lu.runBenchmark(args);
	}

	public BMResults getResults(){
		return res;
	}

	public LU(char classSize, boolean dtiming) {

		CLASS = classSize;
		detailedTiming = dtiming;
	}

	public void build(String[] args) throws NPBException{

		MPI.Init(args);

		/* Active processes (NPB-MPI 3.4: get_active_nprocs with the grid topology, NPB_NPROCS_STRICT) */
		ActiveProcs ap;
		try {
			ap = ActiveProcs.select(ActiveProcs.GRID);
		} catch (IncorrectNumProcessException e) {
			terminate();
			throw e;
		}
		id = ap.rank;
		npTotal = ap.total;
		numProcess = ap.active;
		commWork = ap.comm;
		active = ap.isActive;
		if (!active)
			return; /* excess process masked out: runBenchmark() finalizes MPI */

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
		case 'E':
			problemSize = 1020;
			itmaxDefault = 300;
			dtDefault = 0.5;
			break;
		default:
			terminate();
			throw new BadClassException("Specified class (" + CLASS + ") not valid for this benchmark\n"
					+ "Valid classes are S, W, A, B, C, D and E");
		}
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

		/* read the input data and broadcast it */
		readInput();

		/* The class of the problem actually run (set_class of the original) */
		char runClass = CLASS;
		if(nx0 != problemSize || ny0 != problemSize || nz0 != problemSize || itmax != itmaxDefault)
			runClass = 'U';

		if(id == 0){
			System.out.println(" Size:  "+nx0+"x "+ny0+"x "+nz0+"  (class "+runClass+")");
			System.out.println(" Iterations:  "+itmax);
			System.out.println(" Total number of processes:  "+npTotal);
			if (numProcess != npTotal)
				System.out.println(" WARNING: Number of processes is not in a form of (n1*n2, n1/n2 <= 2).\n"
						+ " Number of active processes: "+numProcess);
			System.out.println();
		}

		/* set up the processor grid, the arrays, the neighbors and the sub-domain sizes */
		procGrid();

		allocSpace();

		neighbors();

		subdomain();

		setcoeff();

		setbv();

		setiv();

		erhs();

		/* One SSOR iteration to touch all data and program pages */
		ssor(1);

		/* reset the boundary and initial values */
		setbv();

		setiv();

		/* the SSOR iterations */
		ssor(itmax);

		error();

		pintgr();

		int verified = 0;
		if(id == 0)
			verified = verify(runClass);

		if(id == 0){
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

			double times[] = {maxtime, maxcomm};

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
					npTotal,
					numProcess,
					detailedTiming);

			res.printMPJ();
		}

		terminate();
	}

	/*
	 * Reads the optional input file inputlu.data (only the root process, as in the
	 * original) or takes the compiled defaults of the class, and broadcasts it. The
	 * file alternates two comment lines with one line of values: ipr and inorm,
	 * itmax, dt, omega, the five tolerances and the grid size.
	 */
	private void readInput(){

		boolean inputData = true;

		if(id == 0){

			BMArgs.bannerMPJ(BM_NAME, CLASS, npTotal);

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
				commWork.Abort(-7);
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
				commWork.Abort(1);
			}
			if(nx0 > problemSize || ny0 > problemSize || nz0 > problemSize){
				System.err.println("     PROBLEM SIZE IS TOO LARGE - \n"
						+ "     NX, NY AND NZ SHOULD BE LESS THAN OR EQUAL TO \n"
						+ "     ISIZ01, ISIZ02 AND ISIZ03 RESPECTIVELY");
				commWork.Abort(1);
			}
		}

		/* the root broadcasts the data (bcast_inputs) */
		int ibuff[] = {ipr, inorm, itmax, nx0, ny0, nz0};
		double dbuff[] = {dt, omega, tolrsd[0], tolrsd[1], tolrsd[2], tolrsd[3], tolrsd[4]};

		commWork.Bcast(ibuff, 0, 6, MPI.INT, 0);
		commWork.Bcast(dbuff, 0, 7, MPI.DOUBLE, 0);

		ipr = ibuff[0];
		inorm = ibuff[1];
		itmax = ibuff[2];
		nx0 = ibuff[3];
		ny0 = ibuff[4];
		nz0 = ibuff[5];
		dt = dbuff[0];
		omega = dbuff[1];
		for(int m = 0; m<5; m++)
			tolrsd[m] = dbuff[2+m];
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

	/*
	 * Largest xdim such that xdim*ydim = i with ydim <= xdim <= 2*ydim, or -1
	 * if there is none (isqrt2 of proc_grid.f90).
	 */
	private static int isqrt2(int i){

		int xd = -1;
		if(i <= 0)
			return xd;

		int square = 0;
		xd = 1;
		while(square <= i){
			square = xd*xd;
			if(square == i)
				return xd;
			xd = xd + 1;
		}

		xd = xd - 1;
		int yd = i / xd;
		while(xd*yd != i && 2*yd >= xd){
			xd = xd + 1;
			yd = i / xd;
		}

		if(xd*yd != i || 2*yd < xd)
			xd = -1;

		return xd;
	}

	/* Sub-domain array sizes and two-dimensional grid of processes (proc_grid) */
	private void procGrid(){

		int xdiv = isqrt2(numProcess);

		if(xdiv <= 0){
			if(id == 0)
				System.out.println(" ERROR: could not determine proper proc_grid for nprocs="+numProcess);
			commWork.Abort(1);
		}

		int ydiv = numProcess/xdiv;
		isiz1 = problemSize/xdiv;
		if(isiz1*xdiv < problemSize)
			isiz1 = isiz1 + 1;
		isiz2 = problemSize/ydiv;
		if(isiz2*ydiv < problemSize)
			isiz2 = isiz2 + 1;
		isiz3 = problemSize;

		/* a two-d grid of processors: column-major ordering of unknowns */
		int xdim0 = xdiv;
		int ydim0 = numProcess/xdim0;

		ydim = (int)(Math.sqrt((double)numProcess) + 0.001);
		xdim = numProcess/ydim;
		while(ydim >= ydim0 && xdim*ydim != numProcess){
			ydim = ydim - 1;
			xdim = numProcess/ydim;
		}

		if(xdim < xdim0 || ydim < ydim0 || xdim*ydim != numProcess){
			if(id == 0)
				System.out.println(" ERROR: could not determine proper proc_grid for nprocs="+numProcess);
			commWork.Abort(1);
		}

		/* nodedim of the original: exponent of the power of two (numProcess = 2**ndim) */
		int ndim = (int)(Math.log((double)numProcess)/Math.log(2.0) + 0.00001);
		if(id == 0 && numProcess != (1 << ndim))
			System.out.println(" Proc_grid for nprocs ="+numProcess+":"+xdim+" x"+ydim+"\n");

		row = id % xdim + 1;
		col = id / xdim + 1;
	}

	/* The neighbors of this process (-1: none) */
	private void neighbors(){

		south = -1;
		east = -1;
		north = -1;
		west = -1;

		if(row > 1)
			north = id - 1;
		else
			north = -1;

		if(row < xdim)
			south = id + 1;
		else
			south = -1;

		if(col > 1)
			west = id - xdim;
		else
			west = -1;

		if(col < ydim)
			east = id + xdim;
		else
			east = -1;
	}

	/* The sub-domain of this process */
	private void subdomain(){

		/* x dimension */
		int mm = nx0 % xdim;
		if(row <= mm){
			nx = nx0/xdim + 1;
			ipt = (row-1)*nx;
		}
		else{
			nx = nx0/xdim;
			ipt = (row-1)*nx + mm;
		}

		/* y dimension */
		mm = ny0 % ydim;
		if(col <= mm){
			ny = ny0/ydim + 1;
			jpt = (col-1)*ny;
		}
		else{
			ny = ny0/ydim;
			jpt = (col-1)*ny + mm;
		}

		/* z dimension */
		nz = nz0;

		/* check the sub-domain size */
		if(nx < 3 || ny < 3 || nz < 3){
			System.err.println("     SUBDOMAIN SIZE IS TOO SMALL - \n"
					+ "     ADJUST PROBLEM SIZE OR NUMBER OF PROCESSORS\n"
					+ "     SO THAT NX, NY AND NZ ARE GREATER THAN OR EQUAL\n"
					+ "     TO 3 THEY ARE CURRENTLY "+nx+" "+ny+" "+nz);
			commWork.Abort(1);
		}

		if(nx > isiz1 || ny > isiz2 || nz > isiz3){
			System.err.println("     SUBDOMAIN SIZE IS TOO LARGE - \n"
					+ "     ADJUST PROBLEM SIZE OR NUMBER OF PROCESSORS\n"
					+ "     SO THAT NX, NY AND NZ ARE LESS THAN OR EQUAL TO \n"
					+ "     ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY.  THEY ARE\n"
					+ "     CURRENTLY "+nx+" "+ny+" "+nz);
			commWork.Abort(1);
		}

		/* the start and end in i and j extents */
		ist = 1;
		iend = nx;
		if(north == -1)
			ist = 2;
		if(south == -1)
			iend = nx - 1;

		jst = 1;
		jend = ny;
		if(west == -1)
			jst = 2;
		if(east == -1)
			jend = ny - 1;
	}

	/* Index (already multiplied by 5) of the point (i,j,k) of u, rsd, frct and flux: i, j from -1, k from 1 */
	private int ix(int i, int j, int k){
		return 5*((i+1) + ldx*((j+1) + ldy*(k-1)));
	}

	private static double sq(double x){
		return x*x;
	}

	private void allocSpace() throws NPBException{

		ldx = isiz1 + 4;
		ldy = isiz2 + 4;

		long points = (long)ldx*ldy*isiz3;
		if(5*points > Integer.MAX_VALUE)
			throw new BadInputException("Problem size too big for Java arrays: use more processes");

		int n = (int)points;

		u = new double[5*n];
		rsd = new double[5*n];
		frct = new double[5*n];
		flux = new double[5*n];

		a = new double[25*isiz1];
		b = new double[25*isiz1];
		c = new double[25*isiz1];
		d = new double[25*isiz1];

		int nphi = Math.max(isiz1, Math.max(isiz2, isiz3)) + 3;
		phi1 = new double[nphi][nphi];
		phi2 = new double[nphi][nphi];

		/* buf(5,2*isiz2*isiz3) of the original; here also valid if isiz1 > isiz2 */
		int nbuf = 10*Math.max(isiz1, isiz2)*isiz3;
		buf = new double[nbuf];
		buf1 = new double[nbuf];
		dum = new double[2*nphi + 4];
	}

	/* The coefficients */
	private void setcoeff(){

		ii1 = 2;
		ii2 = nx0 - 1;
		ji1 = 2;
		ji2 = ny0 - 2;
		ki1 = 3;
		ki2 = nz0 - 1;

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

	/* The exact solution at the point (i,j,k) (1-based), stored from u000ijk[off] */
	private void exact(int i, int j, int k, double u000ijk[], int off){

		double xi, eta, zeta;

		xi   = ( (double)( i - 1 ) ) / ( nx0 - 1 );
		eta  = ( (double)( j - 1 ) ) / ( ny0 - 1 );
		zeta = ( (double)( k - 1 ) ) / ( nz - 1 );

		for(int m = 0; m<5; m++){
			u000ijk[off+m] =  ce[m][0]
				+ ce[m][1] * xi
				+ ce[m][2] * eta
				+ ce[m][3] * zeta
				+ ce[m][4] * xi * xi
				+ ce[m][5] * eta * eta
				+ ce[m][6] * zeta * zeta
				+ ce[m][7] * xi * xi * xi
				+ ce[m][8] * eta * eta * eta
				+ ce[m][9] * zeta * zeta * zeta
				+ ce[m][10] * xi * xi * xi * xi
				+ ce[m][11] * eta * eta * eta * eta
				+ ce[m][12] * zeta * zeta * zeta * zeta;
		}
	}

	/* Boundary values of u (the exact solution on the faces of the whole domain) */
	private void setbv(){
		int i, j, k;
		int iglob, jglob;

		/* the top and bottom faces */
		for (j = 1; j <= ny; j++){
			jglob = jpt + j;
			for (i = 1; i <= nx; i++){
				iglob = ipt + i;
				exact( iglob, jglob, 1, u, ix(i,j,1) );
				exact( iglob, jglob, nz, u, ix(i,j,nz) );
			}
		}

		/* the north and south faces (west and east neighbors: none) */
		if (west == -1){
			for (k = 1; k <= nz; k++){
				for (i = 1; i <= nx; i++){
					iglob = ipt + i;
					exact( iglob, 1, k, u, ix(i,1,k) );
				}
			}
		}

		if (east == -1){
			for (k = 1; k <= nz; k++){
				for (i = 1; i <= nx; i++){
					iglob = ipt + i;
					exact( iglob, ny0, k, u, ix(i,ny,k) );
				}
			}
		}

		/* the east and west faces */
		if (north == -1){
			for (k = 1; k <= nz; k++){
				for (j = 1; j <= ny; j++){
					jglob = jpt + j;
					exact( 1, jglob, k, u, ix(1,j,k) );
				}
			}
		}

		if (south == -1){
			for (k = 1; k <= nz; k++){
				for (j = 1; j <= ny; j++){
					jglob = jpt + j;
					exact( nx0, jglob, k, u, ix(nx,j,k) );
				}
			}
		}
	}

	/* Initial values of u: transfinite interpolation of the boundary values */
	private void setiv(){
		int i, j, k, m;
		int iglob, jglob;
		double eta, peta, pxi, pzeta, xi, zeta;
		double ue_1jk[] = new double[5], ue_nx0jk[] = new double[5], ue_i1k[] = new double[5],
				ue_iny0k[] = new double[5], ue_ij1[] = new double[5], ue_ijnz[] = new double[5];
		for (k = 2; k <= nz - 1; k++){
			zeta = ( (double)(k-1) ) / (nz-1);
			for (j = 1; j <= ny; j++){
				jglob = jpt + j;
				if (jglob != 1 && jglob != ny0){
					eta = ( (double)(jglob-1) ) / (ny0-1);
					for (i = 1; i <= nx; i++){
						iglob = ipt + i;
						if (iglob != 1 && iglob != nx0){
							xi = ( (double)(iglob-1) ) / (nx0-1);
							exact(1,jglob,k,ue_1jk,0);
							exact(nx0,jglob,k,ue_nx0jk,0);
							exact(iglob,1,k,ue_i1k,0);
							exact(iglob,ny0,k,ue_iny0k,0);
							exact(iglob,jglob,1,ue_ij1,0);
							exact(iglob,jglob,nz,ue_ijnz,0);
							for (m = 0; m <= 4; m++){
								pxi = ( 1.0 - xi ) * ue_1jk[m] + xi * ue_nx0jk[m];
								peta = ( 1.0 - eta ) * ue_i1k[m] + eta * ue_iny0k[m];
								pzeta = ( 1.0 - zeta ) * ue_ij1[m] + zeta * ue_ijnz[m];
								u[ix(i,j,k)+m] = pxi + peta + pzeta - pxi * peta - peta * pzeta - pzeta * pxi + pxi * peta * pzeta;
							}
						}
					}
				}
			}
		}
	}

	/* The forcing function frct (and the exact solution in rsd) */
	private void erhs(){
		erhsInit();
		erhsX();
		erhsY();
		erhsZ();
	}

	/*
	 * forcing function: frct to zero and the exact solution in rsd
	 */
	private void erhsInit(){

		int i, j, k, m, iglob, jglob, iex, l1;
		int l2, ist1, iend1, jst1, jend1;
		double dsspm, xi, eta, zeta, q, u21, u31, u41;
		double tmp, u21i, u31i, u41i, u51i, u21j, u31j, u41j;
		double u51j, u21k, u31k, u41k, u51k, u21im1, u31im1, u41im1;
		double u51im1, u21jm1, u31jm1, u41jm1, u51jm1, u21km1, u31km1, u41km1;
		double u51km1;

		for (k = 1; k <= nz; k++){
			for (j = 1; j <= ny; j++){
				for (i = 1; i <= nx; i++){
					for (m = 1; m <= 5; m++){
						frct[ix(i, j, k) + (m - 1)] = 0.0e+00;
					}
				}
			}
		}
		for (k = 1; k <= nz; k++){
			zeta = ((double)(k - 1)) / (nz - 1);
			for (j = 1; j <= ny; j++){
				jglob = jpt + j;
				eta = ((double)(jglob - 1)) / (ny0 - 1);
				for (i = 1; i <= nx; i++){
					iglob = ipt + i;
					xi = ((double)(iglob - 1)) / (nx0 - 1);
					for (m = 1; m <= 5; m++){
						rsd[ix(i, j, k) + (m - 1)] = ce[m - 1][0] + ce[m - 1][1] * xi + ce[m - 1][2] * eta + ce[m - 1][3] * zeta
								+ ce[m - 1][4] * xi * xi + ce[m - 1][5] * eta * eta + ce[m - 1][6] * zeta * zeta + ce[m - 1][7] * xi * xi * xi
								+ ce[m - 1][8] * eta * eta * eta + ce[m - 1][9] * zeta * zeta * zeta + ce[m - 1][10] * xi * xi * xi * xi
								+ ce[m - 1][11] * eta * eta * eta * eta + ce[m - 1][12] * zeta * zeta * zeta * zeta;
					}
				}
			}
		}
	}

	/*
	 * forcing function: x direction
	 */
	private void erhsX(){

		int i, j, k, m, iglob, jglob, iex, l1;
		int l2, ist1, iend1, jst1, jend1;
		double dsspm, xi, eta, zeta, q, u21, u31, u41;
		double tmp, u21i, u31i, u41i, u51i, u21j, u31j, u41j;
		double u51j, u21k, u31k, u41k, u51k, u21im1, u31im1, u41im1;
		double u51im1, u21jm1, u31jm1, u41jm1, u51jm1, u21km1, u31km1, u41km1;
		double u51km1;

		dsspm = dssp;
		iex = 0;
		exchange3(rsd, iex);
		l1 = 0;
		if (north == -1) l1 = 1;
		l2 = nx + 1;
		if (south == -1) l2 = nx;
		ist1 = 1;
		iend1 = nx;
		if (north == -1) ist1 = 4;
		if (south == -1) iend1 = nx - 3;
		for (k = 2; k <= nz - 1; k++){
			for (j = jst; j <= jend; j++){
				for (i = l1; i <= l2; i++){
					flux[ix(i, j, k)] = rsd[ix(i, j, k) + 1];
					u21 = rsd[ix(i, j, k) + 1] / rsd[ix(i, j, k)];
					q = 0.50e+00 * (rsd[ix(i, j, k) + 1] * rsd[ix(i, j, k) + 1] + rsd[ix(i, j, k) + 2] * rsd[ix(i, j, k) + 2]
							+ rsd[ix(i, j, k) + 3] * rsd[ix(i, j, k) + 3]) / rsd[ix(i, j, k)];
					flux[ix(i, j, k) + 1] = rsd[ix(i, j, k) + 1] * u21 + c2 * (rsd[ix(i, j, k) + 4] - q);
					flux[ix(i, j, k) + 2] = rsd[ix(i, j, k) + 2] * u21;
					flux[ix(i, j, k) + 3] = rsd[ix(i, j, k) + 3] * u21;
					flux[ix(i, j, k) + 4] = (c1 * rsd[ix(i, j, k) + 4] - c2 * q) * u21;
				}
			}
		}
		for (k = 2; k <= nz - 1; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					for (m = 1; m <= 5; m++){
						frct[ix(i, j, k) + (m - 1)] = frct[ix(i, j, k) + (m - 1)] - tx2 * (flux[ix(i + 1, j, k) + (m - 1)]
								- flux[ix(i - 1, j, k) + (m - 1)]);
					}
				}
				for (i = ist; i <= l2; i++){
					tmp = 1.0e+00 / rsd[ix(i, j, k)];
					u21i = tmp * rsd[ix(i, j, k) + 1];
					u31i = tmp * rsd[ix(i, j, k) + 2];
					u41i = tmp * rsd[ix(i, j, k) + 3];
					u51i = tmp * rsd[ix(i, j, k) + 4];
					tmp = 1.0e+00 / rsd[ix(i - 1, j, k)];
					u21im1 = tmp * rsd[ix(i - 1, j, k) + 1];
					u31im1 = tmp * rsd[ix(i - 1, j, k) + 2];
					u41im1 = tmp * rsd[ix(i - 1, j, k) + 3];
					u51im1 = tmp * rsd[ix(i - 1, j, k) + 4];
					flux[ix(i, j, k) + 1] = (4.0e+00 / 3.0e+00) * tx3 * (u21i - u21im1);
					flux[ix(i, j, k) + 2] = tx3 * (u31i - u31im1);
					flux[ix(i, j, k) + 3] = tx3 * (u41i - u41im1);
					flux[ix(i, j, k) + 4] = 0.50e+00 * (1.0e+00 - c1 * c5) * tx3 * ((sq(u21i) + sq(u31i) + sq(u41i)) - (sq(u21im1)
							+ sq(u31im1) + sq(u41im1))) + (1.0e+00 / 6.0e+00) * tx3 * (sq(u21i) - sq(u21im1)) + c1 * c5 * tx3 * (u51i - u51im1);
				}
				for (i = ist; i <= iend; i++){
					frct[ix(i, j, k)] = frct[ix(i, j, k)] + dx1 * tx1 * (rsd[ix(i - 1, j, k)] - 2.0e+00 * rsd[ix(i, j, k)]
							+ rsd[ix(i + 1, j, k)]);
					frct[ix(i, j, k) + 1] = frct[ix(i, j, k) + 1] + tx3 * c3 * c4 * (flux[ix(i + 1, j, k) + 1]
							- flux[ix(i, j, k) + 1]) + dx2 * tx1 * (rsd[ix(i - 1, j, k) + 1] - 2.0e+00 * rsd[ix(i, j, k) + 1]
							+ rsd[ix(i + 1, j, k) + 1]);
					frct[ix(i, j, k) + 2] = frct[ix(i, j, k) + 2] + tx3 * c3 * c4 * (flux[ix(i + 1, j, k) + 2]
							- flux[ix(i, j, k) + 2]) + dx3 * tx1 * (rsd[ix(i - 1, j, k) + 2] - 2.0e+00 * rsd[ix(i, j, k) + 2]
							+ rsd[ix(i + 1, j, k) + 2]);
					frct[ix(i, j, k) + 3] = frct[ix(i, j, k) + 3] + tx3 * c3 * c4 * (flux[ix(i + 1, j, k) + 3]
							- flux[ix(i, j, k) + 3]) + dx4 * tx1 * (rsd[ix(i - 1, j, k) + 3] - 2.0e+00 * rsd[ix(i, j, k) + 3]
							+ rsd[ix(i + 1, j, k) + 3]);
					frct[ix(i, j, k) + 4] = frct[ix(i, j, k) + 4] + tx3 * c3 * c4 * (flux[ix(i + 1, j, k) + 4]
							- flux[ix(i, j, k) + 4]) + dx5 * tx1 * (rsd[ix(i - 1, j, k) + 4] - 2.0e+00 * rsd[ix(i, j, k) + 4]
							+ rsd[ix(i + 1, j, k) + 4]);
				}
				if (north == -1){
					for (m = 1; m <= 5; m++){
						frct[ix(2, j, k) + (m - 1)] = frct[ix(2, j, k) + (m - 1)] - dsspm * (5.0e+00 * rsd[ix(2, j, k) + (m - 1)]
								- 4.0e+00 * rsd[ix(3, j, k) + (m - 1)] + rsd[ix(4, j, k) + (m - 1)]);
						frct[ix(3, j, k) + (m - 1)] = frct[ix(3, j, k) + (m - 1)] - dsspm * (-(4.0e+00 * rsd[ix(2, j, k) + (m - 1)])
								+ 6.0e+00 * rsd[ix(3, j, k) + (m - 1)] - 4.0e+00 * rsd[ix(4, j, k) + (m - 1)] + rsd[ix(5, j, k) + (m - 1)]);
					}
				}
				for (i = ist1; i <= iend1; i++){
					for (m = 1; m <= 5; m++){
						frct[ix(i, j, k) + (m - 1)] = frct[ix(i, j, k) + (m - 1)] - dsspm * (rsd[ix(i - 2, j, k) + (m - 1)]
								- 4.0e+00 * rsd[ix(i - 1, j, k) + (m - 1)] + 6.0e+00 * rsd[ix(i, j, k) + (m - 1)]
								- 4.0e+00 * rsd[ix(i + 1, j, k) + (m - 1)] + rsd[ix(i + 2, j, k) + (m - 1)]);
					}
				}
				if (south == -1){
					for (m = 1; m <= 5; m++){
						frct[ix(nx - 2, j, k) + (m - 1)] = frct[ix(nx - 2, j, k) + (m - 1)] - dsspm * (rsd[ix(nx - 4, j, k) + (m - 1)]
								- 4.0e+00 * rsd[ix(nx - 3, j, k) + (m - 1)] + 6.0e+00 * rsd[ix(nx - 2, j, k) + (m - 1)]
								- 4.0e+00 * rsd[ix(nx - 1, j, k) + (m - 1)]);
						frct[ix(nx - 1, j, k) + (m - 1)] = frct[ix(nx - 1, j, k) + (m - 1)] - dsspm * (rsd[ix(nx - 3, j, k) + (m - 1)]
								- 4.0e+00 * rsd[ix(nx - 2, j, k) + (m - 1)] + 5.0e+00 * rsd[ix(nx - 1, j, k) + (m - 1)]);
					}
				}
			}
		}
	}

	/*
	 * forcing function: y direction
	 */
	private void erhsY(){

		int i, j, k, m, iglob, jglob, iex, l1;
		int l2, ist1, iend1, jst1, jend1;
		double dsspm, xi, eta, zeta, q, u21, u31, u41;
		double tmp, u21i, u31i, u41i, u51i, u21j, u31j, u41j;
		double u51j, u21k, u31k, u41k, u51k, u21im1, u31im1, u41im1;
		double u51im1, u21jm1, u31jm1, u41jm1, u51jm1, u21km1, u31km1, u41km1;
		double u51km1;

		dsspm = dssp;
		iex = 1;
		exchange3(rsd, iex);
		l1 = 0;
		if (west == -1) l1 = 1;
		l2 = ny + 1;
		if (east == -1) l2 = ny;
		jst1 = 1;
		jend1 = ny;
		if (west == -1) jst1 = 4;
		if (east == -1) jend1 = ny - 3;
		for (k = 2; k <= nz - 1; k++){
			for (j = l1; j <= l2; j++){
				for (i = ist; i <= iend; i++){
					flux[ix(i, j, k)] = rsd[ix(i, j, k) + 2];
					u31 = rsd[ix(i, j, k) + 2] / rsd[ix(i, j, k)];
					q = 0.50e+00 * (rsd[ix(i, j, k) + 1] * rsd[ix(i, j, k) + 1] + rsd[ix(i, j, k) + 2] * rsd[ix(i, j, k) + 2]
							+ rsd[ix(i, j, k) + 3] * rsd[ix(i, j, k) + 3]) / rsd[ix(i, j, k)];
					flux[ix(i, j, k) + 1] = rsd[ix(i, j, k) + 1] * u31;
					flux[ix(i, j, k) + 2] = rsd[ix(i, j, k) + 2] * u31 + c2 * (rsd[ix(i, j, k) + 4] - q);
					flux[ix(i, j, k) + 3] = rsd[ix(i, j, k) + 3] * u31;
					flux[ix(i, j, k) + 4] = (c1 * rsd[ix(i, j, k) + 4] - c2 * q) * u31;
				}
			}
		}
		for (k = 2; k <= nz - 1; k++){
			for (i = ist; i <= iend; i++){
				for (j = jst; j <= jend; j++){
					for (m = 1; m <= 5; m++){
						frct[ix(i, j, k) + (m - 1)] = frct[ix(i, j, k) + (m - 1)] - ty2 * (flux[ix(i, j + 1, k) + (m - 1)]
								- flux[ix(i, j - 1, k) + (m - 1)]);
					}
				}
			}
			for (j = jst; j <= l2; j++){
				for (i = ist; i <= iend; i++){
					tmp = 1.0e+00 / rsd[ix(i, j, k)];
					u21j = tmp * rsd[ix(i, j, k) + 1];
					u31j = tmp * rsd[ix(i, j, k) + 2];
					u41j = tmp * rsd[ix(i, j, k) + 3];
					u51j = tmp * rsd[ix(i, j, k) + 4];
					tmp = 1.0e+00 / rsd[ix(i, j - 1, k)];
					u21jm1 = tmp * rsd[ix(i, j - 1, k) + 1];
					u31jm1 = tmp * rsd[ix(i, j - 1, k) + 2];
					u41jm1 = tmp * rsd[ix(i, j - 1, k) + 3];
					u51jm1 = tmp * rsd[ix(i, j - 1, k) + 4];
					flux[ix(i, j, k) + 1] = ty3 * (u21j - u21jm1);
					flux[ix(i, j, k) + 2] = (4.0e+00 / 3.0e+00) * ty3 * (u31j - u31jm1);
					flux[ix(i, j, k) + 3] = ty3 * (u41j - u41jm1);
					flux[ix(i, j, k) + 4] = 0.50e+00 * (1.0e+00 - c1 * c5) * ty3 * ((sq(u21j) + sq(u31j) + sq(u41j)) - (sq(u21jm1)
							+ sq(u31jm1) + sq(u41jm1))) + (1.0e+00 / 6.0e+00) * ty3 * (sq(u31j) - sq(u31jm1)) + c1 * c5 * ty3 * (u51j - u51jm1);
				}
			}
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					frct[ix(i, j, k)] = frct[ix(i, j, k)] + dy1 * ty1 * (rsd[ix(i, j - 1, k)] - 2.0e+00 * rsd[ix(i, j, k)]
							+ rsd[ix(i, j + 1, k)]);
					frct[ix(i, j, k) + 1] = frct[ix(i, j, k) + 1] + ty3 * c3 * c4 * (flux[ix(i, j + 1, k) + 1]
							- flux[ix(i, j, k) + 1]) + dy2 * ty1 * (rsd[ix(i, j - 1, k) + 1] - 2.0e+00 * rsd[ix(i, j, k) + 1]
							+ rsd[ix(i, j + 1, k) + 1]);
					frct[ix(i, j, k) + 2] = frct[ix(i, j, k) + 2] + ty3 * c3 * c4 * (flux[ix(i, j + 1, k) + 2]
							- flux[ix(i, j, k) + 2]) + dy3 * ty1 * (rsd[ix(i, j - 1, k) + 2] - 2.0e+00 * rsd[ix(i, j, k) + 2]
							+ rsd[ix(i, j + 1, k) + 2]);
					frct[ix(i, j, k) + 3] = frct[ix(i, j, k) + 3] + ty3 * c3 * c4 * (flux[ix(i, j + 1, k) + 3]
							- flux[ix(i, j, k) + 3]) + dy4 * ty1 * (rsd[ix(i, j - 1, k) + 3] - 2.0e+00 * rsd[ix(i, j, k) + 3]
							+ rsd[ix(i, j + 1, k) + 3]);
					frct[ix(i, j, k) + 4] = frct[ix(i, j, k) + 4] + ty3 * c3 * c4 * (flux[ix(i, j + 1, k) + 4]
							- flux[ix(i, j, k) + 4]) + dy5 * ty1 * (rsd[ix(i, j - 1, k) + 4] - 2.0e+00 * rsd[ix(i, j, k) + 4]
							+ rsd[ix(i, j + 1, k) + 4]);
				}
			}
			if (west == -1){
				for (i = ist; i <= iend; i++){
					for (m = 1; m <= 5; m++){
						frct[ix(i, 2, k) + (m - 1)] = frct[ix(i, 2, k) + (m - 1)] - dsspm * (5.0e+00 * rsd[ix(i, 2, k) + (m - 1)]
								- 4.0e+00 * rsd[ix(i, 3, k) + (m - 1)] + rsd[ix(i, 4, k) + (m - 1)]);
						frct[ix(i, 3, k) + (m - 1)] = frct[ix(i, 3, k) + (m - 1)] - dsspm * (-(4.0e+00 * rsd[ix(i, 2, k) + (m - 1)])
								+ 6.0e+00 * rsd[ix(i, 3, k) + (m - 1)] - 4.0e+00 * rsd[ix(i, 4, k) + (m - 1)] + rsd[ix(i, 5, k) + (m - 1)]);
					}
				}
			}
			for (j = jst1; j <= jend1; j++){
				for (i = ist; i <= iend; i++){
					for (m = 1; m <= 5; m++){
						frct[ix(i, j, k) + (m - 1)] = frct[ix(i, j, k) + (m - 1)] - dsspm * (rsd[ix(i, j - 2, k) + (m - 1)]
								- 4.0e+00 * rsd[ix(i, j - 1, k) + (m - 1)] + 6.0e+00 * rsd[ix(i, j, k) + (m - 1)]
								- 4.0e+00 * rsd[ix(i, j + 1, k) + (m - 1)] + rsd[ix(i, j + 2, k) + (m - 1)]);
					}
				}
			}
			if (east == -1){
				for (i = ist; i <= iend; i++){
					for (m = 1; m <= 5; m++){
						frct[ix(i, ny - 2, k) + (m - 1)] = frct[ix(i, ny - 2, k) + (m - 1)] - dsspm * (rsd[ix(i, ny - 4, k) + (m - 1)]
								- 4.0e+00 * rsd[ix(i, ny - 3, k) + (m - 1)] + 6.0e+00 * rsd[ix(i, ny - 2, k) + (m - 1)]
								- 4.0e+00 * rsd[ix(i, ny - 1, k) + (m - 1)]);
						frct[ix(i, ny - 1, k) + (m - 1)] = frct[ix(i, ny - 1, k) + (m - 1)] - dsspm * (rsd[ix(i, ny - 3, k) + (m - 1)]
								- 4.0e+00 * rsd[ix(i, ny - 2, k) + (m - 1)] + 5.0e+00 * rsd[ix(i, ny - 1, k) + (m - 1)]);
					}
				}
			}
		}
	}

	/*
	 * forcing function: z direction
	 */
	private void erhsZ(){

		int i, j, k, m, iglob, jglob, iex, l1;
		int l2, ist1, iend1, jst1, jend1;
		double dsspm, xi, eta, zeta, q, u21, u31, u41;
		double tmp, u21i, u31i, u41i, u51i, u21j, u31j, u41j;
		double u51j, u21k, u31k, u41k, u51k, u21im1, u31im1, u41im1;
		double u51im1, u21jm1, u31jm1, u41jm1, u51jm1, u21km1, u31km1, u41km1;
		double u51km1;

		dsspm = dssp;
		for (k = 1; k <= nz; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					flux[ix(i, j, k)] = rsd[ix(i, j, k) + 3];
					u41 = rsd[ix(i, j, k) + 3] / rsd[ix(i, j, k)];
					q = 0.50e+00 * (rsd[ix(i, j, k) + 1] * rsd[ix(i, j, k) + 1] + rsd[ix(i, j, k) + 2] * rsd[ix(i, j, k) + 2]
							+ rsd[ix(i, j, k) + 3] * rsd[ix(i, j, k) + 3]) / rsd[ix(i, j, k)];
					flux[ix(i, j, k) + 1] = rsd[ix(i, j, k) + 1] * u41;
					flux[ix(i, j, k) + 2] = rsd[ix(i, j, k) + 2] * u41;
					flux[ix(i, j, k) + 3] = rsd[ix(i, j, k) + 3] * u41 + c2 * (rsd[ix(i, j, k) + 4] - q);
					flux[ix(i, j, k) + 4] = (c1 * rsd[ix(i, j, k) + 4] - c2 * q) * u41;
				}
			}
		}
		for (k = 2; k <= nz - 1; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					for (m = 1; m <= 5; m++){
						frct[ix(i, j, k) + (m - 1)] = frct[ix(i, j, k) + (m - 1)] - tz2 * (flux[ix(i, j, k + 1) + (m - 1)]
								- flux[ix(i, j, k - 1) + (m - 1)]);
					}
				}
			}
		}
		for (k = 2; k <= nz; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					tmp = 1.0e+00 / rsd[ix(i, j, k)];
					u21k = tmp * rsd[ix(i, j, k) + 1];
					u31k = tmp * rsd[ix(i, j, k) + 2];
					u41k = tmp * rsd[ix(i, j, k) + 3];
					u51k = tmp * rsd[ix(i, j, k) + 4];
					tmp = 1.0e+00 / rsd[ix(i, j, k - 1)];
					u21km1 = tmp * rsd[ix(i, j, k - 1) + 1];
					u31km1 = tmp * rsd[ix(i, j, k - 1) + 2];
					u41km1 = tmp * rsd[ix(i, j, k - 1) + 3];
					u51km1 = tmp * rsd[ix(i, j, k - 1) + 4];
					flux[ix(i, j, k) + 1] = tz3 * (u21k - u21km1);
					flux[ix(i, j, k) + 2] = tz3 * (u31k - u31km1);
					flux[ix(i, j, k) + 3] = (4.0e+00 / 3.0e+00) * tz3 * (u41k - u41km1);
					flux[ix(i, j, k) + 4] = 0.50e+00 * (1.0e+00 - c1 * c5) * tz3 * ((sq(u21k) + sq(u31k) + sq(u41k)) - (sq(u21km1)
							+ sq(u31km1) + sq(u41km1))) + (1.0e+00 / 6.0e+00) * tz3 * (sq(u41k) - sq(u41km1)) + c1 * c5 * tz3 * (u51k - u51km1);
				}
			}
		}
		for (k = 2; k <= nz - 1; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					frct[ix(i, j, k)] = frct[ix(i, j, k)] + dz1 * tz1 * (rsd[ix(i, j, k + 1)] - 2.0e+00 * rsd[ix(i, j, k)]
							+ rsd[ix(i, j, k - 1)]);
					frct[ix(i, j, k) + 1] = frct[ix(i, j, k) + 1] + tz3 * c3 * c4 * (flux[ix(i, j, k + 1) + 1]
							- flux[ix(i, j, k) + 1]) + dz2 * tz1 * (rsd[ix(i, j, k + 1) + 1] - 2.0e+00 * rsd[ix(i, j, k) + 1]
							+ rsd[ix(i, j, k - 1) + 1]);
					frct[ix(i, j, k) + 2] = frct[ix(i, j, k) + 2] + tz3 * c3 * c4 * (flux[ix(i, j, k + 1) + 2]
							- flux[ix(i, j, k) + 2]) + dz3 * tz1 * (rsd[ix(i, j, k + 1) + 2] - 2.0e+00 * rsd[ix(i, j, k) + 2]
							+ rsd[ix(i, j, k - 1) + 2]);
					frct[ix(i, j, k) + 3] = frct[ix(i, j, k) + 3] + tz3 * c3 * c4 * (flux[ix(i, j, k + 1) + 3]
							- flux[ix(i, j, k) + 3]) + dz4 * tz1 * (rsd[ix(i, j, k + 1) + 3] - 2.0e+00 * rsd[ix(i, j, k) + 3]
							+ rsd[ix(i, j, k - 1) + 3]);
					frct[ix(i, j, k) + 4] = frct[ix(i, j, k) + 4] + tz3 * c3 * c4 * (flux[ix(i, j, k + 1) + 4]
							- flux[ix(i, j, k) + 4]) + dz5 * tz1 * (rsd[ix(i, j, k + 1) + 4] - 2.0e+00 * rsd[ix(i, j, k) + 4]
							+ rsd[ix(i, j, k - 1) + 4]);
				}
			}
		}
		for (j = jst; j <= jend; j++){
			for (i = ist; i <= iend; i++){
				for (m = 1; m <= 5; m++){
					frct[ix(i, j, 2) + (m - 1)] = frct[ix(i, j, 2) + (m - 1)] - dsspm * (5.0e+00 * rsd[ix(i, j, 2) + (m - 1)]
							- 4.0e+00 * rsd[ix(i, j, 3) + (m - 1)] + rsd[ix(i, j, 4) + (m - 1)]);
					frct[ix(i, j, 3) + (m - 1)] = frct[ix(i, j, 3) + (m - 1)] - dsspm * (-(4.0e+00 * rsd[ix(i, j, 2) + (m - 1)])
							+ 6.0e+00 * rsd[ix(i, j, 3) + (m - 1)] - 4.0e+00 * rsd[ix(i, j, 4) + (m - 1)] + rsd[ix(i, j, 5) + (m - 1)]);
				}
			}
		}
		for (k = 4; k <= nz - 3; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					for (m = 1; m <= 5; m++){
						frct[ix(i, j, k) + (m - 1)] = frct[ix(i, j, k) + (m - 1)] - dsspm * (rsd[ix(i, j, k - 2) + (m - 1)]
								- 4.0e+00 * rsd[ix(i, j, k - 1) + (m - 1)] + 6.0e+00 * rsd[ix(i, j, k) + (m - 1)]
								- 4.0e+00 * rsd[ix(i, j, k + 1) + (m - 1)] + rsd[ix(i, j, k + 2) + (m - 1)]);
					}
				}
			}
		}
		for (j = jst; j <= jend; j++){
			for (i = ist; i <= iend; i++){
				for (m = 1; m <= 5; m++){
					frct[ix(i, j, nz - 2) + (m - 1)] = frct[ix(i, j, nz - 2) + (m - 1)] - dsspm * (rsd[ix(i, j, nz - 4) + (m - 1)]
							- 4.0e+00 * rsd[ix(i, j, nz - 3) + (m - 1)] + 6.0e+00 * rsd[ix(i, j, nz - 2) + (m - 1)]
							- 4.0e+00 * rsd[ix(i, j, nz - 1) + (m - 1)]);
					frct[ix(i, j, nz - 1) + (m - 1)] = frct[ix(i, j, nz - 1) + (m - 1)] - dsspm * (rsd[ix(i, j, nz - 3) + (m - 1)]
							- 4.0e+00 * rsd[ix(i, j, nz - 2) + (m - 1)] + 5.0e+00 * rsd[ix(i, j, nz - 1) + (m - 1)]);
				}
			}
		}
	}

	/* The steady-state residuals rsd */
	private void rhs(){
		rhsInit();
		rhsX();
		rhsY();
		rhsZ();
	}

	/*
	 * steady-state residuals: rsd = -frct
	 */
	private void rhsInit(){

		int i, j, k, m, iex, l1, l2, ist1;
		int iend1, jst1, jend1;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;

		for (k = 1; k <= nz; k++){
			for (j = 1; j <= ny; j++){
				for (i = 1; i <= nx; i++){
					for (m = 1; m <= 5; m++){
						rsd[ix(i, j, k) + (m - 1)] = -frct[ix(i, j, k) + (m - 1)];
					}
				}
			}
		}
	}

	/*
	 * Optimization (2026-11-16): rhsX/rhsY/rhsZ are split into one small method per loop nest (the 2.3-2.7 kB originals were recompiled
	 * several times and, in some compilations, C2 did not inline ix() in their ~100 call sites: ix() at ~71x the cycles from class B to C)
	 * and ix() is written out in place. Same operations and order: bit-identical results.
	 */
	/*
	 * steady-state residuals: x direction
	 */
	private void rhsX(){

		int i, j, k, m, iex, l1, l2, ist1 = 0;
		int iend1 = 0, jst1 = 0, jend1 = 0;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;

		iex = 0;
		exchange3(u, iex);
		l1 = 0;
		if (north == -1) l1 = 1;
		l2 = nx + 1;
		if (south == -1) l2 = nx;
		ist1 = 1;
		iend1 = nx;
		if (north == -1) ist1 = 4;
		if (south == -1) iend1 = nx - 3;
		for (k = 2; k <= nz - 1; k++){
			for (j = jst; j <= jend; j++){
				rhsX_0(j, k, l1, l2, ist1, iend1, jst1, jend1);
				rhsX_1(j, k, l1, l2, ist1, iend1, jst1, jend1);
				rhsX_2(j, k, l1, l2, ist1, iend1, jst1, jend1);
				rhsX_3(j, k, l1, l2, ist1, iend1, jst1, jend1);
				rhsX_4(j, k, l1, l2, ist1, iend1, jst1, jend1);
				rhsX_5(j, k, l1, l2, ist1, iend1, jst1, jend1);
				rhsX_6(j, k, l1, l2, ist1, iend1, jst1, jend1);
			}
		}
	}

	private void rhsX_0(int j, int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
				for (i = l1; i <= l2; i++){
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))] = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1];
					u21 = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] / u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))];
					q = 0.50e+00 * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] + u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2]
							+ u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3]) / u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))];
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] * u21 + c2 * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] - q);
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] * u21;
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] * u21;
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] = (c1 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] - c2 * q) * u21;
				}
	}

	private void rhsX_1(int j, int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
				for (i = ist; i <= iend; i++){
					for (m = 1; m <= 5; m++){
						rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - tx2 * (flux[(5*(((i + 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)]
								- flux[(5*(((i - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)]);
					}
				}
	}

	private void rhsX_2(int j, int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
				for (i = ist; i <= l2; i++){
					tmp = 1.0e+00 / u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))];
					u21i = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1];
					u31i = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2];
					u41i = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3];
					u51i = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4];
					tmp = 1.0e+00 / u[(5*(((i - 1)+1) + ldx*(((j)+1) + ldy*((k)-1))))];
					u21im1 = tmp * u[(5*(((i - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1];
					u31im1 = tmp * u[(5*(((i - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2];
					u41im1 = tmp * u[(5*(((i - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3];
					u51im1 = tmp * u[(5*(((i - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4];
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] = (4.0e+00 / 3.0e+00) * tx3 * (u21i - u21im1);
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] = tx3 * (u31i - u31im1);
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] = tx3 * (u41i - u41im1);
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] = 0.50e+00 * (1.0e+00 - c1 * c5) * tx3 * ((sq(u21i) + sq(u31i) + sq(u41i)) - (sq(u21im1)
							+ sq(u31im1) + sq(u41im1))) + (1.0e+00 / 6.0e+00) * tx3 * (sq(u21i) - sq(u21im1)) + c1 * c5 * tx3 * (u51i - u51im1);
				}
	}

	private void rhsX_3(int j, int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
				for (i = ist; i <= iend; i++){
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))] + dx1 * tx1 * (u[(5*(((i - 1)+1) + ldx*(((j)+1) + ldy*((k)-1))))] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))]
							+ u[(5*(((i + 1)+1) + ldx*(((j)+1) + ldy*((k)-1))))]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] + tx3 * c3 * c4 * (flux[(5*(((i + 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] - flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1])
							+ dx2 * tx1 * (u[(5*(((i - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] + u[(5*(((i + 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] + tx3 * c3 * c4 * (flux[(5*(((i + 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] - flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2])
							+ dx3 * tx1 * (u[(5*(((i - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] + u[(5*(((i + 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] + tx3 * c3 * c4 * (flux[(5*(((i + 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] - flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3])
							+ dx4 * tx1 * (u[(5*(((i - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] + u[(5*(((i + 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] + tx3 * c3 * c4 * (flux[(5*(((i + 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] - flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4])
							+ dx5 * tx1 * (u[(5*(((i - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] + u[(5*(((i + 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4]);
				}
	}

	private void rhsX_4(int j, int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
				if (north == -1){
					for (m = 1; m <= 5; m++){
						rsd[(5*(((2)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((2)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - dssp * (5.0e+00 * u[(5*(((2)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)]
								- 4.0e+00 * u[(5*(((3)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] + u[(5*(((4)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)]);
						rsd[(5*(((3)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((3)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - dssp * (-(4.0e+00 * u[(5*(((2)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)])
								+ 6.0e+00 * u[(5*(((3)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - 4.0e+00 * u[(5*(((4)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] + u[(5*(((5)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)]);
					}
				}
	}

	private void rhsX_5(int j, int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
				for (i = ist1; i <= iend1; i++){
					for (m = 1; m <= 5; m++){
						rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - dssp * (u[(5*(((i - 2)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)]
								- 4.0e+00 * u[(5*(((i - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] + 6.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - 4.0e+00 * u[(5*(((i + 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)]
								+ u[(5*(((i + 2)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)]);
					}
				}
	}

	private void rhsX_6(int j, int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
				if (south == -1){
					for (m = 1; m <= 5; m++){
						rsd[(5*(((nx - 2)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((nx - 2)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - dssp * (u[(5*(((nx - 4)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)]
								- 4.0e+00 * u[(5*(((nx - 3)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] + 6.0e+00 * u[(5*(((nx - 2)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)]
								- 4.0e+00 * u[(5*(((nx - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)]);
						rsd[(5*(((nx - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((nx - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - dssp * (u[(5*(((nx - 3)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)]
								- 4.0e+00 * u[(5*(((nx - 2)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] + 5.0e+00 * u[(5*(((nx - 1)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)]);
					}
				}
	}

	/*
	 * steady-state residuals: y direction
	 */
	private void rhsY(){

		int i, j, k, m, iex, l1, l2, ist1 = 0;
		int iend1 = 0, jst1 = 0, jend1 = 0;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;

		iex = 1;
		exchange3(u, iex);
		l1 = 0;
		if (west == -1) l1 = 1;
		l2 = ny + 1;
		if (east == -1) l2 = ny;
		jst1 = 1;
		jend1 = ny;
		if (west == -1) jst1 = 4;
		if (east == -1) jend1 = ny - 3;
		for (k = 2; k <= nz - 1; k++){
			rhsY_0(k, l1, l2, ist1, iend1, jst1, jend1);
			rhsY_1(k, l1, l2, ist1, iend1, jst1, jend1);
			rhsY_2(k, l1, l2, ist1, iend1, jst1, jend1);
			rhsY_3(k, l1, l2, ist1, iend1, jst1, jend1);
			rhsY_4(k, l1, l2, ist1, iend1, jst1, jend1);
			rhsY_5(k, l1, l2, ist1, iend1, jst1, jend1);
			rhsY_6(k, l1, l2, ist1, iend1, jst1, jend1);
		}
	}

	private void rhsY_0(int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, j, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
			for (j = l1; j <= l2; j++){
				for (i = ist; i <= iend; i++){
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))] = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2];
					u31 = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] / u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))];
					q = 0.50e+00 * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] + u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2]
							+ u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3]) / u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))];
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] * u31;
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] * u31 + c2 * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] - q);
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] * u31;
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] = (c1 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] - c2 * q) * u31;
				}
			}
	}

	private void rhsY_1(int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, j, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					for (m = 1; m <= 5; m++){
						rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - ty2 * (flux[(5*(((i)+1) + ldx*(((j + 1)+1) + ldy*((k)-1)))) + (m - 1)]
								- flux[(5*(((i)+1) + ldx*(((j - 1)+1) + ldy*((k)-1)))) + (m - 1)]);
					}
				}
			}
	}

	private void rhsY_2(int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, j, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
			for (j = jst; j <= l2; j++){
				for (i = ist; i <= iend; i++){
					tmp = 1.0e+00 / u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))];
					u21j = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1];
					u31j = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2];
					u41j = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3];
					u51j = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4];
					tmp = 1.0e+00 / u[(5*(((i)+1) + ldx*(((j - 1)+1) + ldy*((k)-1))))];
					u21jm1 = tmp * u[(5*(((i)+1) + ldx*(((j - 1)+1) + ldy*((k)-1)))) + 1];
					u31jm1 = tmp * u[(5*(((i)+1) + ldx*(((j - 1)+1) + ldy*((k)-1)))) + 2];
					u41jm1 = tmp * u[(5*(((i)+1) + ldx*(((j - 1)+1) + ldy*((k)-1)))) + 3];
					u51jm1 = tmp * u[(5*(((i)+1) + ldx*(((j - 1)+1) + ldy*((k)-1)))) + 4];
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] = ty3 * (u21j - u21jm1);
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] = (4.0e+00 / 3.0e+00) * ty3 * (u31j - u31jm1);
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] = ty3 * (u41j - u41jm1);
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] = 0.50e+00 * (1.0e+00 - c1 * c5) * ty3 * ((sq(u21j) + sq(u31j) + sq(u41j)) - (sq(u21jm1)
							+ sq(u31jm1) + sq(u41jm1))) + (1.0e+00 / 6.0e+00) * ty3 * (sq(u31j) - sq(u31jm1)) + c1 * c5 * ty3 * (u51j - u51jm1);
				}
			}
	}

	private void rhsY_3(int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, j, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))] + dy1 * ty1 * (u[(5*(((i)+1) + ldx*(((j - 1)+1) + ldy*((k)-1))))] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))]
							+ u[(5*(((i)+1) + ldx*(((j + 1)+1) + ldy*((k)-1))))]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] + ty3 * c3 * c4 * (flux[(5*(((i)+1) + ldx*(((j + 1)+1) + ldy*((k)-1)))) + 1] - flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1])
							+ dy2 * ty1 * (u[(5*(((i)+1) + ldx*(((j - 1)+1) + ldy*((k)-1)))) + 1] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] + u[(5*(((i)+1) + ldx*(((j + 1)+1) + ldy*((k)-1)))) + 1]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] + ty3 * c3 * c4 * (flux[(5*(((i)+1) + ldx*(((j + 1)+1) + ldy*((k)-1)))) + 2] - flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2])
							+ dy3 * ty1 * (u[(5*(((i)+1) + ldx*(((j - 1)+1) + ldy*((k)-1)))) + 2] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] + u[(5*(((i)+1) + ldx*(((j + 1)+1) + ldy*((k)-1)))) + 2]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] + ty3 * c3 * c4 * (flux[(5*(((i)+1) + ldx*(((j + 1)+1) + ldy*((k)-1)))) + 3] - flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3])
							+ dy4 * ty1 * (u[(5*(((i)+1) + ldx*(((j - 1)+1) + ldy*((k)-1)))) + 3] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] + u[(5*(((i)+1) + ldx*(((j + 1)+1) + ldy*((k)-1)))) + 3]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] + ty3 * c3 * c4 * (flux[(5*(((i)+1) + ldx*(((j + 1)+1) + ldy*((k)-1)))) + 4] - flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4])
							+ dy5 * ty1 * (u[(5*(((i)+1) + ldx*(((j - 1)+1) + ldy*((k)-1)))) + 4] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] + u[(5*(((i)+1) + ldx*(((j + 1)+1) + ldy*((k)-1)))) + 4]);
				}
			}
	}

	private void rhsY_4(int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, j, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
			if (west == -1){
				for (i = ist; i <= iend; i++){
					for (m = 1; m <= 5; m++){
						rsd[(5*(((i)+1) + ldx*(((2)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((2)+1) + ldy*((k)-1)))) + (m - 1)] - dssp * (5.0e+00 * u[(5*(((i)+1) + ldx*(((2)+1) + ldy*((k)-1)))) + (m - 1)]
								- 4.0e+00 * u[(5*(((i)+1) + ldx*(((3)+1) + ldy*((k)-1)))) + (m - 1)] + u[(5*(((i)+1) + ldx*(((4)+1) + ldy*((k)-1)))) + (m - 1)]);
						rsd[(5*(((i)+1) + ldx*(((3)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((3)+1) + ldy*((k)-1)))) + (m - 1)] - dssp * (-(4.0e+00 * u[(5*(((i)+1) + ldx*(((2)+1) + ldy*((k)-1)))) + (m - 1)])
								+ 6.0e+00 * u[(5*(((i)+1) + ldx*(((3)+1) + ldy*((k)-1)))) + (m - 1)] - 4.0e+00 * u[(5*(((i)+1) + ldx*(((4)+1) + ldy*((k)-1)))) + (m - 1)] + u[(5*(((i)+1) + ldx*(((5)+1) + ldy*((k)-1)))) + (m - 1)]);
					}
				}
			}
	}

	private void rhsY_5(int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, j, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
			for (j = jst1; j <= jend1; j++){
				for (i = ist; i <= iend; i++){
					for (m = 1; m <= 5; m++){
						rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - dssp * (u[(5*(((i)+1) + ldx*(((j - 2)+1) + ldy*((k)-1)))) + (m - 1)]
								- 4.0e+00 * u[(5*(((i)+1) + ldx*(((j - 1)+1) + ldy*((k)-1)))) + (m - 1)] + 6.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - 4.0e+00 * u[(5*(((i)+1) + ldx*(((j + 1)+1) + ldy*((k)-1)))) + (m - 1)]
								+ u[(5*(((i)+1) + ldx*(((j + 2)+1) + ldy*((k)-1)))) + (m - 1)]);
					}
				}
			}
	}

	private void rhsY_6(int k, int l1, int l2, int ist1, int iend1, int jst1, int jend1){
		int i, j, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
			if (east == -1){
				for (i = ist; i <= iend; i++){
					for (m = 1; m <= 5; m++){
						rsd[(5*(((i)+1) + ldx*(((ny - 2)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((ny - 2)+1) + ldy*((k)-1)))) + (m - 1)] - dssp * (u[(5*(((i)+1) + ldx*(((ny - 4)+1) + ldy*((k)-1)))) + (m - 1)]
								- 4.0e+00 * u[(5*(((i)+1) + ldx*(((ny - 3)+1) + ldy*((k)-1)))) + (m - 1)] + 6.0e+00 * u[(5*(((i)+1) + ldx*(((ny - 2)+1) + ldy*((k)-1)))) + (m - 1)]
								- 4.0e+00 * u[(5*(((i)+1) + ldx*(((ny - 1)+1) + ldy*((k)-1)))) + (m - 1)]);
						rsd[(5*(((i)+1) + ldx*(((ny - 1)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((ny - 1)+1) + ldy*((k)-1)))) + (m - 1)] - dssp * (u[(5*(((i)+1) + ldx*(((ny - 3)+1) + ldy*((k)-1)))) + (m - 1)]
								- 4.0e+00 * u[(5*(((i)+1) + ldx*(((ny - 2)+1) + ldy*((k)-1)))) + (m - 1)] + 5.0e+00 * u[(5*(((i)+1) + ldx*(((ny - 1)+1) + ldy*((k)-1)))) + (m - 1)]);
					}
				}
			}
	}

	/*
	 * steady-state residuals: z direction
	 */
	private void rhsZ(){
		rhsZ_0();
		rhsZ_1();
		rhsZ_2();
		rhsZ_3();
		rhsZ_4();
		rhsZ_5();
		rhsZ_6();
	}

	private void rhsZ_0(){
		int i, j, k, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
		for (k = 1; k <= nz; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))] = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3];
					u41 = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] / u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))];
					q = 0.50e+00 * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] + u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2]
							+ u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3]) / u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))];
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] * u41;
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] * u41;
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] * u41 + c2 * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] - q);
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] = (c1 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] - c2 * q) * u41;
				}
			}
		}
	}

	private void rhsZ_1(){
		int i, j, k, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
		for (k = 2; k <= nz - 1; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					for (m = 1; m <= 5; m++){
						rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - tz2 * (flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k + 1)-1)))) + (m - 1)]
								- flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k - 1)-1)))) + (m - 1)]);
					}
				}
			}
		}
	}

	private void rhsZ_2(){
		int i, j, k, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
		for (k = 2; k <= nz; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					tmp = 1.0e+00 / u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))];
					u21k = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1];
					u31k = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2];
					u41k = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3];
					u51k = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4];
					tmp = 1.0e+00 / u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k - 1)-1))))];
					u21km1 = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k - 1)-1)))) + 1];
					u31km1 = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k - 1)-1)))) + 2];
					u41km1 = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k - 1)-1)))) + 3];
					u51km1 = tmp * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k - 1)-1)))) + 4];
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] = tz3 * (u21k - u21km1);
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] = tz3 * (u31k - u31km1);
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] = (4.0e+00 / 3.0e+00) * tz3 * (u41k - u41km1);
					flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] = 0.50e+00 * (1.0e+00 - c1 * c5) * tz3 * ((sq(u21k) + sq(u31k) + sq(u41k)) - (sq(u21km1)
							+ sq(u31km1) + sq(u41km1))) + (1.0e+00 / 6.0e+00) * tz3 * (sq(u41k) - sq(u41km1)) + c1 * c5 * tz3 * (u51k - u51km1);
				}
			}
		}
	}

	private void rhsZ_3(){
		int i, j, k, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
		for (k = 2; k <= nz - 1; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))] + dz1 * tz1 * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k - 1)-1))))] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))]
							+ u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k + 1)-1))))]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] + tz3 * c3 * c4 * (flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k + 1)-1)))) + 1] - flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1])
							+ dz2 * tz1 * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k - 1)-1)))) + 1] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 1] + u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k + 1)-1)))) + 1]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] + tz3 * c3 * c4 * (flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k + 1)-1)))) + 2] - flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2])
							+ dz3 * tz1 * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k - 1)-1)))) + 2] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 2] + u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k + 1)-1)))) + 2]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] + tz3 * c3 * c4 * (flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k + 1)-1)))) + 3] - flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3])
							+ dz4 * tz1 * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k - 1)-1)))) + 3] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 3] + u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k + 1)-1)))) + 3]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] + tz3 * c3 * c4 * (flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k + 1)-1)))) + 4] - flux[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4])
							+ dz5 * tz1 * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k - 1)-1)))) + 4] - 2.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + 4] + u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k + 1)-1)))) + 4]);
				}
			}
		}
	}

	private void rhsZ_4(){
		int i, j, k, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
		for (j = jst; j <= jend; j++){
			for (i = ist; i <= iend; i++){
				for (m = 1; m <= 5; m++){
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((2)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((2)-1)))) + (m - 1)] - dssp * (5.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((2)-1)))) + (m - 1)]
							- 4.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((3)-1)))) + (m - 1)] + u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((4)-1)))) + (m - 1)]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((3)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((3)-1)))) + (m - 1)] - dssp * (-(4.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((2)-1)))) + (m - 1)])
							+ 6.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((3)-1)))) + (m - 1)] - 4.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((4)-1)))) + (m - 1)] + u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((5)-1)))) + (m - 1)]);
				}
			}
		}
	}

	private void rhsZ_5(){
		int i, j, k, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
		for (k = 4; k <= nz - 3; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					for (m = 1; m <= 5; m++){
						rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - dssp * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k - 2)-1)))) + (m - 1)]
								- 4.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k - 1)-1)))) + (m - 1)] + 6.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1)))) + (m - 1)] - 4.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k + 1)-1)))) + (m - 1)]
								+ u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k + 2)-1)))) + (m - 1)]);
					}
				}
			}
		}
	}

	private void rhsZ_6(){
		int i, j, k, m;
		double q, u21, u31, u41, tmp, u21i, u31i, u41i;
		double u51i, u21j, u31j, u41j, u51j, u21k, u31k, u41k;
		double u51k, u21im1, u31im1, u41im1, u51im1, u21jm1, u31jm1, u41jm1;
		double u51jm1, u21km1, u31km1, u41km1, u51km1;
		for (j = jst; j <= jend; j++){
			for (i = ist; i <= iend; i++){
				for (m = 1; m <= 5; m++){
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((nz - 2)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((nz - 2)-1)))) + (m - 1)] - dssp * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((nz - 4)-1)))) + (m - 1)]
							- 4.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((nz - 3)-1)))) + (m - 1)] + 6.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((nz - 2)-1)))) + (m - 1)]
							- 4.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((nz - 1)-1)))) + (m - 1)]);
					rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((nz - 1)-1)))) + (m - 1)] = rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((nz - 1)-1)))) + (m - 1)] - dssp * (u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((nz - 3)-1)))) + (m - 1)]
							- 4.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((nz - 2)-1)))) + (m - 1)] + 5.0e+00 * u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((nz - 1)-1)))) + (m - 1)]);
				}
			}
		}
	}

	/*
	 * Lower triangular part of the jacobian matrix (blocks a, b, c, d) of the line (j,k)
	 */
	private void jacld(int j, int k){

		int i;
		double r43, c1345, c34, tmp1, tmp2, tmp3;

		r43 = (4.0e+00 / 3.0e+00);
		c1345 = c1 * c3 * c4 * c5;
		c34 = c3 * c4;
		/* Optimization: fields in locals and indices without ix(); the operands of u are read once per point and sq(x) is written as x*x (same expressions) */
		final double[] u = this.u, a = this.a, b = this.b, c = this.c, d = this.d;
		final double dt = this.dt, tx1 = this.tx1, tx2 = this.tx2, ty1 = this.ty1, ty2 = this.ty2, tz1 = this.tz1, tz2 = this.tz2, dx1 = this.dx1, dx2 = this.dx2, dx3 = this.dx3, dx4 = this.dx4, dx5 = this.dx5, dy1 = this.dy1, dy2 = this.dy2, dy3 = this.dy3, dy4 = this.dy4, dy5 = this.dy5, dz1 = this.dz1, dz2 = this.dz2, dz3 = this.dz3, dz4 = this.dz4, dz5 = this.dz5;
		final int s5x = 5*ldx, s5xy = 5*ldx*ldy, base = ix(0, j, k);
		for (i = ist; i <= iend; i++){
			final int pc = base + 5*i, pkm = pc - s5xy, pjm = pc - s5x, pim = pc - 5;
			final double u0c = u[pc + 0], u1c = u[pc + 1], u2c = u[pc + 2], u3c = u[pc + 3], u4c = u[pc + 4];
			final double u0km = u[pkm + 0], u1km = u[pkm + 1], u2km = u[pkm + 2], u3km = u[pkm + 3], u4km = u[pkm + 4];
			final double u0jm = u[pjm + 0], u1jm = u[pjm + 1], u2jm = u[pjm + 2], u3jm = u[pjm + 3], u4jm = u[pjm + 4];
			final double u0im = u[pim + 0], u1im = u[pim + 1], u2im = u[pim + 2], u3im = u[pim + 3], u4im = u[pim + 4];
			tmp1 = 1.0e+00 / u0c;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			d[25*(i - 1)] = 1.0e+00 + dt * 2.0e+00 * (tx1 * dx1 + ty1 * dy1 + tz1 * dz1);
			d[25*(i - 1) + 5*1] = 0.0e+00;
			d[25*(i - 1) + 5*2] = 0.0e+00;
			d[25*(i - 1) + 5*3] = 0.0e+00;
			d[25*(i - 1) + 5*4] = 0.0e+00;
			d[25*(i - 1) + 1] = dt * 2.0e+00 * (tx1 * (-(r43 * c34 * tmp2 * u1c))
					+ ty1 * (-(c34 * tmp2 * u1c)) + tz1 * (-(c34 * tmp2 * u1c)));
			d[25*(i - 1) + 1 + 5*1] = 1.0e+00 + dt * 2.0e+00 * (tx1 * r43 * c34 * tmp1 + ty1 * c34 * tmp1 + tz1 * c34 * tmp1)
					+ dt * 2.0e+00 * (tx1 * dx2 + ty1 * dy2 + tz1 * dz2);
			d[25*(i - 1) + 1 + 5*2] = 0.0e+00;
			d[25*(i - 1) + 1 + 5*3] = 0.0e+00;
			d[25*(i - 1) + 1 + 5*4] = 0.0e+00;
			d[25*(i - 1) + 2] = dt * 2.0e+00 * (tx1 * (-(c34 * tmp2 * u2c))
					+ ty1 * (-(r43 * c34 * tmp2 * u2c)) + tz1 * (-(c34 * tmp2 * u2c)));
			d[25*(i - 1) + 2 + 5*1] = 0.0e+00;
			d[25*(i - 1) + 2 + 5*2] = 1.0e+00 + dt * 2.0e+00 * (tx1 * c34 * tmp1 + ty1 * r43 * c34 * tmp1 + tz1 * c34 * tmp1)
					+ dt * 2.0e+00 * (tx1 * dx3 + ty1 * dy3 + tz1 * dz3);
			d[25*(i - 1) + 2 + 5*3] = 0.0e+00;
			d[25*(i - 1) + 2 + 5*4] = 0.0e+00;
			d[25*(i - 1) + 3] = dt * 2.0e+00 * (tx1 * (-(c34 * tmp2 * u3c))
					+ ty1 * (-(c34 * tmp2 * u3c)) + tz1 * (-(r43 * c34 * tmp2 * u3c)));
			d[25*(i - 1) + 3 + 5*1] = 0.0e+00;
			d[25*(i - 1) + 3 + 5*2] = 0.0e+00;
			d[25*(i - 1) + 3 + 5*3] = 1.0e+00 + dt * 2.0e+00 * (tx1 * c34 * tmp1 + ty1 * c34 * tmp1 + tz1 * r43 * c34 * tmp1)
					+ dt * 2.0e+00 * (tx1 * dx4 + ty1 * dy4 + tz1 * dz4);
			d[25*(i - 1) + 3 + 5*4] = 0.0e+00;
			d[25*(i - 1) + 4] = dt * 2.0e+00 * (tx1 * (-((r43 * c34 - c1345) * tmp3 * (u1c*u1c)) - (c34
					- c1345) * tmp3 * (u2c*u2c) - (c34 - c1345) * tmp3 * (u3c*u3c)
					- c1345 * tmp2 * u4c) + ty1 * (-((c34 - c1345) * tmp3 * (u1c*u1c)) - (r43 * c34
					- c1345) * tmp3 * (u2c*u2c) - (c34 - c1345) * tmp3 * (u3c*u3c)
					- c1345 * tmp2 * u4c) + tz1 * (-((c34 - c1345) * tmp3 * (u1c*u1c)) - (c34
					- c1345) * tmp3 * (u2c*u2c) - (r43 * c34 - c1345) * tmp3 * (u3c*u3c)
					- c1345 * tmp2 * u4c));
			d[25*(i - 1) + 4 + 5*1] = dt * 2.0e+00 * (tx1 * (r43 * c34 - c1345) * tmp2 * u1c + ty1 * (c34
					- c1345) * tmp2 * u1c + tz1 * (c34 - c1345) * tmp2 * u1c);
			d[25*(i - 1) + 4 + 5*2] = dt * 2.0e+00 * (tx1 * (c34 - c1345) * tmp2 * u2c + ty1 * (r43 * c34
					- c1345) * tmp2 * u2c + tz1 * (c34 - c1345) * tmp2 * u2c);
			d[25*(i - 1) + 4 + 5*3] = dt * 2.0e+00 * (tx1 * (c34 - c1345) * tmp2 * u3c + ty1 * (c34
					- c1345) * tmp2 * u3c + tz1 * (r43 * c34 - c1345) * tmp2 * u3c);
			d[25*(i - 1) + 4 + 5*4] = 1.0e+00 + dt * 2.0e+00 * (tx1 * c1345 * tmp1 + ty1 * c1345 * tmp1 + tz1 * c1345 * tmp1)
					+ dt * 2.0e+00 * (tx1 * dx5 + ty1 * dy5 + tz1 * dz5);
			tmp1 = 1.0e+00 / u0km;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			a[25*(i - 1)] = -(dt * tz1 * dz1);
			a[25*(i - 1) + 5*1] = 0.0e+00;
			a[25*(i - 1) + 5*2] = 0.0e+00;
			a[25*(i - 1) + 5*3] = -(dt * tz2);
			a[25*(i - 1) + 5*4] = 0.0e+00;
			a[25*(i - 1) + 1] = -(dt * tz2 * (-((u1km * u3km) * tmp2)))
					- dt * tz1 * (-(c34 * tmp2 * u1km));
			a[25*(i - 1) + 1 + 5*1] = -(dt * tz2 * (u3km * tmp1)) - dt * tz1 * c34 * tmp1 - dt * tz1 * dz2;
			a[25*(i - 1) + 1 + 5*2] = 0.0e+00;
			a[25*(i - 1) + 1 + 5*3] = -(dt * tz2 * (u1km * tmp1));
			a[25*(i - 1) + 1 + 5*4] = 0.0e+00;
			a[25*(i - 1) + 2] = -(dt * tz2 * (-((u2km * u3km) * tmp2)))
					- dt * tz1 * (-(c34 * tmp2 * u2km));
			a[25*(i - 1) + 2 + 5*1] = 0.0e+00;
			a[25*(i - 1) + 2 + 5*2] = -(dt * tz2 * (u3km * tmp1)) - dt * tz1 * (c34 * tmp1) - dt * tz1 * dz3;
			a[25*(i - 1) + 2 + 5*3] = -(dt * tz2 * (u2km * tmp1));
			a[25*(i - 1) + 2 + 5*4] = 0.0e+00;
			a[25*(i - 1) + 3] = -(dt * tz2 * (-sq((u3km * tmp1))
					+ 0.50e+00 * c2 * ((u1km * u1km + u2km * u2km
					+ u3km * u3km) * tmp2)))
					- dt * tz1 * (-(r43 * c34 * tmp2 * u3km));
			a[25*(i - 1) + 3 + 5*1] = -(dt * tz2 * (-(c2 * (u1km * tmp1))));
			a[25*(i - 1) + 3 + 5*2] = -(dt * tz2 * (-(c2 * (u2km * tmp1))));
			a[25*(i - 1) + 3 + 5*3] = -(dt * tz2 * (2.0e+00 - c2) * (u3km * tmp1))
					- dt * tz1 * (r43 * c34 * tmp1) - dt * tz1 * dz4;
			a[25*(i - 1) + 3 + 5*4] = -(dt * tz2 * c2);
			a[25*(i - 1) + 4] = -(dt * tz2 * ((c2 * (u1km * u1km
					+ u2km * u2km + u3km * u3km) * tmp2
					- c1 * (u4km * tmp1)) * (u3km * tmp1))) - dt * tz1 * (-((c34
					- c1345) * tmp3 * (u1km*u1km)) - (c34 - c1345) * tmp3 * (u2km*u2km) - (r43 * c34
					- c1345) * tmp3 * (u3km*u3km) - c1345 * tmp2 * u4km);
			a[25*(i - 1) + 4 + 5*1] = -(dt * tz2 * (-(c2 * (u1km * u3km) * tmp2)))
					- dt * tz1 * (c34 - c1345) * tmp2 * u1km;
			a[25*(i - 1) + 4 + 5*2] = -(dt * tz2 * (-(c2 * (u2km * u3km) * tmp2)))
					- dt * tz1 * (c34 - c1345) * tmp2 * u2km;
			a[25*(i - 1) + 4 + 5*3] = -(dt * tz2 * (c1 * (u4km * tmp1)
					- 0.50e+00 * c2 * ((u1km * u1km + u2km * u2km
					+ 3.0e+00 * u3km * u3km) * tmp2))) - dt * tz1 * (r43 * c34
					- c1345) * tmp2 * u3km;
			a[25*(i - 1) + 4 + 5*4] = -(dt * tz2 * (c1 * (u3km * tmp1))) - dt * tz1 * c1345 * tmp1
					- dt * tz1 * dz5;
			tmp1 = 1.0e+00 / u0jm;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			b[25*(i - 1)] = -(dt * ty1 * dy1);
			b[25*(i - 1) + 5*1] = 0.0e+00;
			b[25*(i - 1) + 5*2] = -(dt * ty2);
			b[25*(i - 1) + 5*3] = 0.0e+00;
			b[25*(i - 1) + 5*4] = 0.0e+00;
			b[25*(i - 1) + 1] = -(dt * ty2 * (-((u1jm * u2jm) * tmp2)))
					- dt * ty1 * (-(c34 * tmp2 * u1jm));
			b[25*(i - 1) + 1 + 5*1] = -(dt * ty2 * (u2jm * tmp1)) - dt * ty1 * (c34 * tmp1) - dt * ty1 * dy2;
			b[25*(i - 1) + 1 + 5*2] = -(dt * ty2 * (u1jm * tmp1));
			b[25*(i - 1) + 1 + 5*3] = 0.0e+00;
			b[25*(i - 1) + 1 + 5*4] = 0.0e+00;
			b[25*(i - 1) + 2] = -(dt * ty2 * (-sq((u2jm * tmp1))
					+ 0.50e+00 * c2 * ((u1jm * u1jm + u2jm * u2jm
					+ u3jm * u3jm) * tmp2)))
					- dt * ty1 * (-(r43 * c34 * tmp2 * u2jm));
			b[25*(i - 1) + 2 + 5*1] = -(dt * ty2 * (-(c2 * (u1jm * tmp1))));
			b[25*(i - 1) + 2 + 5*2] = -(dt * ty2 * ((2.0e+00 - c2) * (u2jm * tmp1)))
					- dt * ty1 * (r43 * c34 * tmp1) - dt * ty1 * dy3;
			b[25*(i - 1) + 2 + 5*3] = -(dt * ty2 * (-(c2 * (u3jm * tmp1))));
			b[25*(i - 1) + 2 + 5*4] = -(dt * ty2 * c2);
			b[25*(i - 1) + 3] = -(dt * ty2 * (-((u2jm * u3jm) * tmp2)))
					- dt * ty1 * (-(c34 * tmp2 * u3jm));
			b[25*(i - 1) + 3 + 5*1] = 0.0e+00;
			b[25*(i - 1) + 3 + 5*2] = -(dt * ty2 * (u3jm * tmp1));
			b[25*(i - 1) + 3 + 5*3] = -(dt * ty2 * (u2jm * tmp1)) - dt * ty1 * (c34 * tmp1) - dt * ty1 * dy4;
			b[25*(i - 1) + 3 + 5*4] = 0.0e+00;
			b[25*(i - 1) + 4] = -(dt * ty2 * ((c2 * (u1jm * u1jm
					+ u2jm * u2jm + u3jm * u3jm) * tmp2
					- c1 * (u4jm * tmp1)) * (u2jm * tmp1))) - dt * ty1 * (-((c34
					- c1345) * tmp3 * (u1jm*u1jm)) - (r43 * c34 - c1345) * tmp3 * (u2jm*u2jm) - (c34
					- c1345) * tmp3 * (u3jm*u3jm) - c1345 * tmp2 * u4jm);
			b[25*(i - 1) + 4 + 5*1] = -(dt * ty2 * (-(c2 * (u1jm * u2jm) * tmp2)))
					- dt * ty1 * (c34 - c1345) * tmp2 * u1jm;
			b[25*(i - 1) + 4 + 5*2] = -(dt * ty2 * (c1 * (u4jm * tmp1)
					- 0.50e+00 * c2 * ((u1jm * u1jm
					+ 3.0e+00 * u2jm * u2jm
					+ u3jm * u3jm) * tmp2))) - dt * ty1 * (r43 * c34
					- c1345) * tmp2 * u2jm;
			b[25*(i - 1) + 4 + 5*3] = -(dt * ty2 * (-(c2 * (u2jm * u3jm) * tmp2)))
					- dt * ty1 * (c34 - c1345) * tmp2 * u3jm;
			b[25*(i - 1) + 4 + 5*4] = -(dt * ty2 * (c1 * (u2jm * tmp1))) - dt * ty1 * c1345 * tmp1
					- dt * ty1 * dy5;
			tmp1 = 1.0e+00 / u0im;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			c[25*(i - 1)] = -(dt * tx1 * dx1);
			c[25*(i - 1) + 5*1] = -(dt * tx2);
			c[25*(i - 1) + 5*2] = 0.0e+00;
			c[25*(i - 1) + 5*3] = 0.0e+00;
			c[25*(i - 1) + 5*4] = 0.0e+00;
			c[25*(i - 1) + 1] = -(dt * tx2 * (-sq((u1im * tmp1))
					+ c2 * 0.50e+00 * (u1im * u1im + u2im * u2im
					+ u3im * u3im) * tmp2))
					- dt * tx1 * (-(r43 * c34 * tmp2 * u1im));
			c[25*(i - 1) + 1 + 5*1] = -(dt * tx2 * ((2.0e+00 - c2) * (u1im * tmp1)))
					- dt * tx1 * (r43 * c34 * tmp1) - dt * tx1 * dx2;
			c[25*(i - 1) + 1 + 5*2] = -(dt * tx2 * (-(c2 * (u2im * tmp1))));
			c[25*(i - 1) + 1 + 5*3] = -(dt * tx2 * (-(c2 * (u3im * tmp1))));
			c[25*(i - 1) + 1 + 5*4] = -(dt * tx2 * c2);
			c[25*(i - 1) + 2] = -(dt * tx2 * (-((u1im * u2im) * tmp2)))
					- dt * tx1 * (-(c34 * tmp2 * u2im));
			c[25*(i - 1) + 2 + 5*1] = -(dt * tx2 * (u2im * tmp1));
			c[25*(i - 1) + 2 + 5*2] = -(dt * tx2 * (u1im * tmp1)) - dt * tx1 * (c34 * tmp1) - dt * tx1 * dx3;
			c[25*(i - 1) + 2 + 5*3] = 0.0e+00;
			c[25*(i - 1) + 2 + 5*4] = 0.0e+00;
			c[25*(i - 1) + 3] = -(dt * tx2 * (-((u1im * u3im) * tmp2)))
					- dt * tx1 * (-(c34 * tmp2 * u3im));
			c[25*(i - 1) + 3 + 5*1] = -(dt * tx2 * (u3im * tmp1));
			c[25*(i - 1) + 3 + 5*2] = 0.0e+00;
			c[25*(i - 1) + 3 + 5*3] = -(dt * tx2 * (u1im * tmp1)) - dt * tx1 * (c34 * tmp1) - dt * tx1 * dx4;
			c[25*(i - 1) + 3 + 5*4] = 0.0e+00;
			c[25*(i - 1) + 4] = -(dt * tx2 * ((c2 * (u1im * u1im
					+ u2im * u2im + u3im * u3im) * tmp2
					- c1 * (u4im * tmp1)) * (u1im * tmp1))) - dt * tx1 * (-((r43 * c34
					- c1345) * tmp3 * (u1im*u1im)) - (c34 - c1345) * tmp3 * (u2im*u2im) - (c34
					- c1345) * tmp3 * (u3im*u3im) - c1345 * tmp2 * u4im);
			c[25*(i - 1) + 4 + 5*1] = -(dt * tx2 * (c1 * (u4im * tmp1)
					- 0.50e+00 * c2 * ((3.0e+00 * u1im * u1im
					+ u2im * u2im + u3im * u3im) * tmp2)))
					- dt * tx1 * (r43 * c34 - c1345) * tmp2 * u1im;
			c[25*(i - 1) + 4 + 5*2] = -(dt * tx2 * (-(c2 * (u2im * u1im) * tmp2)))
					- dt * tx1 * (c34 - c1345) * tmp2 * u2im;
			c[25*(i - 1) + 4 + 5*3] = -(dt * tx2 * (-(c2 * (u3im * u1im) * tmp2)))
					- dt * tx1 * (c34 - c1345) * tmp2 * u3im;
			c[25*(i - 1) + 4 + 5*4] = -(dt * tx2 * (c1 * (u1im * tmp1))) - dt * tx1 * c1345 * tmp1
					- dt * tx1 * dx5;
		}
	}

	/*
	 * Strictly upper triangular part of the jacobian matrix (blocks a, b, c, d) of the line (j,k)
	 */
	private void jacu(int j, int k){

		int i;
		double r43, c1345, c34, tmp1, tmp2, tmp3;

		r43 = (4.0e+00 / 3.0e+00);
		c1345 = c1 * c3 * c4 * c5;
		c34 = c3 * c4;
		/* Optimization: fields in locals and indices without ix(); the operands of u are read once per point and sq(x) is written as x*x (same expressions) */
		final double[] u = this.u, a = this.a, b = this.b, c = this.c, d = this.d;
		final double dt = this.dt, tx1 = this.tx1, tx2 = this.tx2, ty1 = this.ty1, ty2 = this.ty2, tz1 = this.tz1, tz2 = this.tz2, dx1 = this.dx1, dx2 = this.dx2, dx3 = this.dx3, dx4 = this.dx4, dx5 = this.dx5, dy1 = this.dy1, dy2 = this.dy2, dy3 = this.dy3, dy4 = this.dy4, dy5 = this.dy5, dz1 = this.dz1, dz2 = this.dz2, dz3 = this.dz3, dz4 = this.dz4, dz5 = this.dz5;
		final int s5x = 5*ldx, s5xy = 5*ldx*ldy, base = ix(0, j, k);
		for (i = ist; i <= iend; i++){
			final int pc = base + 5*i, pip = pc + 5, pjp = pc + s5x, pkp = pc + s5xy;
			final double u0c = u[pc + 0], u1c = u[pc + 1], u2c = u[pc + 2], u3c = u[pc + 3], u4c = u[pc + 4];
			final double u0ip = u[pip + 0], u1ip = u[pip + 1], u2ip = u[pip + 2], u3ip = u[pip + 3], u4ip = u[pip + 4];
			final double u0jp = u[pjp + 0], u1jp = u[pjp + 1], u2jp = u[pjp + 2], u3jp = u[pjp + 3], u4jp = u[pjp + 4];
			final double u0kp = u[pkp + 0], u1kp = u[pkp + 1], u2kp = u[pkp + 2], u3kp = u[pkp + 3], u4kp = u[pkp + 4];
			tmp1 = 1.0e+00 / u0c;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			d[25*(i - 1)] = 1.0e+00 + dt * 2.0e+00 * (tx1 * dx1 + ty1 * dy1 + tz1 * dz1);
			d[25*(i - 1) + 5*1] = 0.0e+00;
			d[25*(i - 1) + 5*2] = 0.0e+00;
			d[25*(i - 1) + 5*3] = 0.0e+00;
			d[25*(i - 1) + 5*4] = 0.0e+00;
			d[25*(i - 1) + 1] = dt * 2.0e+00 * (tx1 * (-(r43 * c34 * tmp2 * u1c))
					+ ty1 * (-(c34 * tmp2 * u1c)) + tz1 * (-(c34 * tmp2 * u1c)));
			d[25*(i - 1) + 1 + 5*1] = 1.0e+00 + dt * 2.0e+00 * (tx1 * r43 * c34 * tmp1 + ty1 * c34 * tmp1 + tz1 * c34 * tmp1)
					+ dt * 2.0e+00 * (tx1 * dx2 + ty1 * dy2 + tz1 * dz2);
			d[25*(i - 1) + 1 + 5*2] = 0.0e+00;
			d[25*(i - 1) + 1 + 5*3] = 0.0e+00;
			d[25*(i - 1) + 1 + 5*4] = 0.0e+00;
			d[25*(i - 1) + 2] = dt * 2.0e+00 * (tx1 * (-(c34 * tmp2 * u2c))
					+ ty1 * (-(r43 * c34 * tmp2 * u2c)) + tz1 * (-(c34 * tmp2 * u2c)));
			d[25*(i - 1) + 2 + 5*1] = 0.0e+00;
			d[25*(i - 1) + 2 + 5*2] = 1.0e+00 + dt * 2.0e+00 * (tx1 * c34 * tmp1 + ty1 * r43 * c34 * tmp1 + tz1 * c34 * tmp1)
					+ dt * 2.0e+00 * (tx1 * dx3 + ty1 * dy3 + tz1 * dz3);
			d[25*(i - 1) + 2 + 5*3] = 0.0e+00;
			d[25*(i - 1) + 2 + 5*4] = 0.0e+00;
			d[25*(i - 1) + 3] = dt * 2.0e+00 * (tx1 * (-(c34 * tmp2 * u3c))
					+ ty1 * (-(c34 * tmp2 * u3c)) + tz1 * (-(r43 * c34 * tmp2 * u3c)));
			d[25*(i - 1) + 3 + 5*1] = 0.0e+00;
			d[25*(i - 1) + 3 + 5*2] = 0.0e+00;
			d[25*(i - 1) + 3 + 5*3] = 1.0e+00 + dt * 2.0e+00 * (tx1 * c34 * tmp1 + ty1 * c34 * tmp1 + tz1 * r43 * c34 * tmp1)
					+ dt * 2.0e+00 * (tx1 * dx4 + ty1 * dy4 + tz1 * dz4);
			d[25*(i - 1) + 3 + 5*4] = 0.0e+00;
			d[25*(i - 1) + 4] = dt * 2.0e+00 * (tx1 * (-((r43 * c34 - c1345) * tmp3 * (u1c*u1c)) - (c34
					- c1345) * tmp3 * (u2c*u2c) - (c34 - c1345) * tmp3 * (u3c*u3c)
					- c1345 * tmp2 * u4c) + ty1 * (-((c34 - c1345) * tmp3 * (u1c*u1c)) - (r43 * c34
					- c1345) * tmp3 * (u2c*u2c) - (c34 - c1345) * tmp3 * (u3c*u3c)
					- c1345 * tmp2 * u4c) + tz1 * (-((c34 - c1345) * tmp3 * (u1c*u1c)) - (c34
					- c1345) * tmp3 * (u2c*u2c) - (r43 * c34 - c1345) * tmp3 * (u3c*u3c)
					- c1345 * tmp2 * u4c));
			d[25*(i - 1) + 4 + 5*1] = dt * 2.0e+00 * (tx1 * (r43 * c34 - c1345) * tmp2 * u1c + ty1 * (c34
					- c1345) * tmp2 * u1c + tz1 * (c34 - c1345) * tmp2 * u1c);
			d[25*(i - 1) + 4 + 5*2] = dt * 2.0e+00 * (tx1 * (c34 - c1345) * tmp2 * u2c + ty1 * (r43 * c34
					- c1345) * tmp2 * u2c + tz1 * (c34 - c1345) * tmp2 * u2c);
			d[25*(i - 1) + 4 + 5*3] = dt * 2.0e+00 * (tx1 * (c34 - c1345) * tmp2 * u3c + ty1 * (c34
					- c1345) * tmp2 * u3c + tz1 * (r43 * c34 - c1345) * tmp2 * u3c);
			d[25*(i - 1) + 4 + 5*4] = 1.0e+00 + dt * 2.0e+00 * (tx1 * c1345 * tmp1 + ty1 * c1345 * tmp1 + tz1 * c1345 * tmp1)
					+ dt * 2.0e+00 * (tx1 * dx5 + ty1 * dy5 + tz1 * dz5);
			tmp1 = 1.0e+00 / u0ip;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			a[25*(i - 1)] = -(dt * tx1 * dx1);
			a[25*(i - 1) + 5*1] = dt * tx2;
			a[25*(i - 1) + 5*2] = 0.0e+00;
			a[25*(i - 1) + 5*3] = 0.0e+00;
			a[25*(i - 1) + 5*4] = 0.0e+00;
			a[25*(i - 1) + 1] = dt * tx2 * (-sq((u1ip * tmp1))
					+ c2 * 0.50e+00 * (u1ip * u1ip + u2ip * u2ip
					+ u3ip * u3ip) * tmp2)
					- dt * tx1 * (-(r43 * c34 * tmp2 * u1ip));
			a[25*(i - 1) + 1 + 5*1] = dt * tx2 * ((2.0e+00 - c2) * (u1ip * tmp1))
					- dt * tx1 * (r43 * c34 * tmp1) - dt * tx1 * dx2;
			a[25*(i - 1) + 1 + 5*2] = dt * tx2 * (-(c2 * (u2ip * tmp1)));
			a[25*(i - 1) + 1 + 5*3] = dt * tx2 * (-(c2 * (u3ip * tmp1)));
			a[25*(i - 1) + 1 + 5*4] = dt * tx2 * c2;
			a[25*(i - 1) + 2] = dt * tx2 * (-((u1ip * u2ip) * tmp2))
					- dt * tx1 * (-(c34 * tmp2 * u2ip));
			a[25*(i - 1) + 2 + 5*1] = dt * tx2 * (u2ip * tmp1);
			a[25*(i - 1) + 2 + 5*2] = dt * tx2 * (u1ip * tmp1) - dt * tx1 * (c34 * tmp1) - dt * tx1 * dx3;
			a[25*(i - 1) + 2 + 5*3] = 0.0e+00;
			a[25*(i - 1) + 2 + 5*4] = 0.0e+00;
			a[25*(i - 1) + 3] = dt * tx2 * (-((u1ip * u3ip) * tmp2))
					- dt * tx1 * (-(c34 * tmp2 * u3ip));
			a[25*(i - 1) + 3 + 5*1] = dt * tx2 * (u3ip * tmp1);
			a[25*(i - 1) + 3 + 5*2] = 0.0e+00;
			a[25*(i - 1) + 3 + 5*3] = dt * tx2 * (u1ip * tmp1) - dt * tx1 * (c34 * tmp1) - dt * tx1 * dx4;
			a[25*(i - 1) + 3 + 5*4] = 0.0e+00;
			a[25*(i - 1) + 4] = dt * tx2 * ((c2 * (u1ip * u1ip
					+ u2ip * u2ip + u3ip * u3ip) * tmp2
					- c1 * (u4ip * tmp1)) * (u1ip * tmp1)) - dt * tx1 * (-((r43 * c34
					- c1345) * tmp3 * (u1ip*u1ip)) - (c34 - c1345) * tmp3 * (u2ip*u2ip) - (c34
					- c1345) * tmp3 * (u3ip*u3ip) - c1345 * tmp2 * u4ip);
			a[25*(i - 1) + 4 + 5*1] = dt * tx2 * (c1 * (u4ip * tmp1)
					- 0.50e+00 * c2 * ((3.0e+00 * u1ip * u1ip
					+ u2ip * u2ip + u3ip * u3ip) * tmp2))
					- dt * tx1 * (r43 * c34 - c1345) * tmp2 * u1ip;
			a[25*(i - 1) + 4 + 5*2] = dt * tx2 * (-(c2 * (u2ip * u1ip) * tmp2))
					- dt * tx1 * (c34 - c1345) * tmp2 * u2ip;
			a[25*(i - 1) + 4 + 5*3] = dt * tx2 * (-(c2 * (u3ip * u1ip) * tmp2))
					- dt * tx1 * (c34 - c1345) * tmp2 * u3ip;
			a[25*(i - 1) + 4 + 5*4] = dt * tx2 * (c1 * (u1ip * tmp1)) - dt * tx1 * c1345 * tmp1
					- dt * tx1 * dx5;
			tmp1 = 1.0e+00 / u0jp;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			b[25*(i - 1)] = -(dt * ty1 * dy1);
			b[25*(i - 1) + 5*1] = 0.0e+00;
			b[25*(i - 1) + 5*2] = dt * ty2;
			b[25*(i - 1) + 5*3] = 0.0e+00;
			b[25*(i - 1) + 5*4] = 0.0e+00;
			b[25*(i - 1) + 1] = dt * ty2 * (-((u1jp * u2jp) * tmp2))
					- dt * ty1 * (-(c34 * tmp2 * u1jp));
			b[25*(i - 1) + 1 + 5*1] = dt * ty2 * (u2jp * tmp1) - dt * ty1 * (c34 * tmp1) - dt * ty1 * dy2;
			b[25*(i - 1) + 1 + 5*2] = dt * ty2 * (u1jp * tmp1);
			b[25*(i - 1) + 1 + 5*3] = 0.0e+00;
			b[25*(i - 1) + 1 + 5*4] = 0.0e+00;
			b[25*(i - 1) + 2] = dt * ty2 * (-sq((u2jp * tmp1))
					+ 0.50e+00 * c2 * ((u1jp * u1jp + u2jp * u2jp
					+ u3jp * u3jp) * tmp2))
					- dt * ty1 * (-(r43 * c34 * tmp2 * u2jp));
			b[25*(i - 1) + 2 + 5*1] = dt * ty2 * (-(c2 * (u1jp * tmp1)));
			b[25*(i - 1) + 2 + 5*2] = dt * ty2 * ((2.0e+00 - c2) * (u2jp * tmp1))
					- dt * ty1 * (r43 * c34 * tmp1) - dt * ty1 * dy3;
			b[25*(i - 1) + 2 + 5*3] = dt * ty2 * (-(c2 * (u3jp * tmp1)));
			b[25*(i - 1) + 2 + 5*4] = dt * ty2 * c2;
			b[25*(i - 1) + 3] = dt * ty2 * (-((u2jp * u3jp) * tmp2))
					- dt * ty1 * (-(c34 * tmp2 * u3jp));
			b[25*(i - 1) + 3 + 5*1] = 0.0e+00;
			b[25*(i - 1) + 3 + 5*2] = dt * ty2 * (u3jp * tmp1);
			b[25*(i - 1) + 3 + 5*3] = dt * ty2 * (u2jp * tmp1) - dt * ty1 * (c34 * tmp1) - dt * ty1 * dy4;
			b[25*(i - 1) + 3 + 5*4] = 0.0e+00;
			b[25*(i - 1) + 4] = dt * ty2 * ((c2 * (u1jp * u1jp
					+ u2jp * u2jp + u3jp * u3jp) * tmp2
					- c1 * (u4jp * tmp1)) * (u2jp * tmp1)) - dt * ty1 * (-((c34
					- c1345) * tmp3 * (u1jp*u1jp)) - (r43 * c34 - c1345) * tmp3 * (u2jp*u2jp) - (c34
					- c1345) * tmp3 * (u3jp*u3jp) - c1345 * tmp2 * u4jp);
			b[25*(i - 1) + 4 + 5*1] = dt * ty2 * (-(c2 * (u1jp * u2jp) * tmp2))
					- dt * ty1 * (c34 - c1345) * tmp2 * u1jp;
			b[25*(i - 1) + 4 + 5*2] = dt * ty2 * (c1 * (u4jp * tmp1)
					- 0.50e+00 * c2 * ((u1jp * u1jp
					+ 3.0e+00 * u2jp * u2jp
					+ u3jp * u3jp) * tmp2)) - dt * ty1 * (r43 * c34
					- c1345) * tmp2 * u2jp;
			b[25*(i - 1) + 4 + 5*3] = dt * ty2 * (-(c2 * (u2jp * u3jp) * tmp2))
					- dt * ty1 * (c34 - c1345) * tmp2 * u3jp;
			b[25*(i - 1) + 4 + 5*4] = dt * ty2 * (c1 * (u2jp * tmp1)) - dt * ty1 * c1345 * tmp1
					- dt * ty1 * dy5;
			tmp1 = 1.0e+00 / u0kp;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			c[25*(i - 1)] = -(dt * tz1 * dz1);
			c[25*(i - 1) + 5*1] = 0.0e+00;
			c[25*(i - 1) + 5*2] = 0.0e+00;
			c[25*(i - 1) + 5*3] = dt * tz2;
			c[25*(i - 1) + 5*4] = 0.0e+00;
			c[25*(i - 1) + 1] = dt * tz2 * (-((u1kp * u3kp) * tmp2))
					- dt * tz1 * (-(c34 * tmp2 * u1kp));
			c[25*(i - 1) + 1 + 5*1] = dt * tz2 * (u3kp * tmp1) - dt * tz1 * c34 * tmp1 - dt * tz1 * dz2;
			c[25*(i - 1) + 1 + 5*2] = 0.0e+00;
			c[25*(i - 1) + 1 + 5*3] = dt * tz2 * (u1kp * tmp1);
			c[25*(i - 1) + 1 + 5*4] = 0.0e+00;
			c[25*(i - 1) + 2] = dt * tz2 * (-((u2kp * u3kp) * tmp2))
					- dt * tz1 * (-(c34 * tmp2 * u2kp));
			c[25*(i - 1) + 2 + 5*1] = 0.0e+00;
			c[25*(i - 1) + 2 + 5*2] = dt * tz2 * (u3kp * tmp1) - dt * tz1 * (c34 * tmp1) - dt * tz1 * dz3;
			c[25*(i - 1) + 2 + 5*3] = dt * tz2 * (u2kp * tmp1);
			c[25*(i - 1) + 2 + 5*4] = 0.0e+00;
			c[25*(i - 1) + 3] = dt * tz2 * (-sq((u3kp * tmp1))
					+ 0.50e+00 * c2 * ((u1kp * u1kp + u2kp * u2kp
					+ u3kp * u3kp) * tmp2))
					- dt * tz1 * (-(r43 * c34 * tmp2 * u3kp));
			c[25*(i - 1) + 3 + 5*1] = dt * tz2 * (-(c2 * (u1kp * tmp1)));
			c[25*(i - 1) + 3 + 5*2] = dt * tz2 * (-(c2 * (u2kp * tmp1)));
			c[25*(i - 1) + 3 + 5*3] = dt * tz2 * (2.0e+00 - c2) * (u3kp * tmp1)
					- dt * tz1 * (r43 * c34 * tmp1) - dt * tz1 * dz4;
			c[25*(i - 1) + 3 + 5*4] = dt * tz2 * c2;
			c[25*(i - 1) + 4] = dt * tz2 * ((c2 * (u1kp * u1kp
					+ u2kp * u2kp + u3kp * u3kp) * tmp2
					- c1 * (u4kp * tmp1)) * (u3kp * tmp1)) - dt * tz1 * (-((c34
					- c1345) * tmp3 * (u1kp*u1kp)) - (c34 - c1345) * tmp3 * (u2kp*u2kp) - (r43 * c34
					- c1345) * tmp3 * (u3kp*u3kp) - c1345 * tmp2 * u4kp);
			c[25*(i - 1) + 4 + 5*1] = dt * tz2 * (-(c2 * (u1kp * u3kp) * tmp2))
					- dt * tz1 * (c34 - c1345) * tmp2 * u1kp;
			c[25*(i - 1) + 4 + 5*2] = dt * tz2 * (-(c2 * (u2kp * u3kp) * tmp2))
					- dt * tz1 * (c34 - c1345) * tmp2 * u2kp;
			c[25*(i - 1) + 4 + 5*3] = dt * tz2 * (c1 * (u4kp * tmp1)
					- 0.50e+00 * c2 * ((u1kp * u1kp + u2kp * u2kp
					+ 3.0e+00 * u3kp * u3kp) * tmp2)) - dt * tz1 * (r43 * c34
					- c1345) * tmp2 * u3kp;
			c[25*(i - 1) + 4 + 5*4] = dt * tz2 * (c1 * (u3kp * tmp1)) - dt * tz1 * c1345 * tmp1
					- dt * tz1 * dz5;
		}
	}

	/*
	 * Block lower triangular solution of the line (j,k): v = rsd; ldz, ldy, ldx = a, b, c
	 */
	private void blts(int j, int k){
		int i;
		double tmp, tmp1;
		/* Optimization: loops over m unrolled, and the block matrix and the vector held in local variables (same operations, same order); same as in SER */
		final double[] rsd = this.rsd;
		final double[] a = this.a, b = this.b, c = this.c, d = this.d;
		final double omega = this.omega;
		double tv0, tv1, tv2, tv3, tv4;
		double t00, t01, t02, t03, t04, t10, t11, t12, t13, t14, t20, t21, t22, t23, t24, t30, t31, t32, t33, t34, t40, t41, t42, t43, t44;
		for (i = ist; i <= iend; i++){
			tv0 = rsd[ix(i,j,k)+0] - omega * ( a[25*(i-1)+0+5*0] * rsd[ix(i,j,k-1)+0] + a[25*(i-1)+0+5*1] * rsd[ix(i,j,k-1)+1] + a[25*(i-1)+0+5*2] * rsd[ix(i,j,k-1)+2] + a[25*(i-1)+0+5*3] * rsd[ix(i,j,k-1)+3] + a[25*(i-1)+0+5*4] * rsd[ix(i,j,k-1)+4] );
			tv0 = tv0 - omega * ( b[25*(i-1)+0+5*0] * rsd[ix(i,j-1,k)+0] + c[25*(i-1)+0+5*0] * rsd[ix(i-1,j,k)+0] + b[25*(i-1)+0+5*1] * rsd[ix(i,j-1,k)+1] + c[25*(i-1)+0+5*1] * rsd[ix(i-1,j,k)+1] + b[25*(i-1)+0+5*2] * rsd[ix(i,j-1,k)+2] + c[25*(i-1)+0+5*2] * rsd[ix(i-1,j,k)+2] + b[25*(i-1)+0+5*3] * rsd[ix(i,j-1,k)+3] + c[25*(i-1)+0+5*3] * rsd[ix(i-1,j,k)+3] + b[25*(i-1)+0+5*4] * rsd[ix(i,j-1,k)+4] + c[25*(i-1)+0+5*4] * rsd[ix(i-1,j,k)+4] );
			tv1 = rsd[ix(i,j,k)+1] - omega * ( a[25*(i-1)+1+5*0] * rsd[ix(i,j,k-1)+0] + a[25*(i-1)+1+5*1] * rsd[ix(i,j,k-1)+1] + a[25*(i-1)+1+5*2] * rsd[ix(i,j,k-1)+2] + a[25*(i-1)+1+5*3] * rsd[ix(i,j,k-1)+3] + a[25*(i-1)+1+5*4] * rsd[ix(i,j,k-1)+4] );
			tv1 = tv1 - omega * ( b[25*(i-1)+1+5*0] * rsd[ix(i,j-1,k)+0] + c[25*(i-1)+1+5*0] * rsd[ix(i-1,j,k)+0] + b[25*(i-1)+1+5*1] * rsd[ix(i,j-1,k)+1] + c[25*(i-1)+1+5*1] * rsd[ix(i-1,j,k)+1] + b[25*(i-1)+1+5*2] * rsd[ix(i,j-1,k)+2] + c[25*(i-1)+1+5*2] * rsd[ix(i-1,j,k)+2] + b[25*(i-1)+1+5*3] * rsd[ix(i,j-1,k)+3] + c[25*(i-1)+1+5*3] * rsd[ix(i-1,j,k)+3] + b[25*(i-1)+1+5*4] * rsd[ix(i,j-1,k)+4] + c[25*(i-1)+1+5*4] * rsd[ix(i-1,j,k)+4] );
			tv2 = rsd[ix(i,j,k)+2] - omega * ( a[25*(i-1)+2+5*0] * rsd[ix(i,j,k-1)+0] + a[25*(i-1)+2+5*1] * rsd[ix(i,j,k-1)+1] + a[25*(i-1)+2+5*2] * rsd[ix(i,j,k-1)+2] + a[25*(i-1)+2+5*3] * rsd[ix(i,j,k-1)+3] + a[25*(i-1)+2+5*4] * rsd[ix(i,j,k-1)+4] );
			tv2 = tv2 - omega * ( b[25*(i-1)+2+5*0] * rsd[ix(i,j-1,k)+0] + c[25*(i-1)+2+5*0] * rsd[ix(i-1,j,k)+0] + b[25*(i-1)+2+5*1] * rsd[ix(i,j-1,k)+1] + c[25*(i-1)+2+5*1] * rsd[ix(i-1,j,k)+1] + b[25*(i-1)+2+5*2] * rsd[ix(i,j-1,k)+2] + c[25*(i-1)+2+5*2] * rsd[ix(i-1,j,k)+2] + b[25*(i-1)+2+5*3] * rsd[ix(i,j-1,k)+3] + c[25*(i-1)+2+5*3] * rsd[ix(i-1,j,k)+3] + b[25*(i-1)+2+5*4] * rsd[ix(i,j-1,k)+4] + c[25*(i-1)+2+5*4] * rsd[ix(i-1,j,k)+4] );
			tv3 = rsd[ix(i,j,k)+3] - omega * ( a[25*(i-1)+3+5*0] * rsd[ix(i,j,k-1)+0] + a[25*(i-1)+3+5*1] * rsd[ix(i,j,k-1)+1] + a[25*(i-1)+3+5*2] * rsd[ix(i,j,k-1)+2] + a[25*(i-1)+3+5*3] * rsd[ix(i,j,k-1)+3] + a[25*(i-1)+3+5*4] * rsd[ix(i,j,k-1)+4] );
			tv3 = tv3 - omega * ( b[25*(i-1)+3+5*0] * rsd[ix(i,j-1,k)+0] + c[25*(i-1)+3+5*0] * rsd[ix(i-1,j,k)+0] + b[25*(i-1)+3+5*1] * rsd[ix(i,j-1,k)+1] + c[25*(i-1)+3+5*1] * rsd[ix(i-1,j,k)+1] + b[25*(i-1)+3+5*2] * rsd[ix(i,j-1,k)+2] + c[25*(i-1)+3+5*2] * rsd[ix(i-1,j,k)+2] + b[25*(i-1)+3+5*3] * rsd[ix(i,j-1,k)+3] + c[25*(i-1)+3+5*3] * rsd[ix(i-1,j,k)+3] + b[25*(i-1)+3+5*4] * rsd[ix(i,j-1,k)+4] + c[25*(i-1)+3+5*4] * rsd[ix(i-1,j,k)+4] );
			tv4 = rsd[ix(i,j,k)+4] - omega * ( a[25*(i-1)+4+5*0] * rsd[ix(i,j,k-1)+0] + a[25*(i-1)+4+5*1] * rsd[ix(i,j,k-1)+1] + a[25*(i-1)+4+5*2] * rsd[ix(i,j,k-1)+2] + a[25*(i-1)+4+5*3] * rsd[ix(i,j,k-1)+3] + a[25*(i-1)+4+5*4] * rsd[ix(i,j,k-1)+4] );
			tv4 = tv4 - omega * ( b[25*(i-1)+4+5*0] * rsd[ix(i,j-1,k)+0] + c[25*(i-1)+4+5*0] * rsd[ix(i-1,j,k)+0] + b[25*(i-1)+4+5*1] * rsd[ix(i,j-1,k)+1] + c[25*(i-1)+4+5*1] * rsd[ix(i-1,j,k)+1] + b[25*(i-1)+4+5*2] * rsd[ix(i,j-1,k)+2] + c[25*(i-1)+4+5*2] * rsd[ix(i-1,j,k)+2] + b[25*(i-1)+4+5*3] * rsd[ix(i,j-1,k)+3] + c[25*(i-1)+4+5*3] * rsd[ix(i-1,j,k)+3] + b[25*(i-1)+4+5*4] * rsd[ix(i,j-1,k)+4] + c[25*(i-1)+4+5*4] * rsd[ix(i-1,j,k)+4] );
			t00 = d[25*(i-1)+0+5*0];
			t01 = d[25*(i-1)+0+5*1];
			t02 = d[25*(i-1)+0+5*2];
			t03 = d[25*(i-1)+0+5*3];
			t04 = d[25*(i-1)+0+5*4];
			t10 = d[25*(i-1)+1+5*0];
			t11 = d[25*(i-1)+1+5*1];
			t12 = d[25*(i-1)+1+5*2];
			t13 = d[25*(i-1)+1+5*3];
			t14 = d[25*(i-1)+1+5*4];
			t20 = d[25*(i-1)+2+5*0];
			t21 = d[25*(i-1)+2+5*1];
			t22 = d[25*(i-1)+2+5*2];
			t23 = d[25*(i-1)+2+5*3];
			t24 = d[25*(i-1)+2+5*4];
			t30 = d[25*(i-1)+3+5*0];
			t31 = d[25*(i-1)+3+5*1];
			t32 = d[25*(i-1)+3+5*2];
			t33 = d[25*(i-1)+3+5*3];
			t34 = d[25*(i-1)+3+5*4];
			t40 = d[25*(i-1)+4+5*0];
			t41 = d[25*(i-1)+4+5*1];
			t42 = d[25*(i-1)+4+5*2];
			t43 = d[25*(i-1)+4+5*3];
			t44 = d[25*(i-1)+4+5*4];
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
			rsd[ix(i,j,k)+4] = tv4 / t44;
			tv3 = tv3 - t34 * rsd[ix(i,j,k)+4];
			rsd[ix(i,j,k)+3] = tv3 / t33;
			tv2 = tv2 - t23 * rsd[ix(i,j,k)+3] - t24 * rsd[ix(i,j,k)+4];
			rsd[ix(i,j,k)+2] = tv2 / t22;
			tv1 = tv1 - t12 * rsd[ix(i,j,k)+2] - t13 * rsd[ix(i,j,k)+3] - t14 * rsd[ix(i,j,k)+4];
			rsd[ix(i,j,k)+1] = tv1 / t11;
			tv0 = tv0 - t01 * rsd[ix(i,j,k)+1] - t02 * rsd[ix(i,j,k)+2] - t03 * rsd[ix(i,j,k)+3] - t04 * rsd[ix(i,j,k)+4];
			rsd[ix(i,j,k)+0] = tv0 / t00;
		}
	}

	/*
	 * Block upper triangular solution of the line (j,k): v = rsd; udx, udy, udz = a, b, c
	 */
	private void buts(int j, int k){
		int i;
		double tmp, tmp1;
		/* Optimization: loops over m unrolled, and the block matrix and the vector held in local variables (same operations, same order); same as in SER */
		final double[] rsd = this.rsd;
		final double[] a = this.a, b = this.b, c = this.c, d = this.d;
		final double omega = this.omega;
		double tv0, tv1, tv2, tv3, tv4;
		double t00, t01, t02, t03, t04, t10, t11, t12, t13, t14, t20, t21, t22, t23, t24, t30, t31, t32, t33, t34, t40, t41, t42, t43, t44;
		for (i = iend; i >= ist; i--){
			tv0 = omega * ( c[25*(i-1)+0+5*0] * rsd[ix(i,j,k+1)+0] + c[25*(i-1)+0+5*1] * rsd[ix(i,j,k+1)+1] + c[25*(i-1)+0+5*2] * rsd[ix(i,j,k+1)+2] + c[25*(i-1)+0+5*3] * rsd[ix(i,j,k+1)+3] + c[25*(i-1)+0+5*4] * rsd[ix(i,j,k+1)+4] );
			tv0 = tv0 + omega * ( b[25*(i-1)+0+5*0] * rsd[ix(i,j+1,k)+0] + a[25*(i-1)+0+5*0] * rsd[ix(i+1,j,k)+0] + b[25*(i-1)+0+5*1] * rsd[ix(i,j+1,k)+1] + a[25*(i-1)+0+5*1] * rsd[ix(i+1,j,k)+1] + b[25*(i-1)+0+5*2] * rsd[ix(i,j+1,k)+2] + a[25*(i-1)+0+5*2] * rsd[ix(i+1,j,k)+2] + b[25*(i-1)+0+5*3] * rsd[ix(i,j+1,k)+3] + a[25*(i-1)+0+5*3] * rsd[ix(i+1,j,k)+3] + b[25*(i-1)+0+5*4] * rsd[ix(i,j+1,k)+4] + a[25*(i-1)+0+5*4] * rsd[ix(i+1,j,k)+4] );
			tv1 = omega * ( c[25*(i-1)+1+5*0] * rsd[ix(i,j,k+1)+0] + c[25*(i-1)+1+5*1] * rsd[ix(i,j,k+1)+1] + c[25*(i-1)+1+5*2] * rsd[ix(i,j,k+1)+2] + c[25*(i-1)+1+5*3] * rsd[ix(i,j,k+1)+3] + c[25*(i-1)+1+5*4] * rsd[ix(i,j,k+1)+4] );
			tv1 = tv1 + omega * ( b[25*(i-1)+1+5*0] * rsd[ix(i,j+1,k)+0] + a[25*(i-1)+1+5*0] * rsd[ix(i+1,j,k)+0] + b[25*(i-1)+1+5*1] * rsd[ix(i,j+1,k)+1] + a[25*(i-1)+1+5*1] * rsd[ix(i+1,j,k)+1] + b[25*(i-1)+1+5*2] * rsd[ix(i,j+1,k)+2] + a[25*(i-1)+1+5*2] * rsd[ix(i+1,j,k)+2] + b[25*(i-1)+1+5*3] * rsd[ix(i,j+1,k)+3] + a[25*(i-1)+1+5*3] * rsd[ix(i+1,j,k)+3] + b[25*(i-1)+1+5*4] * rsd[ix(i,j+1,k)+4] + a[25*(i-1)+1+5*4] * rsd[ix(i+1,j,k)+4] );
			tv2 = omega * ( c[25*(i-1)+2+5*0] * rsd[ix(i,j,k+1)+0] + c[25*(i-1)+2+5*1] * rsd[ix(i,j,k+1)+1] + c[25*(i-1)+2+5*2] * rsd[ix(i,j,k+1)+2] + c[25*(i-1)+2+5*3] * rsd[ix(i,j,k+1)+3] + c[25*(i-1)+2+5*4] * rsd[ix(i,j,k+1)+4] );
			tv2 = tv2 + omega * ( b[25*(i-1)+2+5*0] * rsd[ix(i,j+1,k)+0] + a[25*(i-1)+2+5*0] * rsd[ix(i+1,j,k)+0] + b[25*(i-1)+2+5*1] * rsd[ix(i,j+1,k)+1] + a[25*(i-1)+2+5*1] * rsd[ix(i+1,j,k)+1] + b[25*(i-1)+2+5*2] * rsd[ix(i,j+1,k)+2] + a[25*(i-1)+2+5*2] * rsd[ix(i+1,j,k)+2] + b[25*(i-1)+2+5*3] * rsd[ix(i,j+1,k)+3] + a[25*(i-1)+2+5*3] * rsd[ix(i+1,j,k)+3] + b[25*(i-1)+2+5*4] * rsd[ix(i,j+1,k)+4] + a[25*(i-1)+2+5*4] * rsd[ix(i+1,j,k)+4] );
			tv3 = omega * ( c[25*(i-1)+3+5*0] * rsd[ix(i,j,k+1)+0] + c[25*(i-1)+3+5*1] * rsd[ix(i,j,k+1)+1] + c[25*(i-1)+3+5*2] * rsd[ix(i,j,k+1)+2] + c[25*(i-1)+3+5*3] * rsd[ix(i,j,k+1)+3] + c[25*(i-1)+3+5*4] * rsd[ix(i,j,k+1)+4] );
			tv3 = tv3 + omega * ( b[25*(i-1)+3+5*0] * rsd[ix(i,j+1,k)+0] + a[25*(i-1)+3+5*0] * rsd[ix(i+1,j,k)+0] + b[25*(i-1)+3+5*1] * rsd[ix(i,j+1,k)+1] + a[25*(i-1)+3+5*1] * rsd[ix(i+1,j,k)+1] + b[25*(i-1)+3+5*2] * rsd[ix(i,j+1,k)+2] + a[25*(i-1)+3+5*2] * rsd[ix(i+1,j,k)+2] + b[25*(i-1)+3+5*3] * rsd[ix(i,j+1,k)+3] + a[25*(i-1)+3+5*3] * rsd[ix(i+1,j,k)+3] + b[25*(i-1)+3+5*4] * rsd[ix(i,j+1,k)+4] + a[25*(i-1)+3+5*4] * rsd[ix(i+1,j,k)+4] );
			tv4 = omega * ( c[25*(i-1)+4+5*0] * rsd[ix(i,j,k+1)+0] + c[25*(i-1)+4+5*1] * rsd[ix(i,j,k+1)+1] + c[25*(i-1)+4+5*2] * rsd[ix(i,j,k+1)+2] + c[25*(i-1)+4+5*3] * rsd[ix(i,j,k+1)+3] + c[25*(i-1)+4+5*4] * rsd[ix(i,j,k+1)+4] );
			tv4 = tv4 + omega * ( b[25*(i-1)+4+5*0] * rsd[ix(i,j+1,k)+0] + a[25*(i-1)+4+5*0] * rsd[ix(i+1,j,k)+0] + b[25*(i-1)+4+5*1] * rsd[ix(i,j+1,k)+1] + a[25*(i-1)+4+5*1] * rsd[ix(i+1,j,k)+1] + b[25*(i-1)+4+5*2] * rsd[ix(i,j+1,k)+2] + a[25*(i-1)+4+5*2] * rsd[ix(i+1,j,k)+2] + b[25*(i-1)+4+5*3] * rsd[ix(i,j+1,k)+3] + a[25*(i-1)+4+5*3] * rsd[ix(i+1,j,k)+3] + b[25*(i-1)+4+5*4] * rsd[ix(i,j+1,k)+4] + a[25*(i-1)+4+5*4] * rsd[ix(i+1,j,k)+4] );
			t00 = d[25*(i-1)+0+5*0];
			t01 = d[25*(i-1)+0+5*1];
			t02 = d[25*(i-1)+0+5*2];
			t03 = d[25*(i-1)+0+5*3];
			t04 = d[25*(i-1)+0+5*4];
			t10 = d[25*(i-1)+1+5*0];
			t11 = d[25*(i-1)+1+5*1];
			t12 = d[25*(i-1)+1+5*2];
			t13 = d[25*(i-1)+1+5*3];
			t14 = d[25*(i-1)+1+5*4];
			t20 = d[25*(i-1)+2+5*0];
			t21 = d[25*(i-1)+2+5*1];
			t22 = d[25*(i-1)+2+5*2];
			t23 = d[25*(i-1)+2+5*3];
			t24 = d[25*(i-1)+2+5*4];
			t30 = d[25*(i-1)+3+5*0];
			t31 = d[25*(i-1)+3+5*1];
			t32 = d[25*(i-1)+3+5*2];
			t33 = d[25*(i-1)+3+5*3];
			t34 = d[25*(i-1)+3+5*4];
			t40 = d[25*(i-1)+4+5*0];
			t41 = d[25*(i-1)+4+5*1];
			t42 = d[25*(i-1)+4+5*2];
			t43 = d[25*(i-1)+4+5*3];
			t44 = d[25*(i-1)+4+5*4];
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
			rsd[ix(i,j,k)+0] = rsd[ix(i,j,k)+0] - tv0;
			rsd[ix(i,j,k)+1] = rsd[ix(i,j,k)+1] - tv1;
			rsd[ix(i,j,k)+2] = rsd[ix(i,j,k)+2] - tv2;
			rsd[ix(i,j,k)+3] = rsd[ix(i,j,k)+3] - tv3;
			rsd[ix(i,j,k)+4] = rsd[ix(i,j,k)+4] - tv4;
		}
	}

	/*
	 * Pseudo-time stepping SSOR iterations for the five nonlinear pde's
	 */
	private void ssor(int niter){

		int i, j, k, m;
		int istep;
		double tmp;
		double delunm[] = new double[5];

		tmp = 1.0 / ( omega * ( 2.0 - omega ) );

		/* initialize a,b,c,d to zero (guarantees that page tables have been formed) */
		for (i = 0; i < 25*isiz1; i++){
			a[i] = 0.0;
			b[i] = 0.0;
			c[i] = 0.0;
			d[i] = 0.0;
		}

		/* the steady-state residuals and their L2 norms */
		rhs();

		l2norm(rsdnm);

		timer.reset(T_COMP);
		timer.reset(T_COMM);

		commWork.Barrier();

		timer.start(T_COMP);

		/* the timestep loop */
		for (istep = 1; istep <= niter; istep++){

			if (id == 0){
				if (istep % 20 == 0 || istep == itmax || istep == 1){
					if (niter > 1)
						System.out.println(" Time step "+istep);
				}
			}

			/* perform SSOR iteration */
			for (k = 2; k <= nz - 1; k++){
				for (j = jst; j <= jend; j++){
					for (i = ist; i <= iend; i++){
						for (m = 0; m < 5; m++){
							rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))+m] = dt * rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))+m];
						}
					}
				}
			}

			for (k = 2; k <= nz - 1; k++){
				/* receive data from north and west */
				exchange1(rsd, k, 0);

				for (j = jst; j <= jend; j++){
					/* form the lower triangular part of the jacobian matrix */
					jacld(j, k);

					/* perform the lower triangular solution */
					blts(j, k);
				}

				/* send data to east and south */
				exchange1(rsd, k, 2);
			}

			for (k = nz - 1; k >= 2; k--){
				/* receive data from south and east */
				exchange1(rsd, k, 1);

				for (j = jend; j >= jst; j--){
					/* form the strictly upper triangular part of the jacobian matrix */
					jacu(j, k);

					/* perform the upper triangular solution */
					buts(j, k);
				}

				/* send data to north and west */
				exchange1(rsd, k, 3);
			}

			/* update the variables */
			for (k = 2; k <= nz - 1; k++){
				for (j = jst; j <= jend; j++){
					for (i = ist; i <= iend; i++){
						for (m = 0; m < 5; m++){
							u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))+m] = u[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))+m] + tmp * rsd[(5*(((i)+1) + ldx*(((j)+1) + ldy*((k)-1))))+m];
						}
					}
				}
			}

			/* the norms of the newton iteration corrections */
			if (istep % inorm == 0){
				l2norm(delunm);
			}

			/* the steady-state residuals */
			rhs();

			/* the norms of the newton iteration residuals */
			if (istep % inorm == 0 || istep == itmax){
				l2norm(rsdnm);
			}

			/* check the newton-iteration residuals against the tolerance levels */
			if (rsdnm[0] < tolrsd[0] && rsdnm[1] < tolrsd[1] && rsdnm[2] < tolrsd[2]
					&& rsdnm[3] < tolrsd[3] && rsdnm[4] < tolrsd[4]){
				if (id == 0){
					System.out.println(" convergence was achieved after "+istep+" pseudo-time steps");
				}
				break;
			}
		}

		timer.stop(T_COMP);

		double wtime[] = {timer.read(T_COMP), timer.read(T_COMM)};
		double wmax[] = new double[2];

		commWork.Allreduce(wtime, 0, wmax, 0, 2, MPI.DOUBLE, MPI.MAX);

		maxtime = wmax[0];
		maxcomm = wmax[1];
	}

	/* Exchanges of the SSOR sweeps: the plane k of g with the neighbors (blocking, wavefront) */
	private void exchange1(double g[], int k, int iex){

		int i, j, m;

		if(detailedTiming)
			timer.start(T_COMM);

		if(iex == 0){

			if(north != -1){
				commWork.Recv(buf1, 5*(jst-1), 5*(jend-jst+1), MPI.DOUBLE, north, FROM_N);
				for(j = jst; j <= jend; j++){
					for(m = 0; m<5; m++)
						g[ix(0,j,k)+m] = buf1[5*(j-1)+m];
				}
			}

			if(west != -1){
				commWork.Recv(buf1, 5*(ist-1), 5*(iend-ist+1), MPI.DOUBLE, west, FROM_W);
				for(i = ist; i <= iend; i++){
					for(m = 0; m<5; m++)
						g[ix(i,0,k)+m] = buf1[5*(i-1)+m];
				}
			}
		}
		else if(iex == 1){

			if(south != -1){
				commWork.Recv(buf1, 5*(jst-1), 5*(jend-jst+1), MPI.DOUBLE, south, FROM_S);
				for(j = jst; j <= jend; j++){
					for(m = 0; m<5; m++)
						g[ix(nx+1,j,k)+m] = buf1[5*(j-1)+m];
				}
			}

			if(east != -1){
				commWork.Recv(buf1, 5*(ist-1), 5*(iend-ist+1), MPI.DOUBLE, east, FROM_E);
				for(i = ist; i <= iend; i++){
					for(m = 0; m<5; m++)
						g[ix(i,ny+1,k)+m] = buf1[5*(i-1)+m];
				}
			}
		}
		else if(iex == 2){

			if(south != -1){
				for(j = jst; j <= jend; j++){
					for(m = 0; m<5; m++)
						buf[5*(j-1)+m] = g[ix(nx,j,k)+m];
				}
				commWork.Send(buf, 5*(jst-1), 5*(jend-jst+1), MPI.DOUBLE, south, FROM_N);
			}

			if(east != -1){
				for(i = ist; i <= iend; i++){
					for(m = 0; m<5; m++)
						buf[5*(i-1)+m] = g[ix(i,ny,k)+m];
				}
				commWork.Send(buf, 5*(ist-1), 5*(iend-ist+1), MPI.DOUBLE, east, FROM_W);
			}
		}
		else{

			if(north != -1){
				for(j = jst; j <= jend; j++){
					for(m = 0; m<5; m++)
						buf[5*(j-1)+m] = g[ix(1,j,k)+m];
				}
				commWork.Send(buf, 5*(jst-1), 5*(jend-jst+1), MPI.DOUBLE, north, FROM_S);
			}

			if(west != -1){
				for(i = ist; i <= iend; i++){
					for(m = 0; m<5; m++)
						buf[5*(i-1)+m] = g[ix(i,1,k)+m];
				}
				commWork.Send(buf, 5*(ist-1), 5*(iend-ist+1), MPI.DOUBLE, west, FROM_E);
			}
		}

		if(detailedTiming)
			timer.stop(T_COMM);
	}

	/* Exchange of the two boundary planes of g (rhs and erhs): iex = 0 in x (north and south), otherwise in y (west and east) */
	private void exchange3(double g[], int iex){

		int i, j, k, m;
		int ipos1, ipos2;
		Request msgid2 = null;

		if(detailedTiming)
			timer.start(T_COMM);

		if(iex == 0){

			/* communicate in the south and north directions */
			if(north != -1)
				msgid2 = commWork.Irecv(buf1, 0, 10*ny*nz, MPI.DOUBLE, north, FROM_N);

			/* send south */
			if(south != -1){
				for(k = 1; k <= nz; k++){
					for(j = 1; j <= ny; j++){
						ipos1 = (k-1)*ny + j;
						ipos2 = ipos1 + ny*nz;
						for(m = 0; m<5; m++){
							buf[5*(ipos1-1)+m] = g[ix(nx-1,j,k)+m];
							buf[5*(ipos2-1)+m] = g[ix(nx,j,k)+m];
						}
					}
				}
				commWork.Send(buf, 0, 10*ny*nz, MPI.DOUBLE, south, FROM_N);
			}

			/* receive from north */
			if(north != -1){
				msgid2.Wait();

				for(k = 1; k <= nz; k++){
					for(j = 1; j <= ny; j++){
						ipos1 = (k-1)*ny + j;
						ipos2 = ipos1 + ny*nz;
						for(m = 0; m<5; m++){
							g[ix(-1,j,k)+m] = buf1[5*(ipos1-1)+m];
							g[ix(0,j,k)+m] = buf1[5*(ipos2-1)+m];
						}
					}
				}
			}

			if(south != -1)
				msgid2 = commWork.Irecv(buf1, 0, 10*ny*nz, MPI.DOUBLE, south, FROM_S);

			/* send north */
			if(north != -1){
				for(k = 1; k <= nz; k++){
					for(j = 1; j <= ny; j++){
						ipos1 = (k-1)*ny + j;
						ipos2 = ipos1 + ny*nz;
						for(m = 0; m<5; m++){
							buf[5*(ipos1-1)+m] = g[ix(2,j,k)+m];
							buf[5*(ipos2-1)+m] = g[ix(1,j,k)+m];
						}
					}
				}
				commWork.Send(buf, 0, 10*ny*nz, MPI.DOUBLE, north, FROM_S);
			}

			/* receive from south */
			if(south != -1){
				msgid2.Wait();

				for(k = 1; k <= nz; k++){
					for(j = 1; j <= ny; j++){
						ipos1 = (k-1)*ny + j;
						ipos2 = ipos1 + ny*nz;
						for(m = 0; m<5; m++){
							g[ix(nx+2,j,k)+m] = buf1[5*(ipos1-1)+m];
							g[ix(nx+1,j,k)+m] = buf1[5*(ipos2-1)+m];
						}
					}
				}
			}
		}
		else{

			/* communicate in the east and west directions */
			if(west != -1)
				msgid2 = commWork.Irecv(buf1, 0, 10*nx*nz, MPI.DOUBLE, west, FROM_W);

			/* send east */
			if(east != -1){
				for(k = 1; k <= nz; k++){
					for(i = 1; i <= nx; i++){
						ipos1 = (k-1)*nx + i;
						ipos2 = ipos1 + nx*nz;
						for(m = 0; m<5; m++){
							buf[5*(ipos1-1)+m] = g[ix(i,ny-1,k)+m];
							buf[5*(ipos2-1)+m] = g[ix(i,ny,k)+m];
						}
					}
				}
				commWork.Send(buf, 0, 10*nx*nz, MPI.DOUBLE, east, FROM_W);
			}

			/* receive from west */
			if(west != -1){
				msgid2.Wait();

				for(k = 1; k <= nz; k++){
					for(i = 1; i <= nx; i++){
						ipos1 = (k-1)*nx + i;
						ipos2 = ipos1 + nx*nz;
						for(m = 0; m<5; m++){
							g[ix(i,-1,k)+m] = buf1[5*(ipos1-1)+m];
							g[ix(i,0,k)+m] = buf1[5*(ipos2-1)+m];
						}
					}
				}
			}

			if(east != -1)
				msgid2 = commWork.Irecv(buf1, 0, 10*nx*nz, MPI.DOUBLE, east, FROM_E);

			/* send west */
			if(west != -1){
				for(k = 1; k <= nz; k++){
					for(i = 1; i <= nx; i++){
						ipos1 = (k-1)*nx + i;
						ipos2 = ipos1 + nx*nz;
						for(m = 0; m<5; m++){
							buf[5*(ipos1-1)+m] = g[ix(i,2,k)+m];
							buf[5*(ipos2-1)+m] = g[ix(i,1,k)+m];
						}
					}
				}
				commWork.Send(buf, 0, 10*nx*nz, MPI.DOUBLE, west, FROM_E);
			}

			/* receive from east */
			if(east != -1){
				msgid2.Wait();

				for(k = 1; k <= nz; k++){
					for(i = 1; i <= nx; i++){
						ipos1 = (k-1)*nx + i;
						ipos2 = ipos1 + nx*nz;
						for(m = 0; m<5; m++){
							g[ix(i,ny+2,k)+m] = buf1[5*(ipos1-1)+m];
							g[ix(i,ny+1,k)+m] = buf1[5*(ipos2-1)+m];
						}
					}
				}
			}
		}

		if(detailedTiming)
			timer.stop(T_COMM);
	}

	/* Exchange of two planes phi1 and phi2 of the surface integral, in the east-west and north-south directions (exchange_4) */
	private void exchange4(double g[][], double h[][], int ibeg, int ifin1, int jbeg, int jfin1){

		int i, j;
		int ny2 = ny + 2;
		Request msgid;

		if(detailedTiming)
			timer.start(T_COMM);

		/* communicate in the east and west directions: receive from east */
		if(jfin1 == ny){
			msgid = commWork.Irecv(dum, 0, 2*nx, MPI.DOUBLE, east, FROM_E);
			msgid.Wait();

			for(i = 1; i <= nx; i++){
				g[i][ny+1] = dum[i-1];
				h[i][ny+1] = dum[i+nx-1];
			}
		}

		/* send west */
		if(jbeg == 1){
			for(i = 1; i <= nx; i++){
				dum[i-1] = g[i][1];
				dum[i+nx-1] = h[i][1];
			}
			commWork.Send(dum, 0, 2*nx, MPI.DOUBLE, west, FROM_E);
		}

		/* communicate in the south and north directions: receive from south */
		if(ifin1 == nx){
			msgid = commWork.Irecv(dum, 0, 2*ny2, MPI.DOUBLE, south, FROM_S);
			msgid.Wait();

			for(j = 0; j <= ny+1; j++){
				g[nx+1][j] = dum[j];
				h[nx+1][j] = dum[j+ny2];
			}
		}

		/* send north */
		if(ibeg == 1){
			for(j = 0; j <= ny+1; j++){
				dum[j] = g[1][j];
				dum[j+ny2] = h[1][j];
			}
			commWork.Send(dum, 0, 2*ny2, MPI.DOUBLE, north, FROM_S);
		}

		if(detailedTiming)
			timer.stop(T_COMM);
	}

	/* Exchange of a plane of the surface integral in the north-south direction (exchange_5) */
	private void exchange5(double g[][], int ibeg, int ifin1){

		int k;
		Request msgid;

		if(detailedTiming)
			timer.start(T_COMM);

		/* receive from south */
		if(ifin1 == nx){
			msgid = commWork.Irecv(dum, 0, nz, MPI.DOUBLE, south, FROM_S);
			msgid.Wait();

			for(k = 1; k <= nz; k++)
				g[nx+1][k] = dum[k-1];
		}

		/* send north */
		if(ibeg == 1){
			for(k = 1; k <= nz; k++)
				dum[k-1] = g[1][k];
			commWork.Send(dum, 0, nz, MPI.DOUBLE, north, FROM_S);
		}

		if(detailedTiming)
			timer.stop(T_COMM);
	}

	/* Exchange of a plane of the surface integral in the east-west direction (exchange_6) */
	private void exchange6(double g[][], int jbeg, int jfin1){

		int k;
		Request msgid;

		if(detailedTiming)
			timer.start(T_COMM);

		/* receive from east */
		if(jfin1 == ny){
			msgid = commWork.Irecv(dum, 0, nz, MPI.DOUBLE, east, FROM_E);
			msgid.Wait();

			for(k = 1; k <= nz; k++)
				g[ny+1][k] = dum[k-1];
		}

		/* send west */
		if(jbeg == 1){
			for(k = 1; k <= nz; k++)
				dum[k-1] = g[1][k];
			commWork.Send(dum, 0, nz, MPI.DOUBLE, west, FROM_E);
		}

		if(detailedTiming)
			timer.stop(T_COMM);
	}

	/* Global sum of a double over the active processes */
	private double allreduceSum1(double v){
		double s[] = {v};
		double r[] = new double[1];
		if(detailedTiming)
			timer.start(T_COMM);
		commWork.Allreduce(s, 0, r, 0, 1, MPI.DOUBLE, MPI.SUM);
		if(detailedTiming)
			timer.stop(T_COMM);
		return r[0];
	}

	/* The L2-norm of the vector v (5 components) */
	private void l2norm(double sum[]){

		int i, j, k, m;
		double dummy[] = new double[5];

		for (m = 0; m < 5; m++)
			dummy[m] = 0.0;

		for (k = 2; k <= nz0 - 1; k++){
			for (j = jst; j <= jend; j++){
				for (i = ist; i <= iend; i++){
					for (m = 0; m < 5; m++){
						dummy[m] = dummy[m] + rsd[ix(i,j,k)+m] * rsd[ix(i,j,k)+m];
					}
				}
			}
		}

		/* the global sum of individual contributions to dot product */
		if(detailedTiming)
			timer.start(T_COMM);
		commWork.Allreduce(dummy, 0, sum, 0, 5, MPI.DOUBLE, MPI.SUM);
		if(detailedTiming)
			timer.stop(T_COMM);

		for (m = 0; m < 5; m++)
			sum[m] = Math.sqrt( sum[m] / ( (double)(nx0-2)*(ny0-2)*(nz0-2) ) );
	}

	/* The RMS-norm of the solution error */
	private void error(){

		int i, j, k, m;
		int iglob, jglob;
		double tmp;
		double u000ijk[] = new double[5], dummy[] = new double[5];

		for (m = 0; m < 5; m++){
			errnm[m] = 0.0;
			dummy[m] = 0.0;
		}

		for (k = 2; k <= nz - 1; k++){
			for (j = jst; j <= jend; j++){
				jglob = jpt + j;
				for (i = ist; i <= iend; i++){
					iglob = ipt + i;
					exact( iglob, jglob, k, u000ijk, 0 );
					for (m = 0; m < 5; m++){
						tmp = ( u000ijk[m] - u[ix(i,j,k)+m] );
						dummy[m] = dummy[m] + tmp * tmp;
					}
				}
			}
		}

		/* the global sum of individual contributions to dot product */
		if(detailedTiming)
			timer.start(T_COMM);
		commWork.Allreduce(dummy, 0, errnm, 0, 5, MPI.DOUBLE, MPI.SUM);
		if(detailedTiming)
			timer.stop(T_COMM);

		for (m = 0; m < 5; m++)
			errnm[m] = Math.sqrt( errnm[m] / ( (double)(nx0-2)*(ny0-2)*(nz0-2) ) );
	}

	/*
	 * The surface integral frc
	 */
	private void pintgr(){

		int i, j, k, ibeg, ifin, ifin1, jbeg, jfin;
		int jfin1, iglob, iglob1, iglob2, jglob, jglob1, jglob2, ind1;
		int ind2;
		double frc1, frc2, frc3, dummy;

		ibeg = nx + 1;
		ifin = 0;
		iglob1 = ipt + 1;
		iglob2 = ipt + nx;
		if (iglob1 >= ii1 && iglob2 < ii2 + nx) ibeg = 1;
		if (iglob1 > ii1 - nx && iglob2 <= ii2) ifin = nx;
		if (ii1 >= iglob1 && ii1 <= iglob2) ibeg = ii1 - ipt;
		if (ii2 >= iglob1 && ii2 <= iglob2) ifin = ii2 - ipt;
		jbeg = ny + 1;
		jfin = 0;
		jglob1 = jpt + 1;
		jglob2 = jpt + ny;
		if (jglob1 >= ji1 && jglob2 < ji2 + ny) jbeg = 1;
		if (jglob1 > ji1 - ny && jglob2 <= ji2) jfin = ny;
		if (ji1 >= jglob1 && ji1 <= jglob2) jbeg = ji1 - jpt;
		if (ji2 >= jglob1 && ji2 <= jglob2) jfin = ji2 - jpt;
		ifin1 = ifin;
		jfin1 = jfin;
		if (ipt + ifin1 == ii2) ifin1 = ifin - 1;
		if (jpt + jfin1 == ji2) jfin1 = jfin - 1;
		for (k = 0; k <= isiz3 + 1; k++){
			for (i = 0; i <= isiz2 + 1; i++){
				phi1[i][k] = 0.0;
				phi2[i][k] = 0.0;
			}
		}
		for (j = jbeg; j <= jfin; j++){
			jglob = jpt + j;
			for (i = ibeg; i <= ifin; i++){
				iglob = ipt + i;
				k = ki1;
				phi1[i][j] = c2 * (u[ix(i, j, k) + 4] - 0.50e+00 * (sq(u[ix(i, j, k) + 1]) + sq(u[ix(i, j, k) + 2])
						+ sq(u[ix(i, j, k) + 3])) / u[ix(i, j, k)]);
				k = ki2;
				phi2[i][j] = c2 * (u[ix(i, j, k) + 4] - 0.50e+00 * (sq(u[ix(i, j, k) + 1]) + sq(u[ix(i, j, k) + 2])
						+ sq(u[ix(i, j, k) + 3])) / u[ix(i, j, k)]);
			}
		}
		exchange4(phi1, phi2, ibeg, ifin1, jbeg, jfin1);
		frc1 = 0.0e+00;
		for (j = jbeg; j <= jfin1; j++){
			for (i = ibeg; i <= ifin1; i++){
				frc1 = frc1 + (phi1[i][j] + phi1[i + 1][j] + phi1[i][j + 1] + phi1[i + 1][j + 1] + phi2[i][j] + phi2[i + 1][j]
						+ phi2[i][j + 1] + phi2[i + 1][j + 1]);
			}
		}
		dummy = frc1;
		frc1 = allreduceSum1(dummy);
		frc1 = dxi * deta * frc1;
		for (k = 0; k <= isiz3 + 1; k++){
			for (i = 0; i <= isiz2 + 1; i++){
				phi1[i][k] = 0.0;
				phi2[i][k] = 0.0;
			}
		}
		jglob = jpt + jbeg;
		ind1 = 0;
		if (jglob == ji1){
			ind1 = 1;
			for (k = ki1; k <= ki2; k++){
				for (i = ibeg; i <= ifin; i++){
					iglob = ipt + i;
					phi1[i][k] = c2 * (u[ix(i, jbeg, k) + 4] - 0.50e+00 * (sq(u[ix(i, jbeg, k) + 1]) + sq(u[ix(i, jbeg, k) + 2])
							+ sq(u[ix(i, jbeg, k) + 3])) / u[ix(i, jbeg, k)]);
				}
			}
		}
		jglob = jpt + jfin;
		ind2 = 0;
		if (jglob == ji2){
			ind2 = 1;
			for (k = ki1; k <= ki2; k++){
				for (i = ibeg; i <= ifin; i++){
					iglob = ipt + i;
					phi2[i][k] = c2 * (u[ix(i, jfin, k) + 4] - 0.50e+00 * (sq(u[ix(i, jfin, k) + 1]) + sq(u[ix(i, jfin, k) + 2])
							+ sq(u[ix(i, jfin, k) + 3])) / u[ix(i, jfin, k)]);
				}
			}
		}
		if (ind1 == 1){
			exchange5(phi1, ibeg, ifin1);
		}
		if (ind2 == 1){
			exchange5(phi2, ibeg, ifin1);
		}
		frc2 = 0.0e+00;
		for (k = ki1; k <= ki2 - 1; k++){
			for (i = ibeg; i <= ifin1; i++){
				frc2 = frc2 + (phi1[i][k] + phi1[i + 1][k] + phi1[i][k + 1] + phi1[i + 1][k + 1] + phi2[i][k] + phi2[i + 1][k]
						+ phi2[i][k + 1] + phi2[i + 1][k + 1]);
			}
		}
		dummy = frc2;
		frc2 = allreduceSum1(dummy);
		frc2 = dxi * dzeta * frc2;
		for (k = 0; k <= isiz3 + 1; k++){
			for (i = 0; i <= isiz2 + 1; i++){
				phi1[i][k] = 0.0;
				phi2[i][k] = 0.0;
			}
		}
		iglob = ipt + ibeg;
		ind1 = 0;
		if (iglob == ii1){
			ind1 = 1;
			for (k = ki1; k <= ki2; k++){
				for (j = jbeg; j <= jfin; j++){
					jglob = jpt + j;
					phi1[j][k] = c2 * (u[ix(ibeg, j, k) + 4] - 0.50e+00 * (sq(u[ix(ibeg, j, k) + 1]) + sq(u[ix(ibeg, j, k) + 2])
							+ sq(u[ix(ibeg, j, k) + 3])) / u[ix(ibeg, j, k)]);
				}
			}
		}
		iglob = ipt + ifin;
		ind2 = 0;
		if (iglob == ii2){
			ind2 = 1;
			for (k = ki1; k <= ki2; k++){
				for (j = jbeg; j <= jfin; j++){
					jglob = jpt + j;
					phi2[j][k] = c2 * (u[ix(ifin, j, k) + 4] - 0.50e+00 * (sq(u[ix(ifin, j, k) + 1]) + sq(u[ix(ifin, j, k) + 2])
							+ sq(u[ix(ifin, j, k) + 3])) / u[ix(ifin, j, k)]);
				}
			}
		}
		if (ind1 == 1){
			exchange6(phi1, jbeg, jfin1);
		}
		if (ind2 == 1){
			exchange6(phi2, jbeg, jfin1);
		}
		frc3 = 0.0e+00;
		for (k = ki1; k <= ki2 - 1; k++){
			for (j = jbeg; j <= jfin1; j++){
				frc3 = frc3 + (phi1[j][k] + phi1[j + 1][k] + phi1[j][k + 1] + phi1[j + 1][k + 1] + phi2[j][k] + phi2[j + 1][k]
						+ phi2[j][k + 1] + phi2[j + 1][k + 1]);
			}
		}
		dummy = frc3;
		frc3 = allreduceSum1(dummy);
		frc3 = deta * dzeta * frc3;
		frc = 0.25e+00 * (frc1 + frc2 + frc3);
	}

	/* Verification of the norms and the surface integral (only the root process calls it) */
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
		else if(CLASS == 'E'){
			dtref = 0.5e+0;
			xcrref[0] = 0.2099641687874e+06;
			xcrref[1] = 0.2130403143165e+05;
			xcrref[2] = 0.5319228789371e+05;
			xcrref[3] = 0.4509761639833e+05;
			xcrref[4] = 0.2932360006590e+06;
			xceref[0] = 0.4800572578333e+03;
			xceref[1] = 0.4221993400184e+02;
			xceref[2] = 0.1210851906824e+03;
			xceref[3] = 0.1047888986770e+03;
			xceref[4] = 0.8363028257389e+03;
			xciref = 0.9512163272273e+02;
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
}
