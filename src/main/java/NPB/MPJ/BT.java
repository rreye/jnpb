/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB 3.4.4 MPI (NPB3.4-MPI/BT)
 */
package NPB.MPJ;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import mpi.Comm;
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

public class BT implements Benchmark{

	private static final String BM_NAME="BT";
	private char CLASS;
	private boolean detailedTiming;

	/* compiled values of the class (problem_size, niter_default, dt_default) */
	private int problemSize;
	private int niterDefault;
	private double dtDefault;

	private int niter;
	private double dt;

	private int gridPoints[] = new int[3];

	/*
	 * The cells of the multipartition (NPB-MPI): ncells = sqrt(processes)
	 * cells per process. The cell index c is 0-based here; cellCoord, cellLow
	 * and cellHigh keep the values of the original (cellCoord is 1-based),
	 * and slice[dir][n] is the 0-based number of the cell in plane n.
	 */
	private int ncells;
	private int cellCoord[][];
	private int cellHigh[][];
	private int cellLow[][];
	private int cellSize[][];
	private int slice[][];
	private int start[][];
	private int end[][];

	private int predecessor[] = new int[3];
	private int successor[] = new int[3];

	private int MAX_CELL_DIM;
	private int IMAX, JMAX, KMAX;
	private int MAXD1;				/* MAX_CELL_DIM+1: extent of backsub_info(5,0:MAX_CELL_DIM,0:MAX_CELL_DIM,cells) */
	private int DI, DJ, DK;		/* strides of the arrays with 2 ghost points per side: IMAX+4, ... */

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

	/*
	 * Field arrays. All the 5-component arrays have the components of a
	 * point contiguous: u(m,i,j,k,c) is u[5*ix(i,j,k,c)+m-1], with the
	 * indices of the original (ghost points included: -2..IMAX+1).
	 */
	private double u[];
	private double rhs[];
	private double forcing[];
	private double us[];
	private double vs[];
	private double ws[];
	private double qs[];
	private double rhoI[];
	private double square[];
	/* lhsc(5,5,i,j,k,c): [25*ix(i,j,k,c)+m+5*n] (first index fastest) */
	private double lhsc[];
	private double backsubInfo[];

	/* work arrays of one grid line (-2..MAX_CELL_DIM+1), fjac, njac: [25*(i+2)+m+5*n]; lhsa, lhsb (-1..MAX_CELL_DIM): [25*(i+1)+m+5*n] */
	private double cv[];
	private double cuf[];
	private double q[];
	private double ue[][];
	private double buf[][];
	private double fjac[];
	private double njac[];
	private double lhsa[];
	private double lhsb[];
	private double utmp[][];

	private int westSize, eastSize, bottomSize, topSize;
	private int northSize, southSize;
	private int startSendWest, startSendEast, startSendSouth;
	private int startSendNorth, startSendBottom, startSendTop;
	private int startRecvWest, startRecvEast, startRecvSouth;
	private int startRecvNorth, startRecvBottom, startRecvTop;

	private double inBuffer[], outBuffer[];
	private int BUF_SIZE;

	private final int EAST = 2000;
	private final int WEST = 3000;
	private final int NORTH = 4000;
	private final int SOUTH = 5000;
	private final int BOTTOM = 6000;
	private final int TOP = 7000;

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
	private int myRank;
	private int numProcess; /* active processes (a square) */
	private int npTotal; /* processes launched */
	private boolean active = true; /* false for the excess processes masked out */
	private Intracomm commWork; /* communicator of the active processes */
	private Comm commSolve;
	private Comm commRHS;
	private Request sendReq, recvReq;

	public static void main(String[] args) throws NPBException {
		BT bt = null;

		BMArgs.parseCmdLineArgs(args, BM_NAME);
		try {
			bt = new BT(BMArgs.CLASS, BMArgs.DETAILED_TIMING);
		} catch (OutOfMemoryError e) {
			BMArgs.outOfMemoryMessage();
			throw e;
		}
		bt.runBenchmark(args);
	}

	public BMResults getResults(){
		return res;
	}

	public BT(char classSize, boolean dtiming) {

		CLASS = classSize;
		detailedTiming = dtiming;
	}

	public void build(String[] args) throws NPBException{

		MPI.Init(args);

		/* Active processes (NPB-MPI 3.4: get_active_nprocs, NPB_NPROCS_STRICT) */
		ActiveProcs ap;
		try {
			ap = ActiveProcs.select(ActiveProcs.SQUARE);
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

		switch( CLASS )
		{
		case 'S':
			problemSize = 12;
			niterDefault = 60;
			dtDefault = 0.010;
			break;
		case 'W':
			problemSize = 24;
			niterDefault = 200;
			dtDefault = 0.0008;
			break;
		case 'A':
			problemSize = 64;
			niterDefault = 200;
			dtDefault = 0.0008;
			break;
		case 'B':
			problemSize = 102;
			niterDefault = 200;
			dtDefault = 0.0003;
			break;
		case 'C':
			problemSize = 162;
			niterDefault = 200;
			dtDefault = 0.0001;
			break;
		case 'D':
			problemSize = 408;
			niterDefault = 250;
			dtDefault = 0.00002;
			break;
		case 'E':
			problemSize = 1020;
			niterDefault = 250;
			dtDefault = 0.4e-5;
			break;
		default:
			terminate();
			throw new BadClassException("Specified class (" + CLASS + ") not valid for this benchmark\n"
					+ "Valid classes are S, W, A, B, C, D and E");
		}

		/* cells per process (the original gets it from get_active_nprocs) */
		ncells = (int)Math.sqrt(numProcess + 0.00001);

		cellCoord = new int[3][ncells];
		cellHigh = new int[3][ncells];
		cellLow = new int[3][ncells];
		cellSize = new int[3][ncells];
		slice = new int[3][ncells];
		start = new int[3][ncells];
		end = new int[3][ncells];
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

		setupMPI();
		readData();

		/* The class of the problem actually run (set_class of the original) */
		char runClass = CLASS;
		if(gridPoints[0] != problemSize || gridPoints[1] != problemSize || gridPoints[2] != problemSize
				|| niter != niterDefault)
			runClass = 'U';

		if(myRank == 0){
			System.out.println(" Size:  "+gridPoints[0]+"x "+gridPoints[1]+"x "+gridPoints[2]+"  (class "+runClass+")");
			System.out.println(" Iterations:  "+niter+"    dt:  "+dt);
			System.out.println(" Total number of processes:  "+npTotal);
			if (numProcess != npTotal)
				System.out.println(" WARNING: Number of processes is not a square number ("+numProcess+" active)");
			System.out.println();
		}

		allocSpace();

		makeSet();

		for(int c = 0; c<ncells; c++){
			if( cellSize[0][c] > IMAX ||
					cellSize[1][c] > JMAX ||
					cellSize[2][c] > KMAX ){
				System.err.println(myRank+" "+c+" "+cellSize[0][c]+" "+cellSize[1][c]+" "+cellSize[2][c]);
				System.err.println(" Problem size too big for compiled array sizes");
				terminate();
				return;
			}
		}

		setConstants();

		initialize();

		lhsInit();

		exactRHS();

		computeBufferSize(5);

		/* Do one time step to touch all code, and reinitialize */
		adi();

		initialize();

		/* Synchronize before placing time stamp */
		commWork.Barrier();

		timer.reset(T_COMP);
		timer.reset(T_COMM);
		timer.start(T_COMP);

		for(int step = 1; step<=niter; step++){
			if(myRank == 0){
				if(step%20 == 0 || step == niter || step == 1)
					System.out.println(" Time step "+step);
			}
			adi();
		}

		timer.stop(T_COMP);

		double times[] = { timer.read(T_COMP), timer.read(T_COMM) };
		double maxtimes[] = {0,0};

		int verified = verify(runClass);

		commWork.Reduce(times, 0, maxtimes, 0, 2, MPI.DOUBLE, MPI.MAX, 0);

		if(myRank == 0){
			double n3 = (double)gridPoints[0]*gridPoints[1]*gridPoints[2];
			double navg = (gridPoints[0]+gridPoints[1]+gridPoints[2])/3.0;
			double mflops = 0;
			if(maxtimes[0] != 0)
				/* (the constants are REAL(4) literals in the original, as in FT; here double) */
				mflops = 1.0E-6*(double)niter*(3478.8*n3
						-17655.7*(navg*navg)
						+28023.7*navg)/maxtimes[0];

			res = new BMResults(BM_NAME,
					CLASS,
					gridPoints[0],
					gridPoints[1],
					gridPoints[2],
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
	 * This method only does pretty basic things because
	 * all the original MPI setup stuff is not necessary
	 * with this MPJ/Java implementation.
	 */
	private void setupMPI(){
		commSolve = (Comm)commWork.clone();
		commRHS = (Comm)commWork.clone();
	}

	/*
	 * Read in and broadcast input data
	 */
	private void readData(){

		boolean inputData = true;

		if(myRank == 0){
			BMArgs.bannerMPJ(BM_NAME, CLASS, npTotal);

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
				commWork.Abort(-7);
			}
			if(!inputData){
				System.out.println(" No input file inputbt.data. Using compiled defaults");
				niter = niterDefault;
				dt = dtDefault;
				gridPoints[0] = problemSize;
				gridPoints[1] = problemSize;
				gridPoints[2] = problemSize;
			}

		}

		int buff[] = {niter,gridPoints[0],gridPoints[1],gridPoints[2]};
		double dtArray[] = {dt};

		commWork.Bcast(buff, 0, 4, MPI.INT, 0);
		commWork.Bcast(dtArray, 0, 1, MPI.DOUBLE, 0);

		niter = buff[0];
		gridPoints[0] = buff[1];
		gridPoints[1] = buff[2];
		gridPoints[2] = buff[3];
		dt = dtArray[0];
	}

	/* Index of the point (i,j,k) of the cell c (i, j and k from -2 to the cell size + 1) */
	private int ix(int i, int j, int k, int c){
		return (i+2) + DI*((j+2) + DJ*((k+2) + DK*c));
	}

	private void allocSpace() throws NPBException{

		MAX_CELL_DIM = problemSize/ncells + 1;
		IMAX = MAX_CELL_DIM;
		JMAX = MAX_CELL_DIM;
		KMAX = MAX_CELL_DIM;
		MAXD1 = MAX_CELL_DIM + 1;
		DI = IMAX + 4;
		DJ = JMAX + 4;
		DK = KMAX + 4;
		BUF_SIZE = MAX_CELL_DIM*MAX_CELL_DIM*(ncells-1)*60+1;

		long points = (long)DI*DJ*DK*ncells;
		if(25*points > Integer.MAX_VALUE)
			throw new BadInputException("Problem size too big for Java arrays: use more processes");

		int n = (int)points;

		u = new double[5*n];
		rhs = new double[5*n];
		forcing = new double[5*n];
		lhsc = new double[25*n];
		us = new double[n];
		vs = new double[n];
		ws = new double[n];
		qs = new double[n];
		rhoI = new double[n];
		square = new double[n];
		backsubInfo = new double[5*MAXD1*MAXD1*ncells];
		inBuffer = new double[BUF_SIZE];
		outBuffer = new double[BUF_SIZE];

		int nl = MAX_CELL_DIM + 4;
		cv = new double[nl];
		cuf = new double[nl];
		q = new double[nl];
		ue = new double[5][nl];
		buf = new double[5][nl];
		utmp = new double[6][nl];
		fjac = new double[25*nl];
		njac = new double[25*nl];
		lhsa = new double[25*nl];
		lhsb = new double[25*nl];
	}

	/*
	 * This method allocates space for a set of cells and fills the set     
	 * such that communication between cells on different nodes is only
	 * nearest neighbor
	 */
	private void makeSet() throws BadInputException{

		ncells = (int)Math.sqrt(numProcess);

		/*
		 * Determine the location of the cell at the bottom of the 3D 
		 * array of cells
		 */
		cellCoord[0][0] = myRank%ncells;
		cellCoord[1][0] = myRank/ncells;
		cellCoord[2][0] = 0;

		/*
		 * Set the cellCoord for cells in the rest of the z-layers; 
		 * this comes down to a simple linear numbering in the z-direction,
		 * and to the doubly-cyclic numbering in the other directions.
		 */
		for(int c = 1; c<ncells; c++){
			cellCoord[0][c] = (cellCoord[0][c-1]+1)%ncells;
			cellCoord[1][c] = (cellCoord[1][c-1]-1+ncells)%ncells;
			cellCoord[2][c] = c;
		}

		/*
		 * Offset all the coordinates by 1 to adjust for Fortran arrays
		 */
		for(int dir = 0; dir<3; dir++){
			for(int c = 0; c<ncells; c++){
				cellCoord[dir][c] +=  1;
			}
		}

		/*
		 * slice[dir][n] contains the sequence number of the cell that is in
		 * coordinate plane n in the dir direction
		 */
		for(int dir = 0; dir<3; dir++){
			for(int c = 0; c<ncells; c++){
				slice[dir][cellCoord[dir][c]-1] = c;
			}
		}

		/*
		 * Fill the predecessor and successor entries, using the indices 
		 * of the bottom cells (they are the same at each level of k 
		 * anyway) acting as if full periodicity pertains; note that ncells is
		 * added to those arguments to the mod functions that might
		 * otherwise return wrong values when using the modulo function
		 */
		int i = cellCoord[0][0]-1;
		int j = cellCoord[1][0]-1;

		predecessor[0] = (i-1+ncells)%ncells + ncells*j;
		predecessor[1] = i + ncells*((j-1+ncells)%ncells);
		predecessor[2] = (i+1)%ncells + ncells*((j-1+ncells)%ncells);
		successor[0] = (i+1)%ncells + ncells*j;
		successor[1] = i + ncells*((j+1)%ncells);
		successor[2] = (i-1+ncells)%ncells + ncells*((j+1)%ncells);

		/*
		 * Now compute the sizes of the cells
		 */
		for(int dir = 0; dir<3; dir++){
			/*
			 * Set cellCoord range for each direction
			 */
			int size = gridPoints[dir]/ncells;
			int excess = gridPoints[dir]%ncells;

			for(int c = 0; c<ncells; c++){
				if(cellCoord[dir][c] <= excess){
					cellSize[dir][c] = size + 1;
					cellLow[dir][c] = (cellCoord[dir][c]-1)*(size+1);
					cellHigh[dir][c] = cellLow[dir][c] + size;
				}
				else{
					cellSize[dir][c] = size;
					cellLow[dir][c] = excess*(size+1)+(cellCoord[dir][c]-excess-1)*size;
					cellHigh[dir][c] = cellLow[dir][c]+size-1;
				}
				if(cellSize[dir][c] <= 2){
					throw new BadInputException("Error: Cell size too small. Min size is 3");
				}
			}
		}

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
		int c, fx, fy, fz, i, ii, j, jj, k, kk, m;
		double Peta, Pxi, Pzeta, eta, xi, zeta;
		double Pface[][][] = new double[2][3][5];
		double temp[] = new double[5];
		for (c = 0; c < ncells; c++){
			for (kk = -1; kk <= KMAX; kk++){
				for (jj = -1; jj <= JMAX; jj++){
					for (ii = -1; ii <= IMAX; ii++){
						for (m = 0; m <= 4; m++){
							u[5*ix(ii,jj,kk,c)+m] = 1.0;
						}
					}
				}
			}
		}
		for (c = 0; c < ncells; c++){
			kk = 0;
			for (k = cellLow[2][c]; k <= cellHigh[2][c]; k++){
				zeta = (double)(k) * dnzm1;
				jj = 0;
				for (j = cellLow[1][c]; j <= cellHigh[1][c]; j++){
					eta = (double)(j) * dnym1;
					ii = 0;
					for (i = cellLow[0][c]; i <= cellHigh[0][c]; i++){
						xi = (double)(i) * dnxm1;
						for (fx = 1; fx <= 2; fx++){
							exactSolution((double)(fx-1), eta, zeta, Pface[fx-1][1-1]);
						}
						for (fy = 1; fy <= 2; fy++){
							exactSolution(xi, (double)(fy-1), zeta, Pface[fy-1][2-1]);
						}
						for (fz = 1; fz <= 2; fz++){
							exactSolution(xi, eta, (double)(fz-1), Pface[fz-1][3-1]);
						}
						for (m = 0; m <= 4; m++){
							Pxi = xi * Pface[1][0][m] + (1.0-xi) * Pface[0][0][m];
							Peta = eta * Pface[1][1][m] + (1.0-eta) * Pface[0][1][m];
							Pzeta = zeta * Pface[1][2][m] + (1.0-zeta) * Pface[0][2][m];
							u[5*ix(ii,jj,kk,c)+m] = Pxi + Peta + Pzeta - Pxi*Peta - Pxi*Pzeta - Peta*Pzeta + Pxi*Peta*Pzeta;
						}
						ii = ii + 1;
					}
					jj = jj + 1;
				}
				kk = kk+1;
			}
		}
		c = slice[0][1-1];
		ii = 0;
		xi = 0.0;
		kk = 0;
		for (k = cellLow[2][c]; k <= cellHigh[2][c]; k++){
			zeta = (double)(k) * dnzm1;
			jj = 0;
			for (j = cellLow[1][c]; j <= cellHigh[1][c]; j++){
				eta = (double)(j) * dnym1;
				exactSolution(xi, eta, zeta, temp);
				for (m = 0; m <= 4; m++){
					u[5*ix(ii,jj,kk,c)+m] = temp[m];
				}
				jj = jj + 1;
			}
			kk = kk + 1;
		}
		c = slice[0][ncells-1];
		ii = cellSize[0][c]-1;
		xi = 1.0;
		kk = 0;
		for (k = cellLow[2][c]; k <= cellHigh[2][c]; k++){
			zeta = (double)(k) * dnzm1;
			jj = 0;
			for (j = cellLow[1][c]; j <= cellHigh[1][c]; j++){
				eta = (double)(j) * dnym1;
				exactSolution(xi, eta, zeta, temp);
				for (m = 0; m <= 4; m++){
					u[5*ix(ii,jj,kk,c)+m] = temp[m];
				}
				jj = jj + 1;
			}
			kk = kk + 1;
		}
		c = slice[1][1-1];
		jj = 0;
		eta = 0.0;
		kk = 0;
		for (k = cellLow[2][c]; k <= cellHigh[2][c]; k++){
			zeta = (double)(k) * dnzm1;
			ii = 0;
			for (i = cellLow[0][c]; i <= cellHigh[0][c]; i++){
				xi = (double)(i) * dnxm1;
				exactSolution(xi, eta, zeta, temp);
				for (m = 0; m <= 4; m++){
					u[5*ix(ii,jj,kk,c)+m] = temp[m];
				}
				ii = ii + 1;
			}
			kk = kk + 1;
		}
		c = slice[1][ncells-1];
		jj = cellSize[1][c]-1;
		eta = 1.0;
		kk = 0;
		for (k = cellLow[2][c]; k <= cellHigh[2][c]; k++){
			zeta = (double)(k) * dnzm1;
			ii = 0;
			for (i = cellLow[0][c]; i <= cellHigh[0][c]; i++){
				xi = (double)(i) * dnxm1;
				exactSolution(xi, eta, zeta, temp);
				for (m = 0; m <= 4; m++){
					u[5*ix(ii,jj,kk,c)+m] = temp[m];
				}
				ii = ii + 1;
			}
			kk = kk + 1;
		}
		c = slice[2][1-1];
		kk = 0;
		zeta = 0.0;
		jj = 0;
		for (j = cellLow[1][c]; j <= cellHigh[1][c]; j++){
			eta = (double)(j) * dnym1;
			ii = 0;
			for (i = cellLow[0][c]; i <= cellHigh[0][c]; i++){
				xi = (double)(i) *dnxm1;
				exactSolution(xi, eta, zeta, temp);
				for (m = 0; m <= 4; m++){
					u[5*ix(ii,jj,kk,c)+m] = temp[m];
				}
				ii = ii + 1;
			}
			jj = jj + 1;
		}
		c = slice[2][ncells-1];
		kk = cellSize[2][c]-1;
		zeta = 1.0;
		jj = 0;
		for (j = cellLow[1][c]; j <= cellHigh[1][c]; j++){
			eta = (double)(j) * dnym1;
			ii = 0;
			for (i = cellLow[0][c]; i <= cellHigh[0][c]; i++){
				xi = (double)(i) * dnxm1;
				exactSolution(xi, eta, zeta, temp);
				for (m = 0; m <= 4; m++){
					u[5*ix(ii,jj,kk,c)+m] = temp[m];
				}
				ii = ii + 1;
			}
			jj = jj + 1;
		}
	}

	/* The limits of the cells (start and end flags of each direction) */
	private void lhsInit(){
		for(int c = 0; c<ncells; c++){
			for(int d = 0; d<3; d++){
				start[d][c] = (cellCoord[d][c] == 1) ? 1 : 0;
				end[d][c] = (cellCoord[d][c] == ncells) ? 1 : 0;
			}
			/* (lhsc is already zero: the arrays of Java start cleared) */
		}
	}

	/* The block lhsa, lhsb of the points 0..size of a line: zero and identity */
	private void lhsabInit(int size){
		for(int i = 0; i<=size; i++){
			for(int m = 0; m<5; m++){
				for(int n = 0; n<5; n++){
					lhsa[25*(i+1)+m+5*n] = 0.0;
					lhsb[25*(i+1)+m+5*n] = 0.0;
				}
				lhsb[25*(i+1)+m+5*m] = 1.0;
			}
		}
	}

	/*
	 * This method computes the forcing function (the original exact_rhs);
	 * a literal translation: the three sweeps differ in the order of the operands
	 */
	private void exactRHS(){
		int c, i, im1, ip1, j, jm1, jp1, k, km1, kp1, m;
		double dtpp, eta, xi, zeta;
		double dtemp[] = new double[5];
		for (c = 0; c < ncells; c++){
			for (k = 0; k <= cellSize[2][c]-1; k++){
				for (j = 0; j <= cellSize[1][c]-1; j++){
					for (i = 0; i <= cellSize[0][c]-1; i++){
						for (m = 0; m <= 4; m++){
							forcing[5*ix(i,j,k,c)+m] = 0.0;
						}
					}
				}
			}
			for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
				zeta = (double)(k+cellLow[2][c]) * dnzm1;
				for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
					eta = (double)(j+cellLow[1][c]) * dnym1;
					for (i = -2*(1-start[0][c]); i <= cellSize[0][c]+1-2*end[0][c]; i++){
						xi = (double)(i+cellLow[0][c]) * dnxm1;
						exactSolution(xi, eta, zeta, dtemp);
						for (m = 0; m <= 4; m++){
							ue[m][i+2] = dtemp[m];
						}
						dtpp = 1.0 / dtemp[0];
						for (m = 1; m <= 4; m++){
							buf[m][i+2] = dtpp * dtemp[m];
						}
						cuf[i+2] = buf[1][i+2] * buf[1][i+2];
						buf[0][i+2] = cuf[i+2] + buf[2][i+2] * buf[2][i+2] + buf[3][i+2] * buf[3][i+2];
						q[i+2] = 0.5*(buf[1][i+2]*ue[1][i+2] + buf[2][i+2]*ue[2][i+2] + buf[3][i+2]*ue[3][i+2]);
					}
					for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
						im1 = i-1;
						ip1 = i+1;
						forcing[5*ix(i,j,k,c)+0] = forcing[5*ix(i,j,k,c)+0] - tx2*( ue[1][ip1+2]-ue[1][im1+2] )+ dx1tx1*(ue[0][ip1+2]-2.0*ue[0][i+2]+ue[0][im1+2]);
						forcing[5*ix(i,j,k,c)+1] = forcing[5*ix(i,j,k,c)+1] - tx2 * ( (ue[1][ip1+2]*buf[1][ip1+2]+c2*(ue[4][ip1+2]-q[ip1+2]))- (ue[1][im1+2]*buf[1][im1+2]+c2*(ue[4][im1+2]-q[im1+2])))+ xxcon1*(buf[1][ip1+2]-2.0*buf[1][i+2]+buf[1][im1+2])+ dx2tx1*( ue[1][ip1+2]-2.0* ue[1][i+2]+ue[1][im1+2]);
						forcing[5*ix(i,j,k,c)+2] = forcing[5*ix(i,j,k,c)+2] - tx2 * ( ue[2][ip1+2]*buf[1][ip1+2]-ue[2][im1+2]*buf[1][im1+2])+ xxcon2*(buf[2][ip1+2]-2.0*buf[2][i+2]+buf[2][im1+2])+ dx3tx1*( ue[2][ip1+2]-2.0*ue[2][i+2] +ue[2][im1+2]);
						forcing[5*ix(i,j,k,c)+3] = forcing[5*ix(i,j,k,c)+3] - tx2*( ue[3][ip1+2]*buf[1][ip1+2]-ue[3][im1+2]*buf[1][im1+2])+ xxcon2*(buf[3][ip1+2]-2.0*buf[3][i+2]+buf[3][im1+2])+ dx4tx1*( ue[3][ip1+2]-2.0* ue[3][i+2]+ ue[3][im1+2]);
						forcing[5*ix(i,j,k,c)+4] = forcing[5*ix(i,j,k,c)+4] - tx2*( buf[1][ip1+2]*(c1*ue[4][ip1+2]-c2*q[ip1+2])- buf[1][im1+2]*(c1*ue[4][im1+2]-c2*q[im1+2]))+ 0.5*xxcon3*(buf[0][ip1+2]-2.0*buf[0][i+2]+ buf[0][im1+2])+ xxcon4*(cuf[ip1+2]-2.0*cuf[i+2]+cuf[im1+2])+ xxcon5*(buf[4][ip1+2]-2.0*buf[4][i+2]+buf[4][im1+2])+ dx5tx1*( ue[4][ip1+2]-2.0* ue[4][i+2]+ ue[4][im1+2]);
					}
					if (start[0][c] > 0){
						for (m = 0; m <= 4; m++){
							i = 1;
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp * (5.0*ue[m][i+2] - 4.0*ue[m][i+1+2] +ue[m][i+2+2]);
							i = 2;
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp * (-4.0*ue[m][i-1+2] + 6.0*ue[m][i+2] - 4.0*ue[m][i+1+2] + ue[m][i+2+2]);
						}
					}
					for (i = start[0][c]*3; i <= cellSize[0][c]-3*end[0][c]-1; i++){
						for (m = 0; m <= 4; m++){
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp* (ue[m][i-2+2] - 4.0*ue[m][i-1+2] + 6.0*ue[m][i+2] - 4.0*ue[m][i+1+2] + ue[m][i+2+2]);
						}
					}
					if (end[0][c] > 0){
						for (m = 0; m <= 4; m++){
							i = cellSize[0][c]-3;
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp * (ue[m][i-2+2] - 4.0*ue[m][i-1+2] + 6.0*ue[m][i+2] - 4.0*ue[m][i+1+2]);
							i = cellSize[0][c]-2;
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp * (ue[m][i-2+2] - 4.0*ue[m][i-1+2] + 5.0*ue[m][i+2]);
						}
					}
				}
			}
			for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
				zeta = (double)(k+cellLow[2][c]) * dnzm1;
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					xi = (double)(i+cellLow[0][c]) * dnxm1;
					for (j = -2*(1-start[1][c]); j <= cellSize[1][c]+1-2*end[1][c]; j++){
						eta = (double)(j+cellLow[1][c]) * dnym1;
						exactSolution(xi, eta, zeta, dtemp);
						for (m = 0; m <= 4; m++){
							ue[m][j+2] = dtemp[m];
						}
						dtpp = 1.0/dtemp[0];
						for (m = 1; m <= 4; m++){
							buf[m][j+2] = dtpp * dtemp[m];
						}
						cuf[j+2] = buf[2][j+2] * buf[2][j+2];
						buf[0][j+2] = cuf[j+2] + buf[1][j+2] * buf[1][j+2] + buf[3][j+2] * buf[3][j+2];
						q[j+2] = 0.5*(buf[1][j+2]*ue[1][j+2] + buf[2][j+2]*ue[2][j+2] + buf[3][j+2]*ue[3][j+2]);
					}
					for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
						jm1 = j-1;
						jp1 = j+1;
						forcing[5*ix(i,j,k,c)+0] = forcing[5*ix(i,j,k,c)+0] - ty2*( ue[2][jp1+2]-ue[2][jm1+2] )+ dy1ty1*(ue[0][jp1+2]-2.0*ue[0][j+2]+ue[0][jm1+2]);
						forcing[5*ix(i,j,k,c)+1] = forcing[5*ix(i,j,k,c)+1] - ty2*( ue[1][jp1+2]*buf[2][jp1+2]-ue[1][jm1+2]*buf[2][jm1+2])+ yycon2*(buf[1][jp1+2]-2.0*buf[1][j+2]+buf[1][jm1+2])+ dy2ty1*( ue[1][jp1+2]-2.0* ue[1][j+2]+ ue[1][jm1+2]);
						forcing[5*ix(i,j,k,c)+2] = forcing[5*ix(i,j,k,c)+2] - ty2*( (ue[2][jp1+2]*buf[2][jp1+2]+c2*(ue[4][jp1+2]-q[jp1+2]))- (ue[2][jm1+2]*buf[2][jm1+2]+c2*(ue[4][jm1+2]-q[jm1+2])))+ yycon1*(buf[2][jp1+2]-2.0*buf[2][j+2]+buf[2][jm1+2])+ dy3ty1*( ue[2][jp1+2]-2.0*ue[2][j+2] +ue[2][jm1+2]);
						forcing[5*ix(i,j,k,c)+3] = forcing[5*ix(i,j,k,c)+3] - ty2*( ue[3][jp1+2]*buf[2][jp1+2]-ue[3][jm1+2]*buf[2][jm1+2])+ yycon2*(buf[3][jp1+2]-2.0*buf[3][j+2]+buf[3][jm1+2])+ dy4ty1*( ue[3][jp1+2]-2.0*ue[3][j+2]+ ue[3][jm1+2]);
						forcing[5*ix(i,j,k,c)+4] = forcing[5*ix(i,j,k,c)+4] - ty2*( buf[2][jp1+2]*(c1*ue[4][jp1+2]-c2*q[jp1+2])- buf[2][jm1+2]*(c1*ue[4][jm1+2]-c2*q[jm1+2]))+ 0.5*yycon3*(buf[0][jp1+2]-2.0*buf[0][j+2]+ buf[0][jm1+2])+ yycon4*(cuf[jp1+2]-2.0*cuf[j+2]+cuf[jm1+2])+ yycon5*(buf[4][jp1+2]-2.0*buf[4][j+2]+buf[4][jm1+2])+ dy5ty1*(ue[4][jp1+2]-2.0*ue[4][j+2]+ue[4][jm1+2]);
					}
					if (start[1][c] > 0){
						for (m = 0; m <= 4; m++){
							j = 1;
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp * (5.0*ue[m][j+2] - 4.0*ue[m][j+1+2] +ue[m][j+2+2]);
							j = 2;
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp * (-4.0*ue[m][j-1+2] + 6.0*ue[m][j+2] - 4.0*ue[m][j+1+2] + ue[m][j+2+2]);
						}
					}
					for (j = start[1][c]*3; j <= cellSize[1][c]-3*end[1][c]-1; j++){
						for (m = 0; m <= 4; m++){
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp* (ue[m][j-2+2] - 4.0*ue[m][j-1+2] + 6.0*ue[m][j+2] - 4.0*ue[m][j+1+2] + ue[m][j+2+2]);
						}
					}
					if (end[1][c] > 0){
						for (m = 0; m <= 4; m++){
							j = cellSize[1][c]-3;
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp * (ue[m][j-2+2] - 4.0*ue[m][j-1+2] + 6.0*ue[m][j+2] - 4.0*ue[m][j+1+2]);
							j = cellSize[1][c]-2;
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp * (ue[m][j-2+2] - 4.0*ue[m][j-1+2] + 5.0*ue[m][j+2]);
						}
					}
				}
			}
			for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
				eta = (double)(j+cellLow[1][c]) * dnym1;
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					xi = (double)(i+cellLow[0][c]) * dnxm1;
					for (k = -2*(1-start[2][c]); k <= cellSize[2][c]+1-2*end[2][c]; k++){
						zeta = (double)(k+cellLow[2][c]) * dnzm1;
						exactSolution(xi, eta, zeta, dtemp);
						for (m = 0; m <= 4; m++){
							ue[m][k+2] = dtemp[m];
						}
						dtpp = 1.0/dtemp[0];
						for (m = 1; m <= 4; m++){
							buf[m][k+2] = dtpp * dtemp[m];
						}
						cuf[k+2] = buf[3][k+2] * buf[3][k+2];
						buf[0][k+2] = cuf[k+2] + buf[1][k+2] * buf[1][k+2] + buf[2][k+2] * buf[2][k+2];
						q[k+2] = 0.5*(buf[1][k+2]*ue[1][k+2] + buf[2][k+2]*ue[2][k+2] + buf[3][k+2]*ue[3][k+2]);
					}
					for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
						km1 = k-1;
						kp1 = k+1;
						forcing[5*ix(i,j,k,c)+0] = forcing[5*ix(i,j,k,c)+0] - tz2*( ue[3][kp1+2]-ue[3][km1+2] )+ dz1tz1*(ue[0][kp1+2]-2.0*ue[0][k+2]+ue[0][km1+2]);
						forcing[5*ix(i,j,k,c)+1] = forcing[5*ix(i,j,k,c)+1] - tz2 * ( ue[1][kp1+2]*buf[3][kp1+2]-ue[1][km1+2]*buf[3][km1+2])+ zzcon2*(buf[1][kp1+2]-2.0*buf[1][k+2]+buf[1][km1+2])+ dz2tz1*( ue[1][kp1+2]-2.0* ue[1][k+2]+ ue[1][km1+2]);
						forcing[5*ix(i,j,k,c)+2] = forcing[5*ix(i,j,k,c)+2] - tz2 * ( ue[2][kp1+2]*buf[3][kp1+2]-ue[2][km1+2]*buf[3][km1+2])+ zzcon2*(buf[2][kp1+2]-2.0*buf[2][k+2]+buf[2][km1+2])+ dz3tz1*(ue[2][kp1+2]-2.0*ue[2][k+2]+ue[2][km1+2]);
						forcing[5*ix(i,j,k,c)+3] = forcing[5*ix(i,j,k,c)+3] - tz2 * ( (ue[3][kp1+2]*buf[3][kp1+2]+c2*(ue[4][kp1+2]-q[kp1+2]))- (ue[3][km1+2]*buf[3][km1+2]+c2*(ue[4][km1+2]-q[km1+2])))+ zzcon1*(buf[3][kp1+2]-2.0*buf[3][k+2]+buf[3][km1+2])+ dz4tz1*( ue[3][kp1+2]-2.0*ue[3][k+2] +ue[3][km1+2]);
						forcing[5*ix(i,j,k,c)+4] = forcing[5*ix(i,j,k,c)+4] - tz2 * ( buf[3][kp1+2]*(c1*ue[4][kp1+2]-c2*q[kp1+2])- buf[3][km1+2]*(c1*ue[4][km1+2]-c2*q[km1+2]))+ 0.5*zzcon3*(buf[0][kp1+2]-2.0*buf[0][k+2] +buf[0][km1+2])+ zzcon4*(cuf[kp1+2]-2.0*cuf[k+2]+cuf[km1+2])+ zzcon5*(buf[4][kp1+2]-2.0*buf[4][k+2]+buf[4][km1+2])+ dz5tz1*( ue[4][kp1+2]-2.0*ue[4][k+2]+ ue[4][km1+2]);
					}
					if (start[2][c] > 0){
						for (m = 0; m <= 4; m++){
							k = 1;
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp * (5.0*ue[m][k+2] - 4.0*ue[m][k+1+2] +ue[m][k+2+2]);
							k = 2;
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp * (-4.0*ue[m][k-1+2] + 6.0*ue[m][k+2] - 4.0*ue[m][k+1+2] + ue[m][k+2+2]);
						}
					}
					for (k = start[2][c]*3; k <= cellSize[2][c]-3*end[2][c]-1; k++){
						for (m = 0; m <= 4; m++){
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp* (ue[m][k-2+2] - 4.0*ue[m][k-1+2] + 6.0*ue[m][k+2] - 4.0*ue[m][k+1+2] + ue[m][k+2+2]);
						}
					}
					if (end[2][c] > 0){
						for (m = 0; m <= 4; m++){
							k = cellSize[2][c]-3;
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp * (ue[m][k-2+2] - 4.0*ue[m][k-1+2] + 6.0*ue[m][k+2] - 4.0*ue[m][k+1+2]);
							k = cellSize[2][c]-2;
							forcing[5*ix(i,j,k,c)+m] = forcing[5*ix(i,j,k,c)+m] - dssp * (ue[m][k-2+2] - 4.0*ue[m][k-1+2] + 5.0*ue[m][k+2]);
						}
					}
				}
			}
			for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
				for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
					for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
						for (m = 0; m <= 4; m++){
							forcing[5*ix(i,j,k,c)+m] = -1.0 * forcing[5*ix(i,j,k,c)+m];
						}
					}
				}
			}
		}
	}

	private void computeBufferSize(int dim){

		int faceSize;

		if(ncells == 1)
			return;

		westSize = 0;
		eastSize = 0;

		for(int c = 0; c<ncells; c++){
			faceSize = cellSize[1][c] * cellSize[2][c] * dim * 2;
			if(cellCoord[0][c] != 1)
				westSize = westSize+faceSize;
			if(cellCoord[0][c] != ncells)
				eastSize = eastSize+faceSize;
		}

		northSize = 0;
		southSize = 0;

		for(int c = 0; c<ncells; c++){
			faceSize = cellSize[0][c] * cellSize[2][c] * dim * 2;
			if(cellCoord[1][c] != 1)
				southSize = southSize+faceSize;
			if(cellCoord[1][c] != ncells)
				northSize = northSize+faceSize;			
		}

		topSize = 0;
		bottomSize = 0;

		for(int c = 0; c<ncells; c++){
			faceSize = cellSize[0][c] * cellSize[1][c] * dim * 2;
			if(cellCoord[2][c] != 1)
				bottomSize = bottomSize+faceSize;
			if(cellCoord[2][c] != ncells)
				topSize = topSize+faceSize;			
		}

		startSendWest = 1;
		startSendEast = startSendWest + westSize;
		startSendSouth = startSendEast + eastSize;
		startSendNorth = startSendSouth + southSize;
		startSendBottom = startSendNorth + northSize;
		startSendTop = startSendBottom + bottomSize;

		startRecvWest = 1;
		startRecvEast = startRecvWest + westSize;
		startRecvSouth = startRecvEast + eastSize;
		startRecvNorth = startRecvSouth + southSize;
		startRecvBottom = startRecvNorth + northSize;
		startRecvTop = startRecvBottom + bottomSize;

	}

	private void adi(){

		copyFaces();

		solve(0);

		solve(1);

		solve(2);

		add();
	}

	/*
	 * This method copies the face values of u to the overlap locations of the
	 * adjacent sets of cells, and then computes the right hand side. Because
	 * a set of cells interfaces in each direction with exactly one other set,
	 * only six different buffers are needed.
	 */
	private void copyFaces(){
		int c, i, j, k, m, p0, p1, p2, p3, p4, p5;
		int ss[] = new int[6];
		int sr[] = new int[6];
		int bSize[] = new int[6];
		Request requests[] = new Request[12];

		/* exit immediately if there are no faces to be copied */
		if(numProcess == 1){
			computeRHS();
			return;
		}

		ss[0] = startSendEast;
		ss[1] = startSendWest;
		ss[2] = startSendNorth;
		ss[3] = startSendSouth;
		ss[4] = startSendTop;
		ss[5] = startSendBottom;

		sr[0] = startRecvEast;
		sr[1] = startRecvWest;
		sr[2] = startRecvNorth;
		sr[3] = startRecvSouth;
		sr[4] = startRecvTop;
		sr[5] = startRecvBottom;

		bSize[0] = eastSize;
		bSize[1] = westSize;
		bSize[2] = northSize;
		bSize[3] = southSize;
		bSize[4] = topSize;
		bSize[5] = bottomSize;

		/* Fill the buffers to be sent (east, west, north, south, top, bottom) */
		p0 = 0;
		p1 = 0;
		p2 = 0;
		p3 = 0;
		p4 = 0;
		p5 = 0;
		for (c = 0; c < ncells; c++){
			if (cellCoord[0][c] != ncells){
				for (k = 0; k <= cellSize[2][c]-1; k++){
					for (j = 0; j <= cellSize[1][c]-1; j++){
						for (i = cellSize[0][c]-2; i <= cellSize[0][c]-1; i++){
							for (m = 0; m <= 4; m++){
								outBuffer[ss[0]+p0-1] = u[5*ix(i,j,k,c)+m];
								p0 = p0 + 1;
							}
						}
					}
				}
			}
			if (cellCoord[0][c] != 1){
				for (k = 0; k <= cellSize[2][c]-1; k++){
					for (j = 0; j <= cellSize[1][c]-1; j++){
						for (i = 0; i <= 1; i++){
							for (m = 0; m <= 4; m++){
								outBuffer[ss[1]+p1-1] = u[5*ix(i,j,k,c)+m];
								p1 = p1 + 1;
							}
						}
					}
				}
			}
			if (cellCoord[1][c] != ncells){
				for (k = 0; k <= cellSize[2][c]-1; k++){
					for (j = cellSize[1][c]-2; j <= cellSize[1][c]-1; j++){
						for (i = 0; i <= cellSize[0][c]-1; i++){
							for (m = 0; m <= 4; m++){
								outBuffer[ss[2]+p2-1] = u[5*ix(i,j,k,c)+m];
								p2 = p2 + 1;
							}
						}
					}
				}
			}
			if (cellCoord[1][c]!= 1){
				for (k = 0; k <= cellSize[2][c]-1; k++){
					for (j = 0; j <= 1; j++){
						for (i = 0; i <= cellSize[0][c]-1; i++){
							for (m = 0; m <= 4; m++){
								outBuffer[ss[3]+p3-1] = u[5*ix(i,j,k,c)+m];
								p3 = p3 + 1;
							}
						}
					}
				}
			}
			if (cellCoord[2][c] != ncells){
				for (k = cellSize[2][c]-2; k <= cellSize[2][c]-1; k++){
					for (j = 0; j <= cellSize[1][c]-1; j++){
						for (i = 0; i <= cellSize[0][c]-1; i++){
							for (m = 0; m <= 4; m++){
								outBuffer[ss[4]+p4-1] = u[5*ix(i,j,k,c)+m];
								p4 = p4 + 1;
							}
						}
					}
				}
			}
			if (cellCoord[2][c]!= 1){
				for (k = 0; k <= 1; k++){
					for (j = 0; j <= cellSize[1][c]-1; j++){
						for (i = 0; i <= cellSize[0][c]-1; i++){
							for (m = 0; m <= 4; m++){
								outBuffer[ss[5]+p5-1] = u[5*ix(i,j,k,c)+m];
								p5 = p5 + 1;
							}
						}
					}
				}
			}
		}

		if(detailedTiming)
			timer.start(T_COMM);
		requests[0] = commRHS.Irecv(inBuffer, sr[0]-1, bSize[0], MPI.DOUBLE, successor[0], WEST);
		requests[1] = commRHS.Irecv(inBuffer, sr[1]-1, bSize[1], MPI.DOUBLE, predecessor[0], EAST);
		requests[2] = commRHS.Irecv(inBuffer, sr[2]-1, bSize[2], MPI.DOUBLE, successor[1], SOUTH);
		requests[3] = commRHS.Irecv(inBuffer, sr[3]-1, bSize[3], MPI.DOUBLE, predecessor[1], NORTH);
		requests[4] = commRHS.Irecv(inBuffer, sr[4]-1, bSize[4], MPI.DOUBLE, successor[2], BOTTOM);
		requests[5] = commRHS.Irecv(inBuffer, sr[5]-1, bSize[5], MPI.DOUBLE, predecessor[2], TOP);
		requests[6] = commRHS.Isend(outBuffer, ss[0]-1, bSize[0], MPI.DOUBLE, successor[0], EAST);
		requests[7] = commRHS.Isend(outBuffer, ss[1]-1, bSize[1], MPI.DOUBLE, predecessor[0], WEST);
		requests[8] = commRHS.Isend(outBuffer, ss[2]-1, bSize[2], MPI.DOUBLE, successor[1], NORTH);
		requests[9] = commRHS.Isend(outBuffer, ss[3]-1, bSize[3], MPI.DOUBLE, predecessor[1], SOUTH);
		requests[10] = commRHS.Isend(outBuffer, ss[4]-1, bSize[4], MPI.DOUBLE, successor[2], TOP);
		requests[11] = commRHS.Isend(outBuffer, ss[5]-1, bSize[5], MPI.DOUBLE, predecessor[2], BOTTOM);
		Request.Waitall(requests);
		if(detailedTiming)
			timer.stop(T_COMM);

		/* Unpack the received faces */
		p0 = 0;
		p1 = 0;
		p2 = 0;
		p3 = 0;
		p4 = 0;
		p5 = 0;
		for (c = 0; c < ncells; c++){
			if (cellCoord[0][c] != 1){
				for (k = 0; k <= cellSize[2][c]-1; k++){
					for (j = 0; j <= cellSize[1][c]-1; j++){
						for (i = -2; i <= -1; i++){
							for (m = 0; m <= 4; m++){
								u[5*ix(i,j,k,c)+m] = inBuffer[sr[1]+p0-1];
								p0 = p0 + 1;
							}
						}
					}
				}
			}
			if (cellCoord[0][c] != ncells){
				for (k = 0; k <= cellSize[2][c]-1; k++){
					for (j = 0; j <= cellSize[1][c]-1; j++){
						for (i = cellSize[0][c]; i <= cellSize[0][c]+1; i++){
							for (m = 0; m <= 4; m++){
								u[5*ix(i,j,k,c)+m] = inBuffer[sr[0]+p1-1];
								p1 = p1 + 1;
							}
						}
					}
				}
			}
			if (cellCoord[1][c] != 1){
				for (k = 0; k <= cellSize[2][c]-1; k++){
					for (j = -2; j <= -1; j++){
						for (i = 0; i <= cellSize[0][c]-1; i++){
							for (m = 0; m <= 4; m++){
								u[5*ix(i,j,k,c)+m] = inBuffer[sr[3]+p2-1];
								p2 = p2 + 1;
							}
						}
					}
				}
			}
			if (cellCoord[1][c] != ncells){
				for (k = 0; k <= cellSize[2][c]-1; k++){
					for (j = cellSize[1][c]; j <= cellSize[1][c]+1; j++){
						for (i = 0; i <= cellSize[0][c]-1; i++){
							for (m = 0; m <= 4; m++){
								u[5*ix(i,j,k,c)+m] = inBuffer[sr[2]+p3-1];
								p3 = p3 + 1;
							}
						}
					}
				}
			}
			if (cellCoord[2][c] != 1){
				for (k = -2; k <= -1; k++){
					for (j = 0; j <= cellSize[1][c]-1; j++){
						for (i = 0; i <= cellSize[0][c]-1; i++){
							for (m = 0; m <= 4; m++){
								u[5*ix(i,j,k,c)+m] = inBuffer[sr[5]+p4-1];
								p4 = p4 + 1;
							}
						}
					}
				}
			}
			if (cellCoord[2][c] != ncells){
				for (k = cellSize[2][c]; k <= cellSize[2][c]+1; k++){
					for (j = 0; j <= cellSize[1][c]-1; j++){
						for (i = 0; i <= cellSize[0][c]-1; i++){
							for (m = 0; m <= 4; m++){
								u[5*ix(i,j,k,c)+m] = inBuffer[sr[4]+p5-1];
								p5 = p5 + 1;
							}
						}
					}
				}
			}
		}

		/* Now that we have all the data, compute the rhs */
		computeRHS();
	}

	/*
	 * The right hand side (the original compute_rhs), cell by cell. Each part of a
	 * cell is a separate method: HotSpot does not compile methods of more than 8000
	 * bytes of bytecode, and the whole body of compute_rhs is larger.
	 */
	private void computeRHS(){
		for(int c = 0; c<ncells; c++){
			rhsPrepare(c);
			rhsX(c);
			rhsY(c);
			rhsZ(c);
			rhsScale(c);
		}
	}

	/* Auxiliary fields (reciprocal of density, speeds, kinetic energy) and the forcing term */
	private void rhsPrepare(int c){
		int i, j, k, m;
		double rho_inv;
		for (k = -1; k <= cellSize[2][c]; k++){
			for (j = -1; j <= cellSize[1][c]; j++){
				for (i = -1; i <= cellSize[0][c]; i++){
					rho_inv = 1.0/u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+0];
					rhoI[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] = rho_inv;
					us[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] = u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1] * rho_inv;
					vs[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] = u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2] * rho_inv;
					ws[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] = u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3] * rho_inv;
					square[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] = 0.5* ( u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1]*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1] + u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2]*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2] + u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3]*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3] ) * rho_inv;
					qs[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] = square[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] * rho_inv;
				}
			}
		}
		for (k = 0; k <= cellSize[2][c]-1; k++){
			for (j = 0; j <= cellSize[1][c]-1; j++){
				for (i = 0; i <= cellSize[0][c]-1; i++){
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = forcing[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m];
					}
				}
			}
		}
	}

	/*
	 * Optimization (2026-11-16): rhsX/rhsY/rhsZ are split into one small method per loop nest and ix() is written out in place in
	 * rhsX/Y/Z, rhsPrepare and add (the originals were recompiled and C2 did not always inline ix(): rhsX grew 9.9x from class B to C).
	 * Same operations and order: bit-identical results.
	 */

	/* xi-direction flux differences and dissipation */
	private void rhsX(int c){
		int i, j, k, m;
		double uijk, um1, up1;
		rhsX_0(c);
		rhsX_1(c);
		rhsX_2(c);
		rhsX_3(c);
	}

	private void rhsX_0(int c){
		int i, j, k, m;
		double uijk, um1, up1;
		for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
			for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					uijk = us[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))];
					up1 = us[(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))];
					um1 = us[(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))];
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+0] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+0] + dx1tx1 * (u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+0] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+0] + u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+0]) - tx2 * (u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1] - u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1]);
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1] + dx2tx1 * (u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1] + u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1]) + xxcon2*con43 * (up1 - 2.0*uijk + um1) - tx2 * (u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1]*up1 - u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1]*um1 + (u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4]- square[(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))]- u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4]+ square[(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))])* c2);
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2] + dx3tx1 * (u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2] + u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2]) + xxcon2 * (vs[(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] - 2.0*vs[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] + vs[(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))]) - tx2 * (u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2]*up1 - u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2]*um1);
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3] + dx4tx1 * (u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3] + u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3]) + xxcon2 * (ws[(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] - 2.0*ws[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] + ws[(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))]) - tx2 * (u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3]*up1 - u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3]*um1);
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4] + dx5tx1 * (u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4] + u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4]) + xxcon3 * (qs[(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] - 2.0*qs[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] + qs[(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))]) + xxcon4 * (up1*up1 - 2.0*uijk*uijk + um1*um1) + xxcon5 * (u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4]*rhoI[(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4]*rhoI[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] + u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4]*rhoI[(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))]) - tx2 * ( (c1*u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4] - c2*square[(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))])*up1 - (c1*u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4] - c2*square[(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))])*um1 );
				}
			}
		}
	}

	private void rhsX_1(int c){
		int i, j, k, m;
		double uijk, um1, up1;
		if (start[0][c] > 0){
			for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
				for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
					i = 1;
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m]- dssp * ( 5.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] + u[5*(((i+2)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m]);
					}
					i = 2;
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - dssp * (-4.0*u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] + 6.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] + u[5*(((i+2)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m]);
					}
				}
			}
		}
	}

	private void rhsX_2(int c){
		int i, j, k, m;
		double uijk, um1, up1;
		for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
			for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
				for (i = 3*start[0][c]; i <= cellSize[0][c]-3*end[0][c]-1; i++){
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - dssp * ( u[5*(((i-2)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] + 6.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] + u[5*(((i+2)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] );
					}
				}
			}
		}
	}

	private void rhsX_3(int c){
		int i, j, k, m;
		double uijk, um1, up1;
		if (end[0][c] > 0){
			for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
				for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
					i = cellSize[0][c]-3;
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - dssp * ( u[5*(((i-2)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] + 6.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*(((i+1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] );
					}
					i = cellSize[0][c]-2;
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - dssp * ( u[5*(((i-2)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*(((i-1)+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] + 5.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] );
					}
				}
			}
		}
	}

	/* eta-direction flux differences and dissipation */
	private void rhsY(int c){
		int i, j, k, m;
		double vijk, vm1, vp1;
		rhsY_0(c);
		rhsY_1(c);
		rhsY_2(c);
		rhsY_3(c);
	}

	private void rhsY_0(int c){
		int i, j, k, m;
		double vijk, vm1, vp1;
		for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
			for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					vijk = vs[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))];
					vp1 = vs[((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))];
					vm1 = vs[((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))];
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+0] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+0] + dy1ty1 * (u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+0] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+0] + u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+0]) - ty2 * (u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+2] - u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+2]);
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1] + dy2ty1 * (u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+1] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1] + u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+1]) + yycon2 * (us[((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))] - 2.0*us[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] + us[((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))]) - ty2 * (u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+1]*vp1 - u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+1]*vm1);
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2] + dy3ty1 * (u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+2] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2] + u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+2]) + yycon2*con43 * (vp1 - 2.0*vijk + vm1) - ty2 * (u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+2]*vp1 - u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+2]*vm1 + (u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+4] - square[((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))] - u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+4] + square[((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))]) *c2);
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3] + dy4ty1 * (u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+3] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3] + u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+3]) + yycon2 * (ws[((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))] - 2.0*ws[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] + ws[((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))]) - ty2 * (u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+3]*vp1 - u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+3]*vm1);
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4] + dy5ty1 * (u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+4] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4] + u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+4]) + yycon3 * (qs[((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))] - 2.0*qs[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] + qs[((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))]) + yycon4 * (vp1*vp1 - 2.0*vijk*vijk + vm1*vm1) + yycon5 * (u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+4]*rhoI[((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4]*rhoI[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] + u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+4]*rhoI[((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))]) - ty2 * ((c1*u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+4] - c2*square[((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))]) * vp1 - (c1*u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+4] - c2*square[((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))]) * vm1);
				}
			}
		}
	}

	private void rhsY_1(int c){
		int i, j, k, m;
		double vijk, vm1, vp1;
		if (start[1][c] > 0){
			for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
				j = 1;
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m]- dssp * ( 5.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+m] + u[5*((i+2) + DI*(((j+2)+2) + DJ*((k+2) + DK*c)))+m]);
					}
				}
				j = 2;
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - dssp * (-4.0*u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+m] + 6.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+m] + u[5*((i+2) + DI*(((j+2)+2) + DJ*((k+2) + DK*c)))+m]);
					}
				}
			}
		}
	}

	private void rhsY_2(int c){
		int i, j, k, m;
		double vijk, vm1, vp1;
		for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
			for (j = 3*start[1][c]; j <= cellSize[1][c]-3*end[1][c]-1; j++){
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - dssp * ( u[5*((i+2) + DI*(((j-2)+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+m] + 6.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+m] + u[5*((i+2) + DI*(((j+2)+2) + DJ*((k+2) + DK*c)))+m] );
					}
				}
			}
		}
	}

	private void rhsY_3(int c){
		int i, j, k, m;
		double vijk, vm1, vp1;
		if (end[1][c] > 0){
			for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
				j = cellSize[1][c]-3;
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - dssp * ( u[5*((i+2) + DI*(((j-2)+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+m] + 6.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*(((j+1)+2) + DJ*((k+2) + DK*c)))+m] );
					}
				}
				j = cellSize[1][c]-2;
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - dssp * ( u[5*((i+2) + DI*(((j-2)+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*(((j-1)+2) + DJ*((k+2) + DK*c)))+m] + 5.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] );
					}
				}
			}
		}
	}

	/* zeta-direction flux differences and dissipation */
	private void rhsZ(int c){
		int i, j, k, m;
		double wijk, wm1, wp1;
		rhsZ_0(c);
		rhsZ_1(c);
		rhsZ_2(c);
		rhsZ_3(c);
	}

	private void rhsZ_0(int c){
		int i, j, k, m;
		double wijk, wm1, wp1;
		for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
			for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					wijk = ws[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))];
					wp1 = ws[((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))];
					wm1 = ws[((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))];
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+0] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+0] + dz1tz1 * (u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+0] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+0] + u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+0]) - tz2 * (u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+3] - u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+3]);
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1] + dz2tz1 * (u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+1] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+1] + u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+1]) + zzcon2 * (us[((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))] - 2.0*us[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] + us[((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))]) - tz2 * (u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+1]*wp1 - u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+1]*wm1);
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2] + dz3tz1 * (u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+2] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+2] + u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+2]) + zzcon2 * (vs[((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))] - 2.0*vs[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] + vs[((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))]) - tz2 * (u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+2]*wp1 - u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+2]*wm1);
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3] + dz4tz1 * (u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+3] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+3] + u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+3]) + zzcon2*con43 * (wp1 - 2.0*wijk + wm1) - tz2 * (u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+3]*wp1 - u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+3]*wm1 + (u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+4] - square[((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))] - u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+4] + square[((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))]) *c2);
					rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4] + dz5tz1 * (u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+4] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4] + u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+4]) + zzcon3 * (qs[((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))] - 2.0*qs[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] + qs[((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))]) + zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + wm1*wm1) + zzcon5 * (u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+4]*rhoI[((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))] - 2.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+4]*rhoI[((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))] + u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+4]*rhoI[((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))]) - tz2 * ( (c1*u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+4] - c2*square[((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))])*wp1 - (c1*u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+4] - c2*square[((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))])*wm1);
				}
			}
		}
	}

	private void rhsZ_1(int c){
		int i, j, k, m;
		double wijk, wm1, wp1;
		if (start[2][c] > 0){
			k = 1;
			for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m]- dssp * ( 5.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+m] + u[5*((i+2) + DI*((j+2) + DJ*(((k+2)+2) + DK*c)))+m]);
					}
				}
			}
			k = 2;
			for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - dssp * (-4.0*u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+m] + 6.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+m] + u[5*((i+2) + DI*((j+2) + DJ*(((k+2)+2) + DK*c)))+m]);
					}
				}
			}
		}
	}

	private void rhsZ_2(int c){
		int i, j, k, m;
		double wijk, wm1, wp1;
		for (k = 3*start[2][c]; k <= cellSize[2][c]-3*end[2][c]-1; k++){
			for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - dssp * ( u[5*((i+2) + DI*((j+2) + DJ*(((k-2)+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+m] + 6.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+m] + u[5*((i+2) + DI*((j+2) + DJ*(((k+2)+2) + DK*c)))+m] );
					}
				}
			}
		}
	}

	private void rhsZ_3(int c){
		int i, j, k, m;
		double wijk, wm1, wp1;
		if (end[2][c] > 0){
			k = cellSize[2][c]-3;
			for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - dssp * ( u[5*((i+2) + DI*((j+2) + DJ*(((k-2)+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+m] + 6.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*((j+2) + DJ*(((k+1)+2) + DK*c)))+m] );
					}
				}
			}
			k = cellSize[2][c]-2;
			for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					for (m = 0; m <= 4; m++){
						rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] - dssp * ( u[5*((i+2) + DI*((j+2) + DJ*(((k-2)+2) + DK*c)))+m] - 4.0*u[5*((i+2) + DI*((j+2) + DJ*(((k-1)+2) + DK*c)))+m] + 5.0*u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] );
					}
				}
			}
		}
	}

	/* The time step: rhs = dt*rhs */
	private void rhsScale(int c){
		int i, j, k, m;
		for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
			for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
				for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
					for (m = 0; m <= 4; m++){
						rhs[5*ix(i,j,k,c)+m] = rhs[5*ix(i,j,k,c)+m] * dt;
					}
				}
			}
		}
	}

	/* Addition of the update to the vector u */
	private void add(){
		int c, i, j, k, m;
		for (c = 0; c < ncells; c++){
			for (k = start[2][c]; k <= cellSize[2][c]-end[2][c]-1; k++){
				for (j = start[1][c]; j <= cellSize[1][c]-end[1][c]-1; j++){
					for (i = start[0][c]; i <= cellSize[0][c]-end[0][c]-1; i++){
						for (m = 0; m <= 4; m++){
							u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] = u[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m] + rhs[5*((i+2) + DI*((j+2) + DJ*((k+2) + DK*c)))+m];
						}
					}
				}
			}
		}
	}

	/*
	 * Jacobians fjac and njac of the points of a line of the cell c (the
	 * original does it inside x_solve_cell, y_solve_cell and z_solve_cell, with
	 * a different association of several products in each direction):
	 * line along i for fixed (j,k), along j for fixed (i,k), along k for fixed (i,j).
	 */
	private void jacobiansX(int c, int j, int k){
		int i;
		double tmp1, tmp2, tmp3;
		for (i = start[0][c]-1; i <= cellSize[0][c] - end[0][c]; i++){
			tmp1 = rhoI[ix(i,j,k,c)];
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			fjac[25*(i+2)+0+5*0] = 0.0;
			fjac[25*(i+2)+0+5*1] = 1.0;
			fjac[25*(i+2)+0+5*2] = 0.0;
			fjac[25*(i+2)+0+5*3] = 0.0;
			fjac[25*(i+2)+0+5*4] = 0.0;
			fjac[25*(i+2)+1+5*0] = -(u[5*ix(i,j,k,c)+1] * tmp2 * u[5*ix(i,j,k,c)+1]) + c2 * qs[ix(i,j,k,c)];
			fjac[25*(i+2)+1+5*1] = ( 2.0 - c2 ) * ( u[5*ix(i,j,k,c)+1] * tmp1 );
			fjac[25*(i+2)+1+5*2] = - c2 * ( u[5*ix(i,j,k,c)+2] * tmp1 );
			fjac[25*(i+2)+1+5*3] = - c2 * ( u[5*ix(i,j,k,c)+3] * tmp1 );
			fjac[25*(i+2)+1+5*4] = c2;
			fjac[25*(i+2)+2+5*0] = - ( u[5*ix(i,j,k,c)+1]*u[5*ix(i,j,k,c)+2] ) * tmp2;
			fjac[25*(i+2)+2+5*1] = u[5*ix(i,j,k,c)+2] * tmp1;
			fjac[25*(i+2)+2+5*2] = u[5*ix(i,j,k,c)+1] * tmp1;
			fjac[25*(i+2)+2+5*3] = 0.0;
			fjac[25*(i+2)+2+5*4] = 0.0;
			fjac[25*(i+2)+3+5*0] = - ( u[5*ix(i,j,k,c)+1]*u[5*ix(i,j,k,c)+3] ) * tmp2;
			fjac[25*(i+2)+3+5*1] = u[5*ix(i,j,k,c)+3] * tmp1;
			fjac[25*(i+2)+3+5*2] = 0.0;
			fjac[25*(i+2)+3+5*3] = u[5*ix(i,j,k,c)+1] * tmp1;
			fjac[25*(i+2)+3+5*4] = 0.0;
			fjac[25*(i+2)+4+5*0] = ( c2 * 2.0 * qs[ix(i,j,k,c)] - c1 * ( u[5*ix(i,j,k,c)+4] * tmp1 ) ) * ( u[5*ix(i,j,k,c)+1] * tmp1 );
			fjac[25*(i+2)+4+5*1] = c1 * u[5*ix(i,j,k,c)+4] * tmp1 - c2 * ( u[5*ix(i,j,k,c)+1]*u[5*ix(i,j,k,c)+1] * tmp2 + qs[ix(i,j,k,c)] );
			fjac[25*(i+2)+4+5*2] = - c2 * ( u[5*ix(i,j,k,c)+2]*u[5*ix(i,j,k,c)+1] ) * tmp2;
			fjac[25*(i+2)+4+5*3] = - c2 * ( u[5*ix(i,j,k,c)+3]*u[5*ix(i,j,k,c)+1] ) * tmp2;
			fjac[25*(i+2)+4+5*4] = c1 * ( u[5*ix(i,j,k,c)+1] * tmp1 );
			njac[25*(i+2)+0+5*0] = 0.0;
			njac[25*(i+2)+0+5*1] = 0.0;
			njac[25*(i+2)+0+5*2] = 0.0;
			njac[25*(i+2)+0+5*3] = 0.0;
			njac[25*(i+2)+0+5*4] = 0.0;
			njac[25*(i+2)+1+5*0] = - con43 * c3c4 * tmp2 * u[5*ix(i,j,k,c)+1];
			njac[25*(i+2)+1+5*1] = con43 * c3c4 * tmp1;
			njac[25*(i+2)+1+5*2] = 0.0;
			njac[25*(i+2)+1+5*3] = 0.0;
			njac[25*(i+2)+1+5*4] = 0.0;
			njac[25*(i+2)+2+5*0] = - c3c4 * tmp2 * u[5*ix(i,j,k,c)+2];
			njac[25*(i+2)+2+5*1] = 0.0;
			njac[25*(i+2)+2+5*2] = c3c4 * tmp1;
			njac[25*(i+2)+2+5*3] = 0.0;
			njac[25*(i+2)+2+5*4] = 0.0;
			njac[25*(i+2)+3+5*0] = - c3c4 * tmp2 * u[5*ix(i,j,k,c)+3];
			njac[25*(i+2)+3+5*1] = 0.0;
			njac[25*(i+2)+3+5*2] = 0.0;
			njac[25*(i+2)+3+5*3] = c3c4 * tmp1;
			njac[25*(i+2)+3+5*4] = 0.0;
			njac[25*(i+2)+4+5*0] = - ( con43 * c3c4 - c1345 ) * tmp3 * ((u[5*ix(i,j,k,c)+1])*(u[5*ix(i,j,k,c)+1])) - ( c3c4 - c1345 ) * tmp3 * ((u[5*ix(i,j,k,c)+2])*(u[5*ix(i,j,k,c)+2])) - ( c3c4 - c1345 ) * tmp3 * ((u[5*ix(i,j,k,c)+3])*(u[5*ix(i,j,k,c)+3])) - c1345 * tmp2 * u[5*ix(i,j,k,c)+4];
			njac[25*(i+2)+4+5*1] = ( con43 * c3c4 - c1345 ) * tmp2 * u[5*ix(i,j,k,c)+1];
			njac[25*(i+2)+4+5*2] = ( c3c4 - c1345 ) * tmp2 * u[5*ix(i,j,k,c)+2];
			njac[25*(i+2)+4+5*3] = ( c3c4 - c1345 ) * tmp2 * u[5*ix(i,j,k,c)+3];
			njac[25*(i+2)+4+5*4] = ( c1345 ) * tmp1;
		}
	}

	private void jacobiansY(int c, int i, int k){
		int j;
		double tmp1, tmp2, tmp3;
		for (j = start[1][c]-1; j <= cellSize[1][c]-end[1][c]; j++){
			utmp[0][j+2] = 1.0 / u[5*ix(i,j,k,c)+0];
			utmp[1][j+2] = u[5*ix(i,j,k,c)+1];
			utmp[2][j+2] = u[5*ix(i,j,k,c)+2];
			utmp[3][j+2] = u[5*ix(i,j,k,c)+3];
			utmp[4][j+2] = u[5*ix(i,j,k,c)+4];
			utmp[5][j+2] = qs[ix(i,j,k,c)];
		}
		for (j = start[1][c]-1; j <= cellSize[1][c]-end[1][c]; j++){
			tmp1 = utmp[0][j+2];
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			fjac[25*(j+2)+0+5*0] = 0.0;
			fjac[25*(j+2)+0+5*1] = 0.0;
			fjac[25*(j+2)+0+5*2] = 1.0;
			fjac[25*(j+2)+0+5*3] = 0.0;
			fjac[25*(j+2)+0+5*4] = 0.0;
			fjac[25*(j+2)+1+5*0] = - ( utmp[1][j+2]*utmp[2][j+2] ) * tmp2;
			fjac[25*(j+2)+1+5*1] = utmp[2][j+2] * tmp1;
			fjac[25*(j+2)+1+5*2] = utmp[1][j+2] * tmp1;
			fjac[25*(j+2)+1+5*3] = 0.0;
			fjac[25*(j+2)+1+5*4] = 0.0;
			fjac[25*(j+2)+2+5*0] = - ( utmp[2][j+2]*utmp[2][j+2]*tmp2) + c2 * utmp[5][j+2];
			fjac[25*(j+2)+2+5*1] = - c2 * utmp[1][j+2] * tmp1;
			fjac[25*(j+2)+2+5*2] = ( 2.0 - c2 ) * utmp[2][j+2] * tmp1;
			fjac[25*(j+2)+2+5*3] = - c2 * utmp[3][j+2] * tmp1;
			fjac[25*(j+2)+2+5*4] = c2;
			fjac[25*(j+2)+3+5*0] = - ( utmp[2][j+2]*utmp[3][j+2] ) * tmp2;
			fjac[25*(j+2)+3+5*1] = 0.0;
			fjac[25*(j+2)+3+5*2] = utmp[3][j+2] * tmp1;
			fjac[25*(j+2)+3+5*3] = utmp[2][j+2] * tmp1;
			fjac[25*(j+2)+3+5*4] = 0.0;
			fjac[25*(j+2)+4+5*0] = ( c2 * 2.0 * utmp[5][j+2] - c1 * utmp[4][j+2] * tmp1 ) * utmp[2][j+2] * tmp1;
			fjac[25*(j+2)+4+5*1] = - c2 * utmp[1][j+2]*utmp[2][j+2] * tmp2;
			fjac[25*(j+2)+4+5*2] = c1 * utmp[4][j+2] * tmp1 - c2 * ( utmp[5][j+2] + utmp[2][j+2]*utmp[2][j+2] * tmp2 );
			fjac[25*(j+2)+4+5*3] = - c2 * ( utmp[2][j+2]*utmp[3][j+2] ) * tmp2;
			fjac[25*(j+2)+4+5*4] = c1 * utmp[2][j+2] * tmp1;
			njac[25*(j+2)+0+5*0] = 0.0;
			njac[25*(j+2)+0+5*1] = 0.0;
			njac[25*(j+2)+0+5*2] = 0.0;
			njac[25*(j+2)+0+5*3] = 0.0;
			njac[25*(j+2)+0+5*4] = 0.0;
			njac[25*(j+2)+1+5*0] = - c3c4 * tmp2 * utmp[1][j+2];
			njac[25*(j+2)+1+5*1] = c3c4 * tmp1;
			njac[25*(j+2)+1+5*2] = 0.0;
			njac[25*(j+2)+1+5*3] = 0.0;
			njac[25*(j+2)+1+5*4] = 0.0;
			njac[25*(j+2)+2+5*0] = - con43 * c3c4 * tmp2 * utmp[2][j+2];
			njac[25*(j+2)+2+5*1] = 0.0;
			njac[25*(j+2)+2+5*2] = con43 * c3c4 * tmp1;
			njac[25*(j+2)+2+5*3] = 0.0;
			njac[25*(j+2)+2+5*4] = 0.0;
			njac[25*(j+2)+3+5*0] = - c3c4 * tmp2 * utmp[3][j+2];
			njac[25*(j+2)+3+5*1] = 0.0;
			njac[25*(j+2)+3+5*2] = 0.0;
			njac[25*(j+2)+3+5*3] = c3c4 * tmp1;
			njac[25*(j+2)+3+5*4] = 0.0;
			njac[25*(j+2)+4+5*0] = - ( c3c4 - c1345 ) * tmp3 * ((utmp[1][j+2])*(utmp[1][j+2])) - ( con43 * c3c4 - c1345 ) * tmp3 * ((utmp[2][j+2])*(utmp[2][j+2])) - ( c3c4 - c1345 ) * tmp3 * ((utmp[3][j+2])*(utmp[3][j+2])) - c1345 * tmp2 * utmp[4][j+2];
			njac[25*(j+2)+4+5*1] = ( c3c4 - c1345 ) * tmp2 * utmp[1][j+2];
			njac[25*(j+2)+4+5*2] = ( con43 * c3c4 - c1345 ) * tmp2 * utmp[2][j+2];
			njac[25*(j+2)+4+5*3] = ( c3c4 - c1345 ) * tmp2 * utmp[3][j+2];
			njac[25*(j+2)+4+5*4] = ( c1345 ) * tmp1;
		}
	}

	private void jacobiansZ(int c, int i, int j){
		int k;
		double tmp1, tmp2, tmp3;
		for (k = start[2][c]-1; k <= cellSize[2][c]-end[2][c]; k++){
			utmp[0][k+2] = 1.0 / u[5*ix(i,j,k,c)+0];
			utmp[1][k+2] = u[5*ix(i,j,k,c)+1];
			utmp[2][k+2] = u[5*ix(i,j,k,c)+2];
			utmp[3][k+2] = u[5*ix(i,j,k,c)+3];
			utmp[4][k+2] = u[5*ix(i,j,k,c)+4];
			utmp[5][k+2] = qs[ix(i,j,k,c)];
		}
		for (k = start[2][c]-1; k <= cellSize[2][c]-end[2][c]; k++){
			tmp1 = utmp[0][k+2];
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			fjac[25*(k+2)+0+5*0] = 0.0;
			fjac[25*(k+2)+0+5*1] = 0.0;
			fjac[25*(k+2)+0+5*2] = 0.0;
			fjac[25*(k+2)+0+5*3] = 1.0;
			fjac[25*(k+2)+0+5*4] = 0.0;
			fjac[25*(k+2)+1+5*0] = - ( utmp[1][k+2]*utmp[3][k+2] ) * tmp2;
			fjac[25*(k+2)+1+5*1] = utmp[3][k+2] * tmp1;
			fjac[25*(k+2)+1+5*2] = 0.0;
			fjac[25*(k+2)+1+5*3] = utmp[1][k+2] * tmp1;
			fjac[25*(k+2)+1+5*4] = 0.0;
			fjac[25*(k+2)+2+5*0] = - ( utmp[2][k+2]*utmp[3][k+2] ) * tmp2;
			fjac[25*(k+2)+2+5*1] = 0.0;
			fjac[25*(k+2)+2+5*2] = utmp[3][k+2] * tmp1;
			fjac[25*(k+2)+2+5*3] = utmp[2][k+2] * tmp1;
			fjac[25*(k+2)+2+5*4] = 0.0;
			fjac[25*(k+2)+3+5*0] = - (utmp[3][k+2]*utmp[3][k+2] * tmp2 ) + c2 * utmp[5][k+2];
			fjac[25*(k+2)+3+5*1] = - c2 * utmp[1][k+2] * tmp1;
			fjac[25*(k+2)+3+5*2] = - c2 * utmp[2][k+2] * tmp1;
			fjac[25*(k+2)+3+5*3] = ( 2.0 - c2 ) * utmp[3][k+2] * tmp1;
			fjac[25*(k+2)+3+5*4] = c2;
			fjac[25*(k+2)+4+5*0] = ( c2 * 2.0 * utmp[5][k+2] - c1 * ( utmp[4][k+2] * tmp1 ) ) * ( utmp[3][k+2] * tmp1 );
			fjac[25*(k+2)+4+5*1] = - c2 * ( utmp[1][k+2]*utmp[3][k+2] ) * tmp2;
			fjac[25*(k+2)+4+5*2] = - c2 * ( utmp[2][k+2]*utmp[3][k+2] ) * tmp2;
			fjac[25*(k+2)+4+5*3] = c1 * ( utmp[4][k+2] * tmp1 ) - c2 * ( utmp[5][k+2] + utmp[3][k+2]*utmp[3][k+2] * tmp2 );
			fjac[25*(k+2)+4+5*4] = c1 * utmp[3][k+2] * tmp1;
			njac[25*(k+2)+0+5*0] = 0.0;
			njac[25*(k+2)+0+5*1] = 0.0;
			njac[25*(k+2)+0+5*2] = 0.0;
			njac[25*(k+2)+0+5*3] = 0.0;
			njac[25*(k+2)+0+5*4] = 0.0;
			njac[25*(k+2)+1+5*0] = - c3c4 * tmp2 * utmp[1][k+2];
			njac[25*(k+2)+1+5*1] = c3c4 * tmp1;
			njac[25*(k+2)+1+5*2] = 0.0;
			njac[25*(k+2)+1+5*3] = 0.0;
			njac[25*(k+2)+1+5*4] = 0.0;
			njac[25*(k+2)+2+5*0] = - c3c4 * tmp2 * utmp[2][k+2];
			njac[25*(k+2)+2+5*1] = 0.0;
			njac[25*(k+2)+2+5*2] = c3c4 * tmp1;
			njac[25*(k+2)+2+5*3] = 0.0;
			njac[25*(k+2)+2+5*4] = 0.0;
			njac[25*(k+2)+3+5*0] = - con43 * c3c4 * tmp2 * utmp[3][k+2];
			njac[25*(k+2)+3+5*1] = 0.0;
			njac[25*(k+2)+3+5*2] = 0.0;
			njac[25*(k+2)+3+5*3] = con43 * c3 * c4 * tmp1;
			njac[25*(k+2)+3+5*4] = 0.0;
			njac[25*(k+2)+4+5*0] = - ( c3c4 - c1345 ) * tmp3 * ((utmp[1][k+2])*(utmp[1][k+2])) - ( c3c4 - c1345 ) * tmp3 * ((utmp[2][k+2])*(utmp[2][k+2])) - ( con43 * c3c4 - c1345 ) * tmp3 * ((utmp[3][k+2])*(utmp[3][k+2])) - c1345 * tmp2 * utmp[4][k+2];
			njac[25*(k+2)+4+5*1] = ( c3c4 - c1345 ) * tmp2 * utmp[1][k+2];
			njac[25*(k+2)+4+5*2] = ( c3c4 - c1345 ) * tmp2 * utmp[2][k+2];
			njac[25*(k+2)+4+5*3] = ( con43 * c3c4 - c1345 ) * tmp2 * utmp[3][k+2];
			njac[25*(k+2)+4+5*4] = ( c1345 )* tmp1;
		}
	}

	/*
	 * The original x_solve, y_solve and z_solve are the same code with the roles
	 * of the three directions interchanged (checked against the three Fortran
	 * files); dir is 0 (x), 1 (y) or 2 (z), and (p,q) are the other two
	 * directions in increasing order: the lines of a cell are the points of
	 * the plane (p,q) and a line runs along dir.
	 */
	private int dirP(int dir){ return dir == 0 ? 1 : 0; }
	private int dirQ(int dir){ return dir == 2 ? 1 : 2; }

	/* Index of the point t of the line (a,b) of the cell c (a along p, b along q) */
	private int pointIx(int dir, int t, int a, int b, int c){
		return dir == 0 ? ix(t,a,b,c) : dir == 1 ? ix(a,t,b,c) : ix(a,b,t,c);
	}

	private void solve(int dir){

		int stage, c, first, last;
		int tagSolve = dir == 0 ? WEST : dir == 1 ? SOUTH : BOTTOM;
		int tagBack = dir == 0 ? EAST : dir == 1 ? NORTH : TOP;

		/* Forward elimination, cell by cell along the direction */
		for(stage = 1; stage<=ncells; stage++){
			c = slice[dir][stage-1];
			last = (stage == ncells) ? 1 : 0;

			if(stage == 1){
				first = 1;
				solveCell(dir, first, last, c);
			}
			else{
				first = 0;
				if(detailedTiming)
					timer.start(T_COMM);
				int tag = tagSolve + (cellCoord[dirP(dir)][c]-1) + (cellCoord[dirQ(dir)][c]-1)*ncells;
				recvReq = commSolve.Irecv(outBuffer, 0, MAX_CELL_DIM*MAX_CELL_DIM*30, MPI.DOUBLE, predecessor[dir], tag);
				Request.Waitall(new Request[] { sendReq, recvReq });
				if(detailedTiming)
					timer.stop(T_COMM);
				unpackSolveInfo(dir, c);
				solveCell(dir, first, last, c);
			}
			if(last == 0)
				sendSolveInfo(dir, c, tagSolve);
		}

		/* Back substitution, in the opposite direction */
		for(stage = ncells; stage>=1; stage--){
			c = slice[dir][stage-1];
			first = 0;
			last = 0;
			if(stage == 1)
				first = 1;
			if(stage == ncells){
				last = 1;
				backsubstitute(dir, first, last, c);
			}
			else{
				if(detailedTiming)
					timer.start(T_COMM);
				int tag = tagBack + (cellCoord[dirP(dir)][c]-1) + (cellCoord[dirQ(dir)][c]-1)*ncells;
				recvReq = commSolve.Irecv(outBuffer, 0, MAX_CELL_DIM*MAX_CELL_DIM*5, MPI.DOUBLE, successor[dir], tag);
				Request.Waitall(new Request[] { sendReq, recvReq });
				if(detailedTiming)
					timer.stop(T_COMM);
				unpackBacksubInfo(dir, c);
				backsubstitute(dir, first, last, c);
			}
			if(first == 0)
				sendBacksubInfo(dir, c, tagBack);
		}
	}

	/* The last plane of the line (lhsc blocks and rhs) goes to the next cell */
	private void sendSolveInfo(int dir, int c, int tagSolve){

		int p = dirP(dir), qd = dirQ(dir);
		int lsize = cellSize[dir][c]-1;
		int ptr = 0;

		for(int b = 0; b<=MAX_CELL_DIM-1; b++){
			for(int a = 0; a<=MAX_CELL_DIM-1; a++){
				int pt = pointIx(dir, lsize, a, b, c);
				for(int m = 0; m<5; m++){
					for(int n = 0; n<5; n++)
						inBuffer[ptr+n] = lhsc[25*pt+m+5*n];
					ptr += 5;
				}
				for(int n = 0; n<5; n++)
					inBuffer[ptr+n] = rhs[5*pt+n];
				ptr += 5;
			}
		}

		if(detailedTiming)
			timer.start(T_COMM);
		int tag = tagSolve + (cellCoord[p][c]-1) + (cellCoord[qd][c]-1)*ncells;
		sendReq = commSolve.Isend(inBuffer, 0, MAX_CELL_DIM*MAX_CELL_DIM*30, MPI.DOUBLE, successor[dir], tag);
		if(detailedTiming)
			timer.stop(T_COMM);
	}

	private void unpackSolveInfo(int dir, int c){

		int ptr = 0;

		for(int b = 0; b<=MAX_CELL_DIM-1; b++){
			for(int a = 0; a<=MAX_CELL_DIM-1; a++){
				int pt = pointIx(dir, -1, a, b, c);
				for(int m = 0; m<5; m++){
					for(int n = 0; n<5; n++)
						lhsc[25*pt+m+5*n] = outBuffer[ptr+n];
					ptr += 5;
				}
				for(int n = 0; n<5; n++)
					rhs[5*pt+n] = outBuffer[ptr+n];
				ptr += 5;
			}
		}
	}

	/* The first plane of the line (rhs) goes to the previous cell */
	private void sendBacksubInfo(int dir, int c, int tagBack){

		int p = dirP(dir), qd = dirQ(dir);
		int ptr = 0;

		for(int b = 0; b<=MAX_CELL_DIM-1; b++){
			for(int a = 0; a<=MAX_CELL_DIM-1; a++){
				int pt = pointIx(dir, 0, a, b, c);
				for(int n = 0; n<5; n++)
					inBuffer[ptr+n] = rhs[5*pt+n];
				ptr += 5;
			}
		}

		if(detailedTiming)
			timer.start(T_COMM);
		int tag = tagBack + (cellCoord[p][c]-1) + (cellCoord[qd][c]-1)*ncells;
		sendReq = commSolve.Isend(inBuffer, 0, MAX_CELL_DIM*MAX_CELL_DIM*5, MPI.DOUBLE, predecessor[dir], tag);
		if(detailedTiming)
			timer.stop(T_COMM);
	}

	private void unpackBacksubInfo(int dir, int c){

		int ptr = 0;

		for(int b = 0; b<=MAX_CELL_DIM-1; b++){
			for(int a = 0; a<=MAX_CELL_DIM-1; a++){
				for(int n = 0; n<5; n++)
					backsubInfo[5*(a+MAXD1*(b+MAXD1*c))+n] = outBuffer[ptr+n];
				ptr += 5;
			}
		}
	}

	private void backsubstitute(int dir, int first, int last, int c){

		int p = dirP(dir), qd = dirQ(dir);
		int lsize = cellSize[dir][c]-1;
		int psize = cellSize[p][c]-end[p][c]-1;
		int qsize = cellSize[qd][c]-end[qd][c]-1;

		if(last == 0){
			for(int b = start[qd][c]; b<=qsize; b++){
				for(int a = start[p][c]; a<=psize; a++){
					int pt = pointIx(dir, lsize, a, b, c);
					for(int m = 0; m<5; m++){
						for(int n = 0; n<5; n++){
							rhs[5*pt+m] = rhs[5*pt+m]
									- lhsc[25*pt+m+5*n]*backsubInfo[5*(a+MAXD1*(b+MAXD1*c))+n];
						}
					}
				}
			}
		}

		for(int b = start[qd][c]; b<=qsize; b++){
			for(int a = start[p][c]; a<=psize; a++){
				for(int t = lsize-1; t>=0; t--){
					int pt = pointIx(dir, t, a, b, c);
					int pt1 = pointIx(dir, t+1, a, b, c);
					for(int m = 0; m<5; m++){
						for(int n = 0; n<5; n++){
							rhs[5*pt+m] = rhs[5*pt+m]
									- lhsc[25*pt+m+5*n]*rhs[5*pt1+n];
						}
					}
				}
			}
		}
	}

	/*
	 * Block-tridiagonal elimination of the lines of the cell c along dir
	 * (x_solve_cell, y_solve_cell and z_solve_cell of the original)
	 */
	private void solveCell(int dir, int first, int last, int c){

		int p = dirP(dir), qd = dirQ(dir);
		int lsize = cellSize[dir][c]-1;
		int psize = cellSize[p][c]-end[p][c]-1;
		int qsize = cellSize[qd][c]-end[qd][c]-1;
		double tc1 = dir == 0 ? tx1 : dir == 1 ? ty1 : tz1;
		double tc2 = dir == 0 ? tx2 : dir == 1 ? ty2 : tz2;
		double dd[] = dir == 0 ? new double[] {dx1, dx2, dx3, dx4, dx5}
				: dir == 1 ? new double[] {dy1, dy2, dy3, dy4, dy5}
				: new double[] {dz1, dz2, dz3, dz4, dz5};
		double tmp1, tmp2;

		lhsabInit(lsize);

		for(int b = start[qd][c]; b<=qsize; b++){
			for(int a = start[p][c]; a<=psize; a++){

				if(dir == 0)
					jacobiansX(c, a, b);
				else if(dir == 1)
					jacobiansY(c, a, b);
				else
					jacobiansZ(c, a, b);

				tmp1 = dt * tc1;
				tmp2 = dt * tc2;

				for(int t = start[dir][c]; t<=lsize-end[dir][c]; t++){
					int pt = pointIx(dir, t, a, b, c);
					for(int n = 0; n<5; n++){
						for(int m = 0; m<5; m++){
							int e = m + 5*n;
							/* lhsa */
							double v = - tmp2 * fjac[25*(t-1+2)+e]
									- tmp1 * njac[25*(t-1+2)+e];
							if(m == n)
								v = v - tmp1 * dd[m];
							lhsa[25*(t+1)+e] = v;
							/* lhsb */
							if(m == n)
								v = 1.0
										+ tmp1 * 2.0 * njac[25*(t+2)+e]
										+ tmp1 * 2.0 * dd[m];
							else
								v = tmp1 * 2.0 * njac[25*(t+2)+e];
							lhsb[25*(t+1)+e] = v;
							/* lhsc */
							v = tmp2 * fjac[25*(t+1+2)+e]
									- tmp1 * njac[25*(t+1+2)+e];
							if(m == n)
								v = v - tmp1 * dd[m];
							lhsc[25*pt+e] = v;
						}
					}
				}

				/* The elimination: first and last cells of the line are different */
				if(first == 1){
					int pt = pointIx(dir, 0, a, b, c);
					binvcrhs(lhsb, 25*(0+1), lhsc, 25*pt, rhs, 5*pt);
				}

				for(int t = first; t<=lsize-last; t++){
					int pt = pointIx(dir, t, a, b, c);
					int ptm = pointIx(dir, t-1, a, b, c);
					matvecSub(lhsa, 25*(t+1), rhs, 5*ptm, rhs, 5*pt);
					matmulSub(lhsa, 25*(t+1), lhsc, 25*ptm, lhsb, 25*(t+1));
					binvcrhs(lhsb, 25*(t+1), lhsc, 25*pt, rhs, 5*pt);
				}

				if(last == 1){
					int pt = pointIx(dir, lsize, a, b, c);
					int ptm = pointIx(dir, lsize-1, a, b, c);
					matvecSub(lhsa, 25*(lsize+1), rhs, 5*ptm, rhs, 5*pt);
					matmulSub(lhsa, 25*(lsize+1), lhsc, 25*ptm, lhsb, 25*(lsize+1));
					binvrhs(lhsb, 25*(lsize+1), rhs, 5*pt);
				}
			}
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

	/*
	 * This method computes the norm of the difference between the
	 * computed solution and the exact solution
	 */
	private void errorNorm(double rms[]){

		int c, i, j, k, m, d, ii, jj, kk;
		double xi, eta, zeta, add;
		double uExact[] = new double[5];
		double rmsWork[] = new double[5];

		for(m = 0; m<5; m++){
			rmsWork[m] = 0.0;
		}

		for(c = 0; c<ncells; c++){
			kk = 0;
			for(k = cellLow[2][c]; k<=cellHigh[2][c]; k++){
				zeta = (double)k * dnzm1;
				jj = 0;
				for(j = cellLow[1][c]; j<=cellHigh[1][c]; j++){
					eta = (double)j * dnym1;
					ii = 0;
					for(i = cellLow[0][c]; i<=cellHigh[0][c]; i++){
						xi = (double)i * dnxm1;
						exactSolution(xi, eta, zeta, uExact);
						for(m = 0; m<5; m++){
							add = u[5*ix(ii,jj,kk,c)+m]-uExact[m];
							rmsWork[m] = rmsWork[m] + add*add;
						}
						ii = ii + 1;
					}
					jj = jj + 1;
				}
				kk = kk + 1;
			}
		}

		commWork.Allreduce(rmsWork, 0, rms, 0, 5, MPI.DOUBLE, MPI.SUM);

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

		int c, i, j, k, d, m;
		double add;
		double rmsWork[] = new double[5];

		for(m = 0; m<5; m++){
			rmsWork[m] = 0.0;
		}

		for(c = 0; c<ncells; c++){
			for(k = start[2][c]; k<=cellSize[2][c]-end[2][c]-1; k++){
				for(j = start[1][c]; j<=cellSize[1][c]-end[1][c]-1; j++){
					for(i = start[0][c]; i<=cellSize[0][c]-end[0][c]-1; i++){
						for(m = 0; m<5; m++){
							add = rhs[5*ix(i,j,k,c)+m];
							rmsWork[m] = rmsWork[m] + add*add;
						}
					}
				}
			}
		}

		commWork.Allreduce(rmsWork, 0, rms, 0, 5, MPI.DOUBLE, MPI.SUM);

		for(m = 0; m<5; m++){
			for(d = 0; d<3; d++){
				rms[m] = rms[m] / (double)(gridPoints[d]-2);
			}
			rms[m] = Math.sqrt(rms[m]);
		}
	}

	private int verify(char CLASS){

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

		copyFaces();

		rhsNorm(xcr);

		for(int m = 0; m<5; m++){
			xcr[m] = xcr[m]/dt;
		}

		/* only the root process verifies and prints */
		if(myRank != 0)
			return 0;

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
		else if(CLASS == 'E'){
			dtref = 0.4e-5;
			xcrref[0] = 0.9795372484517e+05;
			xcrref[1] = 0.9739814511521e+04;
			xcrref[2] = 0.2467606342965e+05;
			xcrref[3] = 0.2092419572860e+05;
			xcrref[4] = 0.1392138856939e+06;
			xceref[0] = 0.4327562208414e+03;
			xceref[1] = 0.3699051964887e+02;
			xceref[2] = 0.1089845040954e+03;
			xceref[3] = 0.9462517622043e+02;
			xceref[4] = 0.7765512765309e+03;
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
}
