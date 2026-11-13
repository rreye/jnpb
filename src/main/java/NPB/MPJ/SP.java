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

import mpi.Comm;
import mpi.Intracomm;
import mpi.MPI;
import mpi.Request;

import NPB.Benchmark;
import NPB.common.io.BMArgs;
import NPB.common.io.BMResults;
import NPB.common.numerics.Misc;
import NPB.common.timer.Timer;
import NPB.exceptions.BadClassException;
import NPB.exceptions.BadInputException;
import NPB.exceptions.IncorrectNumProcessException;
import NPB.exceptions.NPBException;

public class SP implements Benchmark{

	private static final String BM_NAME="SP";
	private char CLASS;
	private boolean detailedTiming;

	private int problemSize;
	private int niter;
	private int maxCells;
	private double dt;

	private int gridPoints[] = new int[3];
	private int ncells;

	private int cellCoord[][];
	private int cellHigh[][];
	private int cellLow[][];
	private int cellSize[][];
	private int slice[][];

	private int predecessor[] = new int[3];
	private int successor[] = new int[3];

	private int MAX_CELL_DIM;
	private int IMAX;
	private int JMAX;
	private int KMAX;

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

	private double u[];
	private double lhs[];
	private double forcing[];
	private int start[][];
	private int end[][];
	private double ue[][];
	private double buf[][];
	private double cuf[];
	private double q[];

	private int westSize, eastSize, bottomSize, topSize;
	private int northSize, southSize;
	private int startSendWest, startSendEast, startSendSouth;
	private int startSendNorth, startSendBottom, startSendTop;
	private int startRecvWest, startRecvEast, startRecvSouth;
	private int startRecvNorth, startRecvBottom, startRecvTop;

	private double inBuffer[], outBuffer[];
	private int BUF_SIZE;

	private int EAST = 2000;
	private int WEST = 3000;
	private int NORTH = 4000;
	private int SOUTH = 5000;
	private int BOTTOM = 6000;
	private int TOP = 7000;

	private double rhoI[];
	private double us[];
	private double vs[];
	private double ws[];
	private double square[];
	private double qs[];
	private double speed[];
	private double ainv[];

	private double rhs[];
	private double cv[];
	private double rhon[];
	private double rhoq[];
	private double rhos[];

	private int DIMI, DIMI_DIMJ, DIMI_DIMJ_DIMK, DIMI_DIMJ_DIMK_DIMM;
	private int DIMI_FORC, DIMI_DIMJ_FORC, DIMI_DIMJ_DIMK_FORC, DIMI_DIMJ_DIMK_DIMM_FORC;
	private int DIMI_4D, DIMI_DIMJ_4D, DIMI_DIMJ_DIMK_4D, DIMI_DIMJ_DIMK_DIMM_LHS;

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
	private int numProcess; /* active processes (a power of two / a square) */
	private int npTotal; /* processes launched */
	private boolean active = true; /* false for the excess processes masked out */
	private Intracomm commWork; /* communicator of the active processes */
	private Comm commSolve;
	private Comm commRHS;
	private final int DEFAULT_TAG = 0;

	public static void main(String[] args) throws NPBException {
		SP sp = null;

		BMArgs.parseCmdLineArgs(args, BM_NAME);
		try {
			sp = new SP(BMArgs.CLASS, BMArgs.DETAILED_TIMING);
		} catch (OutOfMemoryError e) {
			BMArgs.outOfMemoryMessage();
			throw e;
		}
		sp.runBenchmark(args);
	}

	public BMResults getResults(){
		return res;
	}

	public SP(char classSize, boolean dtiming) {

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
		case 'E':
			problemSize = 1020;
			niter = 500;
			dt = 0.0001;
			break;
		default:
			terminate();
			throw new BadClassException("Specified class (" + CLASS + ") not valid for this benchmark\n"
					+ "Valid classes are S, W, A, B, C, D and E");
		}

		maxCells = Misc.isqrt(numProcess);
		cellCoord = new int[3][maxCells];
		cellHigh = new int[3][maxCells];
		cellLow = new int[3][maxCells];
		cellSize = new int[3][maxCells];
		slice = new int[3][maxCells];
		MAX_CELL_DIM = (problemSize/maxCells)+1;
		IMAX = MAX_CELL_DIM;
		JMAX = MAX_CELL_DIM;
		KMAX = MAX_CELL_DIM;
		u = new double[(IMAX+4)*(JMAX+4)*(KMAX+4)*5*maxCells];
		start = new int[3][maxCells];
		end = new int[3][maxCells];
		lhs = new double[(IMAX+4)*(JMAX+4)*(KMAX+4)*15*maxCells];
		forcing = new double[IMAX*JMAX*KMAX*5*maxCells];
		ue = new double[MAX_CELL_DIM+4][5];
		buf = new double[MAX_CELL_DIM+4][5];
		cuf = new double[MAX_CELL_DIM+4];
		q = new double[MAX_CELL_DIM+4];
		BUF_SIZE = MAX_CELL_DIM*MAX_CELL_DIM*(maxCells-1)*60*2+1;
		inBuffer = new double[BUF_SIZE];
		outBuffer = new double[BUF_SIZE];
		rhoI = new double[(IMAX+2)*(JMAX+2)*(KMAX+2)*maxCells];
		us = new double[(IMAX+2)*(JMAX+2)*(KMAX+2)*maxCells];
		vs = new double[(IMAX+2)*(JMAX+2)*(KMAX+2)*maxCells];
		ws = new double[(IMAX+2)*(JMAX+2)*(KMAX+2)*maxCells];
		square = new double[(IMAX+2)*(JMAX+2)*(KMAX+2)*maxCells];
		qs = new double[(IMAX+2)*(JMAX+2)*(KMAX+2)*maxCells];
		speed = new double[(IMAX+2)*(JMAX+2)*(KMAX+2)*maxCells];
		ainv = new double[(IMAX+2)*(JMAX+2)*(KMAX+2)*maxCells];
		rhs = new double[(IMAX+4)*(JMAX+4)*(KMAX+4)*5*maxCells];
		cv = new double[MAX_CELL_DIM+4];
		rhon = new double[MAX_CELL_DIM+4];
		rhoq = new double[MAX_CELL_DIM+4];
		rhos = new double[MAX_CELL_DIM+4];
		DIMI = IMAX+4;
		DIMI_DIMJ = DIMI*(JMAX+4);
		DIMI_DIMJ_DIMK = DIMI_DIMJ*(KMAX+4);
		DIMI_DIMJ_DIMK_DIMM = DIMI_DIMJ_DIMK*5;
		DIMI_FORC = IMAX;
		DIMI_DIMJ_FORC = DIMI_FORC*JMAX;
		DIMI_DIMJ_DIMK_FORC = DIMI_DIMJ_FORC*KMAX;
		DIMI_DIMJ_DIMK_DIMM_FORC = DIMI_DIMJ_DIMK_FORC*5;
		DIMI_DIMJ_DIMK_DIMM_LHS = DIMI_DIMJ_DIMK*15;
		DIMI_4D = IMAX+2;
		DIMI_DIMJ_4D = DIMI_4D*(JMAX+2);
		DIMI_DIMJ_DIMK_4D = DIMI_DIMJ_4D*(KMAX+2);		
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

		if(myRank == 0){
			System.out.println(" Size:  "+gridPoints[0]+"x "+gridPoints[1]+"x "+gridPoints[2]);
			System.out.println(" Iterations:  "+niter+"    dt:  "+dt);
			System.out.println(" Total number of processes:  "+npTotal);
			if (numProcess != npTotal)
				System.out.println(" WARNING: Number of processes is not a square ("+numProcess+" active)");
			System.out.println();
		}

		makeSet();

		for(int i = 0; i<ncells; i++){
			if( cellSize[0][i] > IMAX ||
					cellSize[1][i] > JMAX ||
					cellSize[2][i] > KMAX ){
				System.err.println(myRank+" "+i+" "+cellSize[0][i]+" "+cellSize[1][i]+" "+cellSize[2][i]);
				System.err.println(" Problem size too big for compiled array sizes");
				terminate();
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

		timer.start(T_COMP);

		for(int step = 1; step<=niter; step++){
			if(myRank == 0){
				if(step == 1 || step%20 == 0)
					System.out.println(" Time step "+step);
			}
			adi();			
		}

		timer.stop(T_COMP);

		double times[] = { timer.read(T_COMP), timer.read(T_COMM) };
		double maxtimes[] = {0,0};

		int verified = verify(niter, CLASS);

		commWork.Reduce(times, 0, maxtimes, 0, 2, MPI.DOUBLE, MPI.MAX, 0);

		if(myRank == 0){
			double mflops = 0;
			if(maxtimes[0] != 0)
				mflops = (881.174*Math.pow(problemSize,3)
						-4683.91*Math.pow(problemSize,2)
						+11484.5*problemSize
						-19272.4)*niter/(maxtimes[0]*1000000);

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

				BufferedReader input = new BufferedReader(new FileReader("inputsp.data"));

				System.out.println(" Reading from input file inputsp.data");

				String line[] = input.readLine().split(" ");
				niter = Integer.valueOf(line[0]);

				line = input.readLine().split(" ");
				/* the sample file writes dt in Fortran form (0.0015d0) */
				dt = Double.valueOf(line[0].replace('d', 'e').replace('D', 'E'));

				line = input.readLine().split(" ");
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
				System.out.println(" No input file inputsp.data. Using compiled defaults");
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
	 * This method initializes the field variable u using
	 * tri-linear transfinite interpolation of the boundary values
	 */
	private void initialize(){

		int c, ii, jj, kk;
		double xi,eta,zeta;
		double pface[][][] = new double[3][2][5];
		double temp[] = new double[5];

		/*
		 * Later (in computeRhs) we compute 1/u for every element. A few of 
		 * the corner elements are not used, but it convenient (and faster) 
		 * to compute the whole thing with a simple loop. Make sure those 
		 * values are nonzero by initializing the whole thing here.
		 */
		for(c = 0; c<ncells;c++){
			for(kk = 1; kk<=IMAX+1; kk++){
				for(jj = 1; jj<=IMAX+1; jj++){
					for(ii = 1; ii<=IMAX+1; ii++){
						u[position5D(ii,jj,kk,0,c)] = 1;
						u[position5D(ii,jj,kk,1,c)] = 0;
						u[position5D(ii,jj,kk,2,c)] = 0;
						u[position5D(ii,jj,kk,3,c)] = 0;
						u[position5D(ii,jj,kk,4,c)] = 1;
					}
				}
			}
		}

		/*
		 * First store the "interpolated" values everywhere on the grid
		 */
		for(c = 0; c<ncells; c++){
			kk = 2;
			for(int k = cellLow[2][c]; k<=cellHigh[2][c]; k++){
				zeta = k*dnzm1;
				jj = 2;
				for(int j = cellLow[1][c]; j<=cellHigh[1][c]; j++){
					eta = j*dnym1;
					ii = 2;
					for(int i = cellLow[0][c]; i<=cellHigh[0][c]; i++){
						xi = i*dnxm1;
						for(int ix = 0; ix<2; ix++){
							exactSolution(ix,eta,zeta,pface[0][ix]);
						}
						for(int iy = 0; iy<2; iy++){
							exactSolution(xi,iy,zeta,pface[1][iy]);
						}
						for(int iz = 0; iz<2; iz++){
							exactSolution(xi,eta,iz,pface[2][iz]);
						}
						for(int m = 0; m<5; m++){
							double pxi = xi * pface[0][1][m] + (1-xi) * pface[0][0][m];
							double peta = eta * pface[1][1][m] + (1-eta) * pface[1][0][m];
							double pzeta = zeta * pface[2][1][m] + (1-zeta) * pface[2][0][m];
							u[position5D(ii,jj,kk,m,c)] = pxi + peta + pzeta -
							pxi*peta - pxi*pzeta - peta*pzeta +
							pxi*peta*pzeta;
						}
						ii++;
					}
					jj++;	
				}
				kk++;	
			}
		}

		/*
		 * Now store the exact values on the boundaries
		 */
		/*
		 * West face
		 */
		c = slice[0][0];
		ii = 2;
		xi = 0;
		kk = 2;
		for(int k = cellLow[2][c]; k<=cellHigh[2][c]; k++){
			zeta = k*dnzm1;
			jj = 2;
			for(int j = cellLow[1][c]; j<=cellHigh[1][c]; j++){
				eta = j*dnym1;
				exactSolution(xi,eta,zeta,temp);
				for(int m = 0; m<5; m++){
					u[position5D(ii,jj,kk,m,c)] = temp[m];
				}
				jj++;
			}
			kk++;
		}

		/*
		 * East face
		 */
		c = slice[0][ncells-1];
		ii = cellSize[0][c]+1;
		xi = 1;
		kk = 2;
		for(int k = cellLow[2][c]; k<=cellHigh[2][c]; k++){
			zeta = k*dnzm1;
			jj = 2;
			for(int j = cellLow[1][c]; j<=cellHigh[1][c]; j++){
				eta = j*dnym1;
				exactSolution(xi,eta,zeta,temp);
				for(int m = 0; m<5; m++){
					u[position5D(ii,jj,kk,m,c)] = temp[m];
				}
				jj++;
			}
			kk++;
		}

		/*
		 * South face
		 */
		c = slice[1][0];
		jj = 2;
		eta = 0;
		kk = 2;
		for(int k = cellLow[2][c]; k<=cellHigh[2][c]; k++){
			zeta = k*dnzm1;
			ii = 2;
			for(int i = cellLow[0][c]; i<=cellHigh[0][c]; i++){
				xi = i*dnxm1;
				exactSolution(xi,eta,zeta,temp);
				for(int m = 0; m<5; m++){
					u[position5D(ii,jj,kk,m,c)] = temp[m];
				}
				ii++;
			}
			kk++;
		}

		/*
		 * North face
		 */
		c = slice[1][ncells-1];
		jj = cellSize[1][c]+1;
		eta = 1;
		kk = 2;
		for(int k = cellLow[2][c]; k<=cellHigh[2][c]; k++){
			zeta = k*dnzm1;
			ii = 2;
			for(int i = cellLow[0][c]; i<=cellHigh[0][c]; i++){
				xi = i*dnxm1;
				exactSolution(xi,eta,zeta,temp);
				for(int m = 0; m<5; m++){
					u[position5D(ii,jj,kk,m,c)] = temp[m];					
				}
				ii++;
			}
			kk++;
		}

		/*
		 * Bottom face
		 */
		c = slice[2][0];
		kk = 2;
		zeta = 0;
		ii = 2;
		for(int i = cellLow[0][c]; i<=cellHigh[0][c]; i++){
			xi = i*dnxm1;
			jj = 2;
			for(int j = cellLow[1][c]; j<=cellHigh[1][c]; j++){
				eta = j*dnym1;
				exactSolution(xi,eta,zeta,temp);
				for(int m = 0; m<5; m++){
					u[position5D(ii,jj,kk,m,c)] = temp[m];
				}
				jj++;
			}
			ii++;
		}

		/*
		 * Top face
		 */
		c = slice[2][ncells-1];
		kk = cellSize[2][c]+1;
		zeta = 1;
		ii = 2;
		for(int i = cellLow[0][c]; i<=cellHigh[0][c]; i++){
			xi = i*dnxm1;
			jj = 2;
			for(int j = cellLow[1][c]; j<=cellHigh[1][c]; j++){
				eta = j*dnym1;
				exactSolution(xi,eta,zeta,temp);
				for(int m = 0; m<5; m++){
					u[position5D(ii,jj,kk,m,c)] = temp[m];
				}
				jj++;
			}
			ii++;
		}

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

	private void lhsInit(){

		/*
		 * Loop over all cells
		 */
		for(int c = 0; c<ncells; c++){
			/*
			 * First, initialize the start and end arrays
			 */
			for(int d = 0; d<3; d++){
				if(cellCoord[d][c] == 1){
					start[d][c] = 1;
				}
				else{
					start[d][c] = 0;
				}
				if(cellCoord[d][c] == ncells){
					end[d][c] = 1;
				}
				else{
					end[d][c] = 0;
				}
			}

			/*
			 * Zap the whole left hand side for starters
			 */
			for(int n = 0; n<15; n++){
				for(int k = 2; k<cellSize[2][c]+2; k++){
					for(int j = 2; j<cellSize[1][c]+2; j++){
						for(int i = 2; i<cellSize[0][c]+2; i++){
							lhs[lhsPosition5D(i,j,k,n,c)] = 0;
						}	
					}	
				}
			}

			/*
			 * Next, set all diagonal values to 1. This is overkill, but convenient
			 */
			for(int n = 1; n<=3; n++){
				int n53 = 5*n-3;
				for(int k = 2; k<cellSize[2][c]+2; k++){
					for(int j = 2; j<cellSize[1][c]+2; j++){
						for(int i = 2; i<cellSize[0][c]+2; i++){
							lhs[lhsPosition5D(i,j,k,n53,c)] = 1;
						}	
					}	
				}
			}

		}

	}

	/*
	 * Compute the right hand side based on exact solution
	 */
	private void exactRHS(){

		double zeta, eta, xi;
		double dtemp[] = new double[5];

		/* Loop over all cells owned by this node */
		for(int c = 0; c<ncells; c++){

			/* Initialize */
			for(int m = 0; m<5; m++){
				for(int k = 0; k<=cellSize[2][c]-1; k++){
					for(int j = 0; j<=cellSize[1][c]-1; j++){
						for(int i = 0; i<=cellSize[0][c]-1; i++){
							forcing[forcPosition5D(i,j,k,m,c)] = 0;
						}
					}
				}
			}

			/* 
			 * xi-direction flux differences
			 */
			for(int k = start[2][c]; k<=cellSize[2][c]-end[2][c]-1; k++){
				zeta = (k+cellLow[2][c])*dnzm1;
				for(int j = start[1][c]; j<=cellSize[1][c]-end[1][c]-1; j++){
					eta = (j+cellLow[1][c])*dnym1;
					for(int i = 2*start[0][c]; i<=cellSize[0][c]+1+2*(1-end[0][c]); i++){
						xi = (i-2+cellLow[0][c])*dnxm1;

						exactSolution(xi, eta, zeta, dtemp);

						for(int m = 0; m<5; m++){
							ue[i][m] = dtemp[m];
						}

						double dtpp = 1.0d/dtemp[0];

						for(int m = 1; m<5; m++){
							buf[i][m] = dtpp * dtemp[m];
						}

						cuf[i] = buf[i][1] * buf[i][1];
						buf[i][0] = cuf[i] + buf[i][2] * buf[i][2] + buf[i][3] * buf[i][3];
						q[i] = 0.5*(buf[i][1]*ue[i][1] + buf[i][2]*ue[i][2] + buf[i][3]*ue[i][3]);

					}

					for(int i = start[0][c]; i<=cellSize[0][c]-end[0][c]-1; i++){
						int im1 = i+1;
						int ip1 = i+3;
						int i2 = i+2;

						forcing[forcPosition5D(i,j,k,0,c)] = forcing[forcPosition5D(i,j,k,0,c)]-tx2*(ue[ip1][1]-ue[im1][1]) +
						dx1tx1*(ue[ip1][0]-2*ue[i2][0]+ue[im1][0]);

						forcing[forcPosition5D(i,j,k,1,c)] = forcing[forcPosition5D(i,j,k,1,c)]-tx2*(
								(ue[ip1][1]*buf[ip1][1] + c2*(ue[ip1][4]-q[ip1]))-
								(ue[im1][1]*buf[im1][1] + c2*(ue[im1][4]-q[im1])))+
								xxcon1*(buf[ip1][1]-2*buf[i2][1]+buf[im1][1])+
								dx2tx1*( ue[ip1][1]-2* ue[i2][1]+ ue[im1][1]);

						forcing[forcPosition5D(i,j,k,2,c)] = forcing[forcPosition5D(i,j,k,2,c)]-tx2*(
								ue[ip1][2]*buf[ip1][1]-ue[im1][2]*buf[im1][1])+
								xxcon2*(buf[ip1][2]-2*buf[i2][2]+buf[im1][2])+
								dx3tx1*( ue[ip1][2]-2* ue[i2][2]+ ue[im1][2]);

						forcing[forcPosition5D(i,j,k,3,c)] = forcing[forcPosition5D(i,j,k,3,c)]-tx2*(
								ue[ip1][3]*buf[ip1][1]-ue[im1][3]*buf[im1][1])+
								xxcon2*(buf[ip1][3]-2*buf[i2][3]+buf[im1][3])+
								dx4tx1*( ue[ip1][3]-2* ue[i2][3]+ ue[im1][3]);

						forcing[forcPosition5D(i,j,k,4,c)] = forcing[forcPosition5D(i,j,k,4,c)]-tx2*(
								buf[ip1][1]*(c1*ue[ip1][4]-c2*q[ip1])-
								buf[im1][1]*(c1*ue[im1][4]-c2*q[im1]))+
								0.5*xxcon3*(buf[ip1][0]-2*buf[i2][0]+buf[im1][0])+
								xxcon4*(cuf[ip1]-2*cuf[i2]+cuf[im1])+
								xxcon5*(buf[ip1][4]-2*buf[i2][4]+buf[im1][4])+
								dx5tx1*(ue[ip1][4]-2*ue[i2][4]+ue[im1][4]);

					}

					/*
					 * Fourth-order dissipation
					 */
					 if(start[0][c] > 0){
						 for(int m = 0; m<5; m++){
							 int i = 1;
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(5*ue[i+2][m] - 4*ue[i+3][m] + ue[i+4][m]);
							 i = 2;
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(-4*ue[i+1][m] + 6*ue[i+2][m] - 4*ue[i+3][m] + ue[i+4][m]);
						 }
					 }

					 for(int m = 0; m<5; m++){
						 for(int i = start[0][c]*3; i<=cellSize[0][c]-3*end[0][c]-1; i++){
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(ue[i][m] - 4*ue[i+1][m] + 6*ue[i+2][m] - 4*ue[i+3][m] + ue[i+4][m]);
						 }
					 }

					 if(end[0][c] > 0){
						 for(int m = 0; m<5; m++){
							 int i = cellSize[0][c]-3;
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(ue[i][m] - 4*ue[i+1][m] + 6*ue[i+2][m] - 4*ue[i+3][m]);
							 i = cellSize[0][c]-2;
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(ue[i][m] - 4*ue[i+1][m] + 5*ue[i+2][m]);
						 }
					 }

				}
			}

			/*
			 * eta-direction flux differences
			 */
			for(int k = start[2][c]; k<=cellSize[2][c]-end[2][c]-1; k++){
				zeta = (k+cellLow[2][c])*dnzm1;
				for(int i = start[0][c]; i<=cellSize[0][c]-end[0][c]-1; i++){
					xi = (i+cellLow[0][c])*dnxm1;
					for(int j = 2*start[1][c]; j<=cellSize[1][c]+1+2*(1-end[1][c]); j++){
						eta = (j-2+cellLow[1][c])*dnym1;

						exactSolution(xi, eta, zeta, dtemp);

						for(int m = 0; m<5; m++){
							ue[j][m] = dtemp[m];
						}

						double dtpp = 1.0d/dtemp[0];

						for(int m = 1; m<5; m++){
							buf[j][m] = dtpp * dtemp[m];
						}

						cuf[j] = buf[j][2] * buf[j][2];
						buf[j][0] = cuf[j] + buf[j][1] * buf[j][1] + buf[j][3] * buf[j][3];
						q[j] = 0.5*(buf[j][1]*ue[j][1] + buf[j][2]*ue[j][2] + buf[j][3]*ue[j][3]);

					}

					for(int j = start[1][c]; j<=cellSize[1][c]-end[1][c]-1; j++){
						int jm1 = j+1;
						int jp1 = j+3;
						int j2 = j+2;

						forcing[forcPosition5D(i,j,k,0,c)] = forcing[forcPosition5D(i,j,k,0,c)]-ty2*(ue[jp1][2]-ue[jm1][2]) +
						dy1ty1*(ue[jp1][0]-2*ue[j2][0]+ue[jm1][0]);

						forcing[forcPosition5D(i,j,k,1,c)] = forcing[forcPosition5D(i,j,k,1,c)]-ty2*
						(ue[jp1][1]*buf[jp1][2] - ue[jm1][1]*buf[jm1][2])+
						yycon2*(buf[jp1][1]-2*buf[j2][1]+buf[jm1][1])+
						dy2ty1*( ue[jp1][1]-2* ue[j2][1]+ ue[jm1][1]);

						forcing[forcPosition5D(i,j,k,2,c)] = forcing[forcPosition5D(i,j,k,2,c)]-ty2*(
								(ue[jp1][2]*buf[jp1][2] + c2*(ue[jp1][4]-q[jp1]))-
								(ue[jm1][2]*buf[jm1][2] + c2*(ue[jm1][4]-q[jm1])))+
								yycon1*(buf[jp1][2]-2*buf[j2][2]+buf[jm1][2])+
								dy3ty1*( ue[jp1][2]-2* ue[j2][2]+ ue[jm1][2]);

						forcing[forcPosition5D(i,j,k,3,c)] = forcing[forcPosition5D(i,j,k,3,c)]-ty2*(
								ue[jp1][3]*buf[jp1][2]-ue[jm1][3]*buf[jm1][2])+
								yycon2*(buf[jp1][3]-2*buf[j2][3]+buf[jm1][3])+
								dy4ty1*( ue[jp1][3]-2* ue[j2][3]+ ue[jm1][3]);

						forcing[forcPosition5D(i,j,k,4,c)] = forcing[forcPosition5D(i,j,k,4,c)]-ty2*(
								buf[jp1][2]*(c1*ue[jp1][4]-c2*q[jp1])-
								buf[jm1][2]*(c1*ue[jm1][4]-c2*q[jm1]))+
								0.5*yycon3*(buf[jp1][0]-2*buf[j2][0]+buf[jm1][0])+
								yycon4*(cuf[jp1]-2*cuf[j2]+cuf[jm1])+
								yycon5*(buf[jp1][4]-2*buf[j2][4]+buf[jm1][4])+
								dy5ty1*(ue[jp1][4]-2*ue[j2][4]+ue[jm1][4]);

					}

					/*
					 * Fourth-order dissipation
					 */
					 if(start[1][c] > 0){
						 for(int m = 0; m<5; m++){
							 int j = 1;
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(5*ue[j+2][m] - 4*ue[j+3][m] + ue[j+4][m]);
							 j = 2;
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(-4*ue[j+1][m] + 6*ue[j+2][m] - 4*ue[j+3][m] + ue[j+4][m]);
						 }
					 }

					 for(int m = 0; m<5; m++){
						 for(int j = start[1][c]*3; j<=cellSize[1][c]-3*end[1][c]-1; j++){
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(ue[j][m] - 4*ue[j+1][m] + 6*ue[j+2][m] - 4*ue[j+3][m] + ue[j+4][m]);
						 }
					 }

					 if(end[1][c] > 0){
						 for(int m = 0; m<5; m++){
							 int j = cellSize[1][c]-3;
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(ue[j][m] - 4*ue[j+1][m] + 6*ue[j+2][m] - 4*ue[j+3][m]);
							 j = cellSize[1][c]-2;
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(ue[j][m] - 4*ue[j+1][m] + 5*ue[j+2][m]);
						 }
					 }

				}
			}

			/*
			 * zeta-direction flux differences
			 */
			for(int j = start[1][c]; j<=cellSize[1][c]-end[1][c]-1; j++){
				eta = (j+cellLow[1][c])*dnym1;
				for(int i = start[0][c]; i<=cellSize[0][c]-end[0][c]-1; i++){
					xi = (i+cellLow[0][c])*dnxm1;
					for(int k = 2*start[2][c]; k<=cellSize[2][c]+1+2*(1-end[2][c]); k++){
						zeta = (k-2+cellLow[2][c])*dnzm1;

						exactSolution(xi, eta, zeta, dtemp);

						for(int m = 0; m<5; m++){
							ue[k][m] = dtemp[m];
						}

						double dtpp = 1.0d/dtemp[0];

						for(int m = 1; m<5; m++){
							buf[k][m] = dtpp * dtemp[m];
						}

						cuf[k] = buf[k][3] * buf[k][3];
						buf[k][0] = cuf[k] + buf[k][1] * buf[k][1] + buf[k][2] * buf[k][2];
						q[k] = 0.5*(buf[k][1]*ue[k][1] + buf[k][2]*ue[k][2] + buf[k][3]*ue[k][3]);

					}

					for(int k = start[2][c]; k<=cellSize[2][c]-end[2][c]-1; k++){
						int km1 = k+1;
						int kp1 = k+3;
						int k2 = k+2;

						forcing[forcPosition5D(i,j,k,0,c)] = forcing[forcPosition5D(i,j,k,0,c)]-tz2*(ue[kp1][3]-ue[km1][3]) +
						dz1tz1*(ue[kp1][0]-2*ue[k2][0]+ue[km1][0]);

						forcing[forcPosition5D(i,j,k,1,c)] = forcing[forcPosition5D(i,j,k,1,c)]-tz2*
						(ue[kp1][1]*buf[kp1][3] - ue[km1][1]*buf[km1][3])+
						zzcon2*(buf[kp1][1]-2*buf[k2][1]+buf[km1][1])+
						dz2tz1*( ue[kp1][1]-2* ue[k2][1]+ ue[km1][1]);

						forcing[forcPosition5D(i,j,k,2,c)] = forcing[forcPosition5D(i,j,k,2,c)]-tz2*(
								ue[kp1][2]*buf[kp1][3]-ue[km1][2]*buf[km1][3])+
								zzcon2*(buf[kp1][2]-2*buf[k2][2]+buf[km1][2])+
								dz3tz1*( ue[kp1][2]-2* ue[k2][2]+ ue[km1][2]);

						forcing[forcPosition5D(i,j,k,3,c)] = forcing[forcPosition5D(i,j,k,3,c)]-tz2*(
								(ue[kp1][3]*buf[kp1][3] + c2*(ue[kp1][4]-q[kp1]))-
								(ue[km1][3]*buf[km1][3] + c2*(ue[km1][4]-q[km1])))+
								zzcon1*(buf[kp1][3]-2*buf[k2][3]+buf[km1][3])+
								dz4tz1*( ue[kp1][3]-2* ue[k2][3]+ ue[km1][3]);

						forcing[forcPosition5D(i,j,k,4,c)] = forcing[forcPosition5D(i,j,k,4,c)]-tz2*(
								buf[kp1][3]*(c1*ue[kp1][4]-c2*q[kp1])-
								buf[km1][3]*(c1*ue[km1][4]-c2*q[km1]))+
								0.5*zzcon3*(buf[kp1][0]-2*buf[k2][0]+buf[km1][0])+
								zzcon4*(cuf[kp1]-2*cuf[k2]+cuf[km1])+
								zzcon5*(buf[kp1][4]-2*buf[k2][4]+buf[km1][4])+
								dz5tz1*(ue[kp1][4]-2*ue[k2][4]+ue[km1][4]);

					}

					/*
					 * Fourth-order dissipation
					 */
					 if(start[2][c] > 0){
						 for(int m = 0; m<5; m++){
							 int k = 1;
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(5*ue[k+2][m] - 4*ue[k+3][m] + ue[k+4][m]);
							 k = 2;
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(-4*ue[k+1][m] + 6*ue[k+2][m] - 4*ue[k+3][m] + ue[k+4][m]);
						 }
					 }

					 for(int m = 0; m<5; m++){
						 for(int k = start[2][c]*3; k<=cellSize[2][c]-3*end[2][c]-1; k++){
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(ue[k][m] - 4*ue[k+1][m] + 6*ue[k+2][m] - 4*ue[k+3][m] + ue[k+4][m]);
						 }
					 }

					 if(end[2][c] > 0){
						 for(int m = 0; m<5; m++){
							 int k = cellSize[2][c]-3;
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(ue[k][m] - 4*ue[k+1][m] + 6*ue[k+2][m] - 4*ue[k+3][m]);
							 k = cellSize[2][c]-2;
							 forcing[forcPosition5D(i,j,k,m,c)] = forcing[forcPosition5D(i,j,k,m,c)] - dssp*(ue[k][m] - 4*ue[k+1][m] + 5*ue[k+2][m]);
						 }
					 }

				}
			}

			/*
			 * Now change the sign of the forcing function
			 */
			for(int m = 0; m<5; m++){
				for(int k = start[2][c]; k<=cellSize[2][c]-end[2][c]-1; k++){
					for(int j = start[1][c]; j<=cellSize[1][c]-end[1][c]-1; j++){
						for(int i = start[0][c]; i<=cellSize[0][c]-end[0][c]-1; i++){
							forcing[forcPosition5D(i,j,k,m,c)] = -forcing[forcPosition5D(i,j,k,m,c)];
						}
					}
				}
			}

		}
	}

	/*
	 * Compute the actual size of the buffers; note that there is
	 * always one cell face that doesn't need buffer space, because it
	 * is at the boundary of the grid
	 */
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

		txinvr();

		xSolve();

		ySolve();

		zSolve();

		add();

	}

	/*
	 * This method copies the face values of a variable defined on a set
	 * of cells to the overlap locations of the adjacent sets of cells.
	 * Because a set of cells interfaces in each direction with exactly one
	 * other set, we only need to fill six different buffers. We could try
	 * to overlap communication with computation, by computing some internal
	 * values while communicating boundary values, but this adds so much
	 * overhead that it's not clearly useful.
	 */
	private void copyFaces(){

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

		/*
		 * Because the difference stencil for the diagonalized scheme is 
		 * orthogonal, we do not have to perform the staged copying of faces, 
		 * but can send all face information simultaneously to the neighboring 
		 * cells in all directions
		 */
		int p0 = -1;
		int p1 = -1;
		int p2 = -1;
		int p3 = -1;
		int p4 = -1;
		int p5 = -1;

		for(int c = 0; c<ncells; c++){
			for(int m = 0; m<5; m++){

				/* Fill the buffer to be sent to eastern neighbors (i-dir) */
				if(cellCoord[0][c] != ncells){
					for(int k = 2; k<=cellSize[2][c]+1; k++){
						for(int j = 2; j<=cellSize[1][c]+1; j++){
							for(int i = cellSize[0][c]; i<=cellSize[0][c]+1; i++){
								outBuffer[ss[0]+p0] = u[position5D(i,j,k,m,c)];
								p0++;
							}	
						}
					}
				}

				/* Fill the buffer to be sent to western neighbors */
				if(cellCoord[0][c] != 1){
					for(int k = 2; k<=cellSize[2][c]+1; k++){
						for(int j = 2; j<=cellSize[1][c]+1; j++){
							for(int i = 2; i<=3; i++){
								outBuffer[ss[1]+p1] = u[position5D(i,j,k,m,c)];
								p1++;
							}	
						}
					}
				}

				/* Fill the buffer to be sent to northern neighbors (j-dir) */
				if(cellCoord[1][c] != ncells){
					for(int k = 2; k<=cellSize[2][c]+1; k++){
						for(int j = cellSize[1][c]; j<=cellSize[1][c]+1; j++){
							for(int i = 2; i<=cellSize[0][c]+1; i++){
								outBuffer[ss[2]+p2] = u[position5D(i,j,k,m,c)];
								p2++;
							}	
						}
					}
				}

				/* Fill the buffer to be sent to southern neighbors */
				if(cellCoord[1][c] != 1){
					for(int k = 2; k<=cellSize[2][c]+1; k++){
						for(int j = 2; j<=3; j++){
							for(int i = 2; i<=cellSize[0][c]+1; i++){
								outBuffer[ss[3]+p3] = u[position5D(i,j,k,m,c)];
								p3++;
							}	
						}
					}
				}

				/* Fill the buffer to be sent to top neighbors (k-dir) */
				if(cellCoord[2][c] != ncells){
					for(int k = cellSize[2][c]; k<=cellSize[2][c]+1; k++){
						for(int j = 2; j<=cellSize[1][c]+1; j++){
							for(int i = 2; i<=cellSize[0][c]+1; i++){
								outBuffer[ss[4]+p4] = u[position5D(i,j,k,m,c)];
								p4++;
							}	
						}
					}
				}

				/* Fill the buffer to be sent to bottom neighbors */
				if(cellCoord[2][c] != 1){
					for(int k = 2; k<=3; k++){
						for(int j = 2; j<=cellSize[1][c]+1; j++){
							for(int i = 2; i<=cellSize[0][c]+1; i++){
								outBuffer[ss[5]+p5] = u[position5D(i,j,k,m,c)];
								p5++;
							}	
						}
					}
				}
				/* m loop */
			}
			/* cell loop */
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

		/* Unpack the data that has just been received */
		p0 = -1;
		p1 = -1;
		p2 = -1;
		p3 = -1;
		p4 = -1;
		p5 = -1;

		for(int c = 0; c<ncells; c++){
			for(int m = 0; m<5; m++){
				if(cellCoord[0][c] != 1){
					for(int k = 2; k<=cellSize[2][c]+1; k++){
						for(int j = 2; j<=cellSize[1][c]+1; j++){
							for(int i = 0; i<=1; i++){
								u[position5D(i,j,k,m,c)] = inBuffer[sr[1]+p0];
								p0++;
							}	
						}	
					}
				}

				if(cellCoord[0][c] != ncells){
					for(int k = 2; k<=cellSize[2][c]+1; k++){
						for(int j = 2; j<=cellSize[1][c]+1; j++){
							for(int i = cellSize[0][c]+2; i<=cellSize[0][c]+3; i++){
								u[position5D(i,j,k,m,c)] = inBuffer[sr[0]+p1];
								p1++;
							}	
						}	
					}
				}

				if(cellCoord[1][c] != 1){
					for(int k = 2; k<=cellSize[2][c]+1; k++){
						for(int j = 0; j<=1; j++){
							for(int i = 2; i<=cellSize[0][c]+1; i++){
								u[position5D(i,j,k,m,c)] = inBuffer[sr[3]+p2];
								p2++;
							}	
						}	
					}
				}

				if(cellCoord[1][c] != ncells){
					for(int k = 2; k<=cellSize[2][c]+1; k++){
						for(int j = cellSize[1][c]+2; j<=cellSize[1][c]+3; j++){
							for(int i = 2; i<=cellSize[0][c]+1; i++){
								u[position5D(i,j,k,m,c)] = inBuffer[sr[2]+p3];
								p3++;
							}	
						}	
					}
				}

				if(cellCoord[2][c] != 1){
					for(int k = 0; k<=1; k++){
						for(int j = 2; j<=cellSize[1][c]+1; j++){
							for(int i = 2; i<=cellSize[0][c]+1; i++){
								u[position5D(i,j,k,m,c)] = inBuffer[sr[5]+p4];
								p4++;
							}	
						}	
					}
				}

				if(cellCoord[2][c] != ncells){
					for(int k = cellSize[2][c]+2; k<=cellSize[2][c]+3; k++){
						for(int j = 2; j<=cellSize[1][c]+1; j++){
							for(int i = 2; i<=cellSize[0][c]+1; i++){
								u[position5D(i,j,k,m,c)] = inBuffer[sr[4]+p5];
								p5++;
							}	
						}	
					}
				}
				/* m loop */
			}
			/* cells loop */
		}

		/* Now that we have all the data, compute the rhs */
		computeRHS();

	}

	/*
	 * Block-diagonal matrix-vector multiplication
	 */
	private void txinvr(){
		/* Loop over all cells owned by this node */
		for(int c = 0; c<ncells; c++){
			for(int k = start[2][c]+1; k<=cellSize[2][c]-end[2][c]; k++){
				int k1 = k+1;
				for(int j = start[1][c]+1; j<=cellSize[1][c]-end[1][c]; j++){
					int j1 = j+1;
					for(int i = start[0][c]+1; i<=cellSize[0][c]-end[0][c]; i++){
						int i1 = i+1;
						double ru1 = rhoI[position4D(i,j,k,c)];
						double uu = us[position4D(i,j,k,c)];
						double vv = vs[position4D(i,j,k,c)];
						double ww = ws[position4D(i,j,k,c)];
						double ac = speed[position4D(i,j,k,c)];
						double ac2inv = ainv[position4D(i,j,k,c)]*ainv[position4D(i,j,k,c)];

						double r1 = rhs[position5D(i1,j1,k1,0,c)];
						double r2 = rhs[position5D(i1,j1,k1,1,c)];
						double r3 = rhs[position5D(i1,j1,k1,2,c)];
						double r4 = rhs[position5D(i1,j1,k1,3,c)];
						double r5 = rhs[position5D(i1,j1,k1,4,c)];

						double t1 = c2 * ac2inv * (qs[position4D(i,j,k,c)]*r1 -uu*r2 - vv*r3 - ww*r4 + r5);
						double t2 = bt * ru1 * (uu*r1-r2);
						double t3 = (bt*ru1*ac) * t1;

						rhs[position5D(i1,j1,k1,0,c)] = r1 - t1;
						rhs[position5D(i1,j1,k1,1,c)] = -ru1 * (ww*r1 - r4);
						rhs[position5D(i1,j1,k1,2,c)] =  ru1 * (vv*r1 - r3);
						rhs[position5D(i1,j1,k1,3,c)] = -t2 + t3;
						rhs[position5D(i1,j1,k1,4,c)] =  t2 + t3;

					}
				}
			}
		}

	}

	private void xThomasForward(int c, int istart, int iend, int jsize, int ksize){
		/* The recurrence runs along the inner loop: the rows i (d, c4, c5 and q of rhs) and i+1 (b2, b3, b4 and s of
		 * rhs) travel in locals instead of being read back from lhs and rhs right after they are written (those
		 * loads wait for the stores: the Thomas recurrence would go through memory, and lhs and rhs have the same
		 * layout and alignment). Every store of the original is kept, so lhs and rhs end with the same contents */
		final int S = DIMI_DIMJ_DIMK;
		final int n = -1;
		final int o1 = (n+1)*S, o2 = (n+2)*S, o3 = (n+3)*S, o4 = (n+4)*S, o5 = (n+5)*S;
		for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
			for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
				if(istart+2 > iend) continue;
				int l = lhsPosition5D(istart+2,j,k,0,c);
				int r = position5D(istart+2,j,k,0,c);
				double d = lhs[l+o3], c4 = lhs[l+o4], c5 = lhs[l+o5];
				double b2 = lhs[l+1+o2], b3 = lhs[l+1+o3], b4 = lhs[l+1+o4];
				double q0 = rhs[r+0], s0 = rhs[r+1+0];
				double q1 = rhs[r+1*S], s1 = rhs[r+1+1*S];
				double q2 = rhs[r+2*S], s2 = rhs[r+1+2*S];
				for(int i = istart+2; i<=iend; i++){
					final double fac1 = 1.0d/d;
					c4 = fac1*c4;
					lhs[l+o4] = c4;
					c5 = fac1*c5;
					lhs[l+o5] = c5;
					q0 = fac1*q0;
					rhs[r+0] = q0;
					q1 = fac1*q1;
					rhs[r+1*S] = q1;
					q2 = fac1*q2;
					rhs[r+2*S] = q2;
					b3 = b3 - b2*c4;
					lhs[l+1+o3] = b3;
					b4 = b4 - b2*c5;
					lhs[l+1+o4] = b4;
					s0 = s0 - b2*q0;
					rhs[r+1+0] = s0;
					s1 = s1 - b2*q1;
					rhs[r+1+1*S] = s1;
					s2 = s2 - b2*q2;
					rhs[r+1+2*S] = s2;
					final double e1 = lhs[l+2+o1];
					final double e2 = lhs[l+2+o2] - e1*c4;
					lhs[l+2+o2] = e2;
					final double e3 = lhs[l+2+o3] - e1*c5;
					lhs[l+2+o3] = e3;
					final double t0 = rhs[r+2+0] - e1*q0;
					rhs[r+2+0] = t0;
					final double t1 = rhs[r+2+1*S] - e1*q1;
					rhs[r+2+1*S] = t1;
					final double t2 = rhs[r+2+2*S] - e1*q2;
					rhs[r+2+2*S] = t2;
					/* the row i+1 becomes the row i, and i+2 becomes i+1 */
					d = b3; c4 = b4; c5 = lhs[l+1+o5];
					b2 = e2; b3 = e3; b4 = lhs[l+2+o4];
					q0 = s0; s0 = t0;
					q1 = s1; s1 = t1;
					q2 = s2; s2 = t2;
					l++;
					r++;
				}
			}
		}
	}

	private void xThomasForwardLast2(int c, int istart, int iend, int jsize, int ksize){
		int n = -1;
		int i = iend+1;
		int i1 = iend+2;
		for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
			for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
				double fac1 = 1.0d/lhs[lhsPosition5D(i,j,k,n+3,c)];
				lhs[lhsPosition5D(i,j,k,n+4,c)] = fac1*lhs[lhsPosition5D(i,j,k,n+4,c)];
				lhs[lhsPosition5D(i,j,k,n+5,c)] = fac1*lhs[lhsPosition5D(i,j,k,n+5,c)];
				for(int m = 0; m<3; m++){
					rhs[position5D(i,j,k,m,c)] = fac1*rhs[position5D(i,j,k,m,c)];
				}
				lhs[lhsPosition5D(i1,j,k,n+3,c)] = lhs[lhsPosition5D(i1,j,k,n+3,c)] - lhs[lhsPosition5D(i1,j,k,n+2,c)]*lhs[lhsPosition5D(i,j,k,n+4,c)];
				lhs[lhsPosition5D(i1,j,k,n+4,c)] = lhs[lhsPosition5D(i1,j,k,n+4,c)] - lhs[lhsPosition5D(i1,j,k,n+2,c)]*lhs[lhsPosition5D(i,j,k,n+5,c)];
				for(int m = 0; m<3; m++){
					rhs[position5D(i1,j,k,m,c)] = rhs[position5D(i1,j,k,m,c)] - lhs[lhsPosition5D(i1,j,k,n+2,c)]*rhs[position5D(i,j,k,m,c)];
				}
				/*
				 * Scale the last row immediately (some of this is
				 * overkill in case this is the last cell)
				 */
				double fac2 = 1.0d/lhs[lhsPosition5D(i1,j,k,n+3,c)];
				lhs[lhsPosition5D(i1,j,k,n+4,c)] = fac2*lhs[lhsPosition5D(i1,j,k,n+4,c)];
				lhs[lhsPosition5D(i1,j,k,n+5,c)] = fac2*lhs[lhsPosition5D(i1,j,k,n+5,c)];
				for(int m = 0; m<3; m++){
					rhs[position5D(i1,j,k,m,c)] = fac2*rhs[position5D(i1,j,k,m,c)];
				}

			}
		}
	}

	private void xUCFactors(int m, int c, int istart, int iend, int jsize, int ksize){
		/* The recurrence runs along the inner loop: the rows i (d, c4, c5 and q of rhs) and i+1 (b2, b3, b4 and s of
		 * rhs) travel in locals instead of being read back from lhs and rhs right after they are written (those
		 * loads wait for the stores: the Thomas recurrence would go through memory, and lhs and rhs have the same
		 * layout and alignment). Every store of the original is kept, so lhs and rhs end with the same contents */
		final int S = DIMI_DIMJ_DIMK;
		final int n = (m-2)*5-1;
		final int o1 = (n+1)*S, o2 = (n+2)*S, o3 = (n+3)*S, o4 = (n+4)*S, o5 = (n+5)*S;
		final int om = m*S;
		for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
			for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
				if(istart+2 > iend) continue;
				int l = lhsPosition5D(istart+2,j,k,0,c);
				int r = position5D(istart+2,j,k,0,c);
				double d = lhs[l+o3], c4 = lhs[l+o4], c5 = lhs[l+o5];
				double b2 = lhs[l+1+o2], b3 = lhs[l+1+o3], b4 = lhs[l+1+o4];
				double qm = rhs[r+om], sm = rhs[r+1+om];
				for(int i = istart+2; i<=iend; i++){
					final double fac1 = 1.0d/d;
					c4 = fac1*c4;
					lhs[l+o4] = c4;
					c5 = fac1*c5;
					lhs[l+o5] = c5;
					qm = fac1*qm;
					rhs[r+om] = qm;
					b3 = b3 - b2*c4;
					lhs[l+1+o3] = b3;
					b4 = b4 - b2*c5;
					lhs[l+1+o4] = b4;
					sm = sm - b2*qm;
					rhs[r+1+om] = sm;
					final double e1 = lhs[l+2+o1];
					final double e2 = lhs[l+2+o2] - e1*c4;
					lhs[l+2+o2] = e2;
					final double e3 = lhs[l+2+o3] - e1*c5;
					lhs[l+2+o3] = e3;
					final double tm = rhs[r+2+om] - e1*qm;
					rhs[r+2+om] = tm;
					/* the row i+1 becomes the row i, and i+2 becomes i+1 */
					d = b3; c4 = b4; c5 = lhs[l+1+o5];
					b2 = e2; b3 = e3; b4 = lhs[l+2+o4];
					qm = sm; sm = tm;
					l++;
					r++;
				}
			}
		}
	}

	private void xUCFactorsLast2(int m, int c, int istart, int iend, int jsize, int ksize){
		int n = (m-2)*5-1;
		int i = iend+1;
		int i1 = iend+2;
		for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
			for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
				double fac1 = 1.0d/lhs[lhsPosition5D(i,j,k,n+3,c)];
				lhs[lhsPosition5D(i,j,k,n+4,c)] = fac1*lhs[lhsPosition5D(i,j,k,n+4,c)];
				lhs[lhsPosition5D(i,j,k,n+5,c)] = fac1*lhs[lhsPosition5D(i,j,k,n+5,c)];
				rhs[position5D(i,j,k,m,c)] = fac1*rhs[position5D(i,j,k,m,c)];
				lhs[lhsPosition5D(i1,j,k,n+3,c)] = lhs[lhsPosition5D(i1,j,k,n+3,c)] - lhs[lhsPosition5D(i1,j,k,n+2,c)]*lhs[lhsPosition5D(i,j,k,n+4,c)];
				lhs[lhsPosition5D(i1,j,k,n+4,c)] = lhs[lhsPosition5D(i1,j,k,n+4,c)] - lhs[lhsPosition5D(i1,j,k,n+2,c)]*lhs[lhsPosition5D(i,j,k,n+5,c)];
				rhs[position5D(i1,j,k,m,c)] = rhs[position5D(i1,j,k,m,c)] - lhs[lhsPosition5D(i1,j,k,n+2,c)]*rhs[position5D(i,j,k,m,c)];
				/*
				 * Scale the last row immediately (some o this is
				 * overkill in case this is the last cell)
				 */
				double fac2 = 1.0d/lhs[lhsPosition5D(i1,j,k,n+3,c)];
				lhs[lhsPosition5D(i1,j,k,n+4,c)] = fac2*lhs[lhsPosition5D(i1,j,k,n+4,c)];
				lhs[lhsPosition5D(i1,j,k,n+5,c)] = fac2*lhs[lhsPosition5D(i1,j,k,n+5,c)];
				rhs[position5D(i1,j,k,m,c)] = fac2*rhs[position5D(i1,j,k,m,c)];
			}
		}
	}

	/*
	 * This method performs the solution of the approximate factorization
	 * step in the x-direction for all five matrix components
	 * simultaneously. The Thomas algorithm is employed to solve the
	 * systems for the x-lines. Boundary conditions are non-periodic
	 */
	private void xSolve(){
		Request requests[] = new Request[2];
		double s[] = new double[5];
		double d, e, r1, r2;
		int n;

		/*
		 * Now do a sweep on a layer-by-layer basis, i.e. sweeping through cells
		 * on this node in the direction of increasing i for the forward sweep,
		 * and afer that reversing the direction for the backsubstitution.
		 */

		/* Forward elimination */
		for(int stage = 0; stage<ncells; stage++){
			int c = slice[0][stage];

			int istart = 0;
			int iend = cellSize[0][c]-1;

			int jsize = cellSize[1][c];
			int ksize = cellSize[2][c];

			int bufferSize = (jsize-start[1][c]-end[1][c])*(ksize-start[2][c]-end[2][c]);

			if(stage != 0){

				/*
				 * If this is not the first processor in this row of cells,
				 * receive data from predecessor containing the right hand
				 * sides and the upper diagonal elements of the previous two rows
				 */
				if(detailedTiming)
					timer.start(T_COMM);
				requests[0] = commSolve.Irecv(inBuffer, 0, 22*bufferSize, MPI.DOUBLE, predecessor[0], DEFAULT_TAG);
				if(detailedTiming)
					timer.stop(T_COMM);

				/*
				 * Communication has already been started.
				 * Compute the left hand side while waiting for the msg
				 */
				lhsx(c);

				/*
				 * Wait for pending communication to complete.
				 * This waits on the current receive and on the send
				 * from the previous stage. They always come in pairs.
				 */
				if(detailedTiming)
					timer.start(T_COMM);
				Request.Waitall(requests);
				if(detailedTiming)
					timer.stop(T_COMM);

				/* Unpack the buffer */
				int i = istart+2;
				int i1 = i+1;
				n = -1;

				/* Create a running pointer */
				int p = -1;
				for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
					for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
						lhs[lhsPosition5D(i,j,k,n+2,c)] = lhs[lhsPosition5D(i,j,k,n+2,c)] - inBuffer[p+1]*lhs[lhsPosition5D(i,j,k,n+1,c)];
						lhs[lhsPosition5D(i,j,k,n+3,c)] = lhs[lhsPosition5D(i,j,k,n+3,c)] - inBuffer[p+2]*lhs[lhsPosition5D(i,j,k,n+1,c)];
						for(int m = 0; m<3; m++){
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - inBuffer[p+3+m]*lhs[lhsPosition5D(i,j,k,n+1,c)];
						}
						d = inBuffer[p+6];
						e = inBuffer[p+7];
						for(int m = 0; m<3; m++){
							s[m] = inBuffer[p+8+m];
						}

						r1 = lhs[lhsPosition5D(i,j,k,n+2,c)];
						lhs[lhsPosition5D(i,j,k,n+3,c)] = lhs[lhsPosition5D(i,j,k,n+3,c)] - d*r1;
						lhs[lhsPosition5D(i,j,k,n+4,c)] = lhs[lhsPosition5D(i,j,k,n+4,c)] - e*r1;
						for(int m = 0; m<3; m++){
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - s[m]*r1;
						}
						r2 = lhs[lhsPosition5D(i1,j,k,n+1,c)];
						lhs[lhsPosition5D(i1,j,k,n+2,c)] = lhs[lhsPosition5D(i1,j,k,n+2,c)] - d*r2;
						lhs[lhsPosition5D(i1,j,k,n+3,c)] = lhs[lhsPosition5D(i1,j,k,n+3,c)] - e*r2;
						for(int m = 0; m<3; m++){
							rhs[position5D(i1,j,k,m,c)] = rhs[position5D(i1,j,k,m,c)] - s[m]*r2;
						}
						p += 10;
					}
				}

				for(int m = 3; m<5; m++){
					n = (m-2)*5-1;
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
							lhs[lhsPosition5D(i,j,k,n+2,c)] = lhs[lhsPosition5D(i,j,k,n+2,c)] - inBuffer[p+1]*lhs[lhsPosition5D(i,j,k,n+1,c)];
							lhs[lhsPosition5D(i,j,k,n+3,c)] = lhs[lhsPosition5D(i,j,k,n+3,c)] - inBuffer[p+2]*lhs[lhsPosition5D(i,j,k,n+1,c)];
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - inBuffer[p+3]*lhs[lhsPosition5D(i,j,k,n+1,c)];

							d = inBuffer[p+4];
							e = inBuffer[p+5];
							s[m] = inBuffer[p+6];

							r1 = lhs[lhsPosition5D(i,j,k,n+2,c)];
							lhs[lhsPosition5D(i,j,k,n+3,c)] = lhs[lhsPosition5D(i,j,k,n+3,c)] - d*r1;
							lhs[lhsPosition5D(i,j,k,n+4,c)] = lhs[lhsPosition5D(i,j,k,n+4,c)] - e*r1;
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - s[m]*r1;

							r2 = lhs[lhsPosition5D(i1,j,k,n+1,c)];
							lhs[lhsPosition5D(i1,j,k,n+2,c)] = lhs[lhsPosition5D(i1,j,k,n+2,c)] - d*r2;
							lhs[lhsPosition5D(i1,j,k,n+3,c)] = lhs[lhsPosition5D(i1,j,k,n+3,c)] - e*r2;
							rhs[position5D(i1,j,k,m,c)] = rhs[position5D(i1,j,k,m,c)] - s[m]*r2;

							p += 6;
						}
					}

				}

			}
			else{
				/* If this is the first cell, we still compute the lhs */
				lhsx(c);
			}

			/* Perform the Thomas algorithm; first, FORWARD ELIMINATION */
			xThomasForward(c, istart, iend, jsize, ksize);

			/*
			 * The last two rows in this grid block are a bit different,
			 * since they do not have two more rows available for the
			 * elimination of off-diagonal entries
			 */
			xThomasForwardLast2(c, istart, iend, jsize, ksize);

			/* Do the u+c and the u-c factors */
			for(int m = 3; m<5; m++){
				xUCFactors(m, c, istart, iend, jsize, ksize);

				/* And again the last two rows separately */
				xUCFactorsLast2(m, c, istart, iend, jsize, ksize);
			}

			/*
			 * Send information to the next processor, except when this
			 * is the last grid block
			 */
			int i;
			if(stage != ncells-1){

				/* Create a running pointer for the send buffer */
				int p = -1;
				n = -1;
				for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
					for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
						for(i = iend+1; i<=iend+2; i++){
							outBuffer[p+1] = lhs[lhsPosition5D(i,j,k,n+4,c)];
							outBuffer[p+2] = lhs[lhsPosition5D(i,j,k,n+5,c)]; 
							for(int m = 0; m<3; m++){
								outBuffer[p+3+m] = rhs[position5D(i,j,k,m,c)];
							}
							p += 5;
						}
					}
				}

				for(int m = 3; m<5; m++){
					n = (m-2)*5-1;
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
							for(i = iend+1; i<=iend+2; i++){
								outBuffer[p+1] = lhs[lhsPosition5D(i,j,k,n+4,c)];
								outBuffer[p+2] = lhs[lhsPosition5D(i,j,k,n+5,c)];
								outBuffer[p+3] = rhs[position5D(i,j,k,m,c)];
								p += 3;
							}
						}
					}
				}

				/*
				 * Send data to next phase
				 * Can't receive data yet because buffer size will be wrong
				 */
				if(detailedTiming)
					timer.start(T_COMM);
				requests[1] = commSolve.Isend(outBuffer, 0, 22*bufferSize, MPI.DOUBLE, successor[0], DEFAULT_TAG);
				if(detailedTiming)
					timer.stop(T_COMM);

			}

		}

		/* Now go in the reverse direction */

		/* BACKSUBSTITUTION */
		for(int stage = ncells-1; stage>=0; stage--){
			int c = slice[0][stage];

			int istart = 0;
			int iend = cellSize[0][c]-1;

			int jsize = cellSize[1][c];
			int ksize = cellSize[2][c];

			int bufferSize = (jsize-start[1][c]-end[1][c])*(ksize-start[2][c]-end[2][c]);

			if(stage != ncells-1){
				/*
				 * If this is not the starting cell in this row of cells,
				 * wait for a message to be received, containing the
				 * solution of the previous two stations
				 */
				if(detailedTiming)
					timer.start(T_COMM);
				requests[0] = commSolve.Irecv(inBuffer, 0, 10*bufferSize, MPI.DOUBLE, successor[0], DEFAULT_TAG);
				if(detailedTiming)
					timer.stop(T_COMM);

				/*
				 * Communication has already been started
				 * while waiting, do the block-diagonal inversion for the
				 * cell that was just finished
				 */
				ninvr(slice[0][stage+1]);

				/* Wait for pending communication to complete */
				if(detailedTiming)
					timer.start(T_COMM);
				Request.Waitall(requests);
				if(detailedTiming)
					timer.stop(T_COMM);

				/* Unpack the buffer for the first three factors */
				n = -1;
				int p = -1;
				int i = iend+2;
				int i1 = i-1;
				for(int m = 0; m<3; m++){
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
							double sm1 = inBuffer[p+1];
							double sm2 = inBuffer[p+2];
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*sm1-lhs[lhsPosition5D(i,j,k,n+5,c)]*sm2;
							rhs[position5D(i1,j,k,m,c)] = rhs[position5D(i1,j,k,m,c)] - lhs[lhsPosition5D(i1,j,k,n+4,c)]*rhs[position5D(i,j,k,m,c)]-lhs[lhsPosition5D(i1,j,k,n+5,c)]*sm1;
							p += 2;
						}
					}
				}

				/* Now unpack the buffer for the remaining two factors */
				for(int m = 3; m<5; m++){
					n = (m-2)*5-1;
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
							double sm1 = inBuffer[p+1];
							double sm2 = inBuffer[p+2];
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*sm1-lhs[lhsPosition5D(i,j,k,n+5,c)]*sm2;
							rhs[position5D(i1,j,k,m,c)] = rhs[position5D(i1,j,k,m,c)] - lhs[lhsPosition5D(i1,j,k,n+4,c)]*rhs[position5D(i,j,k,m,c)]-lhs[lhsPosition5D(i1,j,k,n+5,c)]*sm1;
							p += 2;
						}
					}
				}

			}
			else{
				/*
				 * Now we know this is the first grid block on the back sweep,
				 * so we don't need a message to start the substitution
				 */
				int i = iend+1;
				int i1 = iend+2;
				n = -1;
				for(int m = 0; m<3; m++){
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*rhs[position5D(i1,j,k,m,c)];
						}
					}
				}

				for(int m = 3; m<5; m++){
					n = (m-2)*5-1;
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*rhs[position5D(i1,j,k,m,c)];
						}
					}
				}

			}

			/*
			 * Whether or not this is the last processor, we always have
			 * to complete the back-substitution
			 */

			/* The first three factors */
			n = -1;
			for(int m = 0; m<3; m++){
				for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
					for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
						/* rhs(i+1) and rhs(i+2) travel in locals (x1, x2) instead of being read back right after the store */
						if(istart+2 <= iend){
							int r = position5D(iend,j,k,m,c);
							int l = lhsPosition5D(iend,j,k,n+4,c);
							double x1 = rhs[r+1], x2 = rhs[r+2];
							for(int i = iend; i>=istart+2; i--){
								final double x0 = rhs[r] - lhs[l]*x1-lhs[l+DIMI_DIMJ_DIMK]*x2;
								rhs[r] = x0;
								x2 = x1; x1 = x0;
								r--; l--;
							}
						}
					}
				}
			}

			/* And the remaining two */

			for(int m = 3; m<5; m++){
				n = (m-2)*5-1;
				for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
					for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
						/* rhs(i+1) and rhs(i+2) travel in locals (x1, x2) instead of being read back right after the store */
						if(istart+2 <= iend){
							int r = position5D(iend,j,k,m,c);
							int l = lhsPosition5D(iend,j,k,n+4,c);
							double x1 = rhs[r+1], x2 = rhs[r+2];
							for(int i = iend; i>=istart+2; i--){
								final double x0 = rhs[r] - lhs[l]*x1-lhs[l+DIMI_DIMJ_DIMK]*x2;
								rhs[r] = x0;
								x2 = x1; x1 = x0;
								r--; l--;
							}
						}
					}
				}
			}

			/* Send on information to the previous processor, if needed */
			if(stage != 0){
				int i = istart+2;
				int i1 = istart+3;
				int p = -1;
				for(int m = 0; m<5; m++){
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
							outBuffer[p+1] = rhs[position5D(i,j,k,m,c)];
							outBuffer[p+2] = rhs[position5D(i1,j,k,m,c)];
							p += 2;
						}
					}
				}

				/* Pack and send the buffer */
				if(detailedTiming)
					timer.start(T_COMM);
				requests[1] = commSolve.Isend(outBuffer, 0, 10*bufferSize, MPI.DOUBLE, predecessor[0], DEFAULT_TAG);
				if(detailedTiming)
					timer.stop(T_COMM);

			}

			/* If this was the last stage, do the block-diagonal inversion */
			if(stage == 0)
				ninvr(c);

		}

	}

	private void yThomasForward(int c, int jstart, int jend, int isize, int ksize){
		int n = -1;
		final int S = DIMI_DIMJ_DIMK, D = DIMI;
		final int o1 = (n+1)*S, o2 = (n+2)*S, o3 = (n+3)*S, o4 = (n+4)*S, o5 = (n+5)*S;
		for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
			for(int j = jstart+2; j<=jend; j++){
				for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
					/* every operand is read once into a local; the stores are those of the original */
					final int l = lhsPosition5D(i,j,k,0,c);
					final int r = position5D(i,j,k,0,c);
					final double fac1 = 1.0d/lhs[l+o3];
					final double c4 = fac1*lhs[l+o4];
					lhs[l+o4] = c4;
					final double c5 = fac1*lhs[l+o5];
					lhs[l+o5] = c5;
					final double q0 = fac1*rhs[r+0];
					rhs[r+0] = q0;
					final double q1 = fac1*rhs[r+S];
					rhs[r+S] = q1;
					final double q2 = fac1*rhs[r+2*S];
					rhs[r+2*S] = q2;
					final double b2 = lhs[l+D+o2];
					lhs[l+D+o3] = lhs[l+D+o3] - b2*c4;
					lhs[l+D+o4] = lhs[l+D+o4] - b2*c5;
					rhs[r+D+0] = rhs[r+D+0] - b2*q0;
					rhs[r+D+S] = rhs[r+D+S] - b2*q1;
					rhs[r+D+2*S] = rhs[r+D+2*S] - b2*q2;
					final double e1 = lhs[l+2*D+o1];
					lhs[l+2*D+o2] = lhs[l+2*D+o2] - e1*c4;
					lhs[l+2*D+o3] = lhs[l+2*D+o3] - e1*c5;
					rhs[r+2*D+0] = rhs[r+2*D+0] - e1*q0;
					rhs[r+2*D+S] = rhs[r+2*D+S] - e1*q1;
					rhs[r+2*D+2*S] = rhs[r+2*D+2*S] - e1*q2;
				}
			}
		}
	}

	private void yThomasForwardLast2(int c, int jstart, int jend, int isize, int ksize){
		int n=-1;
		int j = jend+1;
		int j1 = jend+2;
		for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
			for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
				double fac1 = 1.0d/lhs[lhsPosition5D(i,j,k,n+3,c)];
				lhs[lhsPosition5D(i,j,k,n+4,c)] = fac1*lhs[lhsPosition5D(i,j,k,n+4,c)];
				lhs[lhsPosition5D(i,j,k,n+5,c)] = fac1*lhs[lhsPosition5D(i,j,k,n+5,c)];
				for(int m = 0; m<3; m++){
					rhs[position5D(i,j,k,m,c)] = fac1*rhs[position5D(i,j,k,m,c)];
				}
				lhs[lhsPosition5D(i,j1,k,n+3,c)] = lhs[lhsPosition5D(i,j1,k,n+3,c)] - lhs[lhsPosition5D(i,j1,k,n+2,c)]*lhs[lhsPosition5D(i,j,k,n+4,c)];
				lhs[lhsPosition5D(i,j1,k,n+4,c)] = lhs[lhsPosition5D(i,j1,k,n+4,c)] - lhs[lhsPosition5D(i,j1,k,n+2,c)]*lhs[lhsPosition5D(i,j,k,n+5,c)];
				for(int m = 0; m<3; m++){
					rhs[position5D(i,j1,k,m,c)] = rhs[position5D(i,j1,k,m,c)] - lhs[lhsPosition5D(i,j1,k,n+2,c)]*rhs[position5D(i,j,k,m,c)];
				}
				/*
				 * Scale the last row immediately (some o this is
				 * overkill in case this is the last cell)
				 */
				double fac2 = 1.0d/lhs[lhsPosition5D(i,j1,k,n+3,c)];
				lhs[lhsPosition5D(i,j1,k,n+4,c)] = fac2*lhs[lhsPosition5D(i,j1,k,n+4,c)];
				lhs[lhsPosition5D(i,j1,k,n+5,c)] = fac2*lhs[lhsPosition5D(i,j1,k,n+5,c)];
				for(int m = 0; m<3; m++){
					rhs[position5D(i,j1,k,m,c)] = fac2*rhs[position5D(i,j1,k,m,c)];
				}
			}
		}
	}

	private void yUCFactors(int m, int c, int jstart, int jend, int isize, int ksize){
		int j, j1;
		int n = (m-2)*5-1;
		final int S = DIMI_DIMJ_DIMK, D = DIMI;
		final int o1 = (n+1)*S, o2 = (n+2)*S, o3 = (n+3)*S, o4 = (n+4)*S, o5 = (n+5)*S;
		final int om = m*S;
		for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
			for(j = jstart+2; j<=jend; j++){
				for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
					/* every operand is read once into a local; the stores are those of the original */
					final int l = lhsPosition5D(i,j,k,0,c);
					final int r = position5D(i,j,k,0,c);
					final double fac1 = 1.0d/lhs[l+o3];
					final double c4 = fac1*lhs[l+o4];
					lhs[l+o4] = c4;
					final double c5 = fac1*lhs[l+o5];
					lhs[l+o5] = c5;
					final double q0 = fac1*rhs[r+om];
					rhs[r+om] = q0;
					final double b2 = lhs[l+D+o2];
					lhs[l+D+o3] = lhs[l+D+o3] - b2*c4;
					lhs[l+D+o4] = lhs[l+D+o4] - b2*c5;
					rhs[r+D+om] = rhs[r+D+om] - b2*q0;
					final double e1 = lhs[l+2*D+o1];
					lhs[l+2*D+o2] = lhs[l+2*D+o2] - e1*c4;
					lhs[l+2*D+o3] = lhs[l+2*D+o3] - e1*c5;
					rhs[r+2*D+om] = rhs[r+2*D+om] - e1*q0;
				}
			}
		}
	}

	private void yUCFactorsLast2(int m, int c, int jstart, int jend, int isize, int ksize){
		int j = jend+1;
		int j1 = jend+2;
		int n = (m-2)*5-1;
		for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
			for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
				double fac1 = 1.0d/lhs[lhsPosition5D(i,j,k,n+3,c)];
				lhs[lhsPosition5D(i,j,k,n+4,c)] = fac1*lhs[lhsPosition5D(i,j,k,n+4,c)];
				lhs[lhsPosition5D(i,j,k,n+5,c)] = fac1*lhs[lhsPosition5D(i,j,k,n+5,c)];
				rhs[position5D(i,j,k,m,c)] = fac1*rhs[position5D(i,j,k,m,c)];
				lhs[lhsPosition5D(i,j1,k,n+3,c)] = lhs[lhsPosition5D(i,j1,k,n+3,c)] - lhs[lhsPosition5D(i,j1,k,n+2,c)]*lhs[lhsPosition5D(i,j,k,n+4,c)];
				lhs[lhsPosition5D(i,j1,k,n+4,c)] = lhs[lhsPosition5D(i,j1,k,n+4,c)] - lhs[lhsPosition5D(i,j1,k,n+2,c)]*lhs[lhsPosition5D(i,j,k,n+5,c)];
				rhs[position5D(i,j1,k,m,c)] = rhs[position5D(i,j1,k,m,c)] - lhs[lhsPosition5D(i,j1,k,n+2,c)]*rhs[position5D(i,j,k,m,c)];
				/*
				 * Scale the last row immediately (some o this is
				 * overkill in case this is the last cell)
				 */
				double fac2 = 1.0d/lhs[lhsPosition5D(i,j1,k,n+3,c)];
				lhs[lhsPosition5D(i,j1,k,n+4,c)] = fac2*lhs[lhsPosition5D(i,j1,k,n+4,c)];
				lhs[lhsPosition5D(i,j1,k,n+5,c)] = fac2*lhs[lhsPosition5D(i,j1,k,n+5,c)];
				rhs[position5D(i,j1,k,m,c)] = fac2*rhs[position5D(i,j1,k,m,c)];
			}
		}
	}

	/*
	 * This method performs the solution of the approximate factorization
	 * step in the y-direction for all five matrix components
	 * simultaneously. The Thomas algorithm is employed to solve the
	 * systems for the y-lines. Boundary conditions are non-periodic
	 */
	private void ySolve(){
		Request requests[] = new Request[2];
		double s[] = new double[5];
		double d, e, r1, r2;
		int n;

		/*
		 * Now do a sweep on a layer-by-layer basis, i.e. sweeping through cells
		 * on this node in the direction of increasing i for the forward sweep,
		 * and afer that reversing the direction for the backsubstitution.
		 */

		/* Forward elimination */
		for(int stage = 0; stage<ncells; stage++){
			int c = slice[1][stage];

			int jstart = 0;
			int jend = cellSize[1][c]-1;

			int isize = cellSize[0][c];
			int ksize = cellSize[2][c];

			int bufferSize = (isize-start[0][c]-end[0][c])*(ksize-start[2][c]-end[2][c]);

			if(stage != 0){

				/*
				 * If this is not the first processor in this row of cells,
				 * receive data from predecessor containing the right hand
				 * sides and the upper diagonal elements of the previous two rows
				 */
				if(detailedTiming)
					timer.start(T_COMM);
				requests[0] = commSolve.Irecv(inBuffer, 0, 22*bufferSize, MPI.DOUBLE, predecessor[1], DEFAULT_TAG);
				if(detailedTiming)
					timer.stop(T_COMM);

				/*
				 * Communication has already been started.
				 * Compute the left hand side while waiting for the msg
				 */
				lhsy(c);

				/*
				 * Wait for pending communication to complete.
				 * This waits on the current receive and on the send
				 * from the previous stage. They always come in pairs.
				 */
				if(detailedTiming)
					timer.start(T_COMM);
				Request.Waitall(requests);
				if(detailedTiming)
					timer.stop(T_COMM);

				/* Unpack the buffer */
				int j = jstart+2;
				int j1 = j+1;
				n = -1;

				/* Create a running pointer */
				int p = -1;
				for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
					for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
						lhs[lhsPosition5D(i,j,k,n+2,c)] = lhs[lhsPosition5D(i,j,k,n+2,c)] - inBuffer[p+1]*lhs[lhsPosition5D(i,j,k,n+1,c)];
						lhs[lhsPosition5D(i,j,k,n+3,c)] = lhs[lhsPosition5D(i,j,k,n+3,c)] - inBuffer[p+2]*lhs[lhsPosition5D(i,j,k,n+1,c)];
						for(int m = 0; m<3; m++){
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - inBuffer[p+3+m]*lhs[lhsPosition5D(i,j,k,n+1,c)];
						}
						d = inBuffer[p+6];
						e = inBuffer[p+7];

						for(int m = 0; m<3; m++){
							s[m] = inBuffer[p+8+m];
						}
						r1 = lhs[lhsPosition5D(i,j,k,n+2,c)];
						lhs[lhsPosition5D(i,j,k,n+3,c)] = lhs[lhsPosition5D(i,j,k,n+3,c)] - d*r1;
						lhs[lhsPosition5D(i,j,k,n+4,c)] = lhs[lhsPosition5D(i,j,k,n+4,c)] - e*r1;
						for(int m = 0; m<3; m++){
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - s[m]*r1;
						}
						r2 = lhs[lhsPosition5D(i,j1,k,n+1,c)];
						lhs[lhsPosition5D(i,j1,k,n+2,c)] = lhs[lhsPosition5D(i,j1,k,n+2,c)] - d*r2;
						lhs[lhsPosition5D(i,j1,k,n+3,c)] = lhs[lhsPosition5D(i,j1,k,n+3,c)] - e*r2;
						for(int m = 0; m<3; m++){
							rhs[position5D(i,j1,k,m,c)] = rhs[position5D(i,j1,k,m,c)] - s[m]*r2;
						}

						p += 10;
					}
				}

				for(int m = 3; m<5; m++){
					n = (m-2)*5-1;
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							lhs[lhsPosition5D(i,j,k,n+2,c)] = lhs[lhsPosition5D(i,j,k,n+2,c)] - inBuffer[p+1]*lhs[lhsPosition5D(i,j,k,n+1,c)];
							lhs[lhsPosition5D(i,j,k,n+3,c)] = lhs[lhsPosition5D(i,j,k,n+3,c)] - inBuffer[p+2]*lhs[lhsPosition5D(i,j,k,n+1,c)];
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - inBuffer[p+3]*lhs[lhsPosition5D(i,j,k,n+1,c)];
							d = inBuffer[p+4];
							e = inBuffer[p+5];
							s[m] = inBuffer[p+6];
							r1 = lhs[lhsPosition5D(i,j,k,n+2,c)];
							lhs[lhsPosition5D(i,j,k,n+3,c)] = lhs[lhsPosition5D(i,j,k,n+3,c)] - d*r1;
							lhs[lhsPosition5D(i,j,k,n+4,c)] = lhs[lhsPosition5D(i,j,k,n+4,c)] - e*r1;
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - s[m]*r1;
							r2 = lhs[lhsPosition5D(i,j1,k,n+1,c)];
							lhs[lhsPosition5D(i,j1,k,n+2,c)] = lhs[lhsPosition5D(i,j1,k,n+2,c)] - d*r2;
							lhs[lhsPosition5D(i,j1,k,n+3,c)] = lhs[lhsPosition5D(i,j1,k,n+3,c)] - e*r2;
							rhs[position5D(i,j1,k,m,c)] = rhs[position5D(i,j1,k,m,c)] - s[m]*r2;
							p += 6;
						}
					}

				}

			}
			else{
				/* If this is the first cell, we still compute the lhs */
				lhsy(c);
			}

			/* Perform the Thomas algorithm; first, FORWARD ELIMINATION */
			yThomasForward(c, jstart, jend, isize, ksize);

			/*
			 * The last two rows in this grid block are a bit different,
			 * since they do not have two more rows available for the
			 * elimination of off-diagonal entries
			 */
			yThomasForwardLast2(c, jstart, jend, isize, ksize);

			/* Do the u+c and the u-c factors */
			for(int m = 3; m<5; m++){
				yUCFactors(m, c, jstart, jend, isize, ksize);

				/* And again the last two rows separately */
				yUCFactorsLast2(m, c, jstart, jend, isize, ksize);
			}

			/*
			 * Send information to the next processor, except when this
			 * is the last grid block
			 */
			if(stage != ncells-1){
				int j;
				/* Create a running pointer for the send buffer */
				int p = -1;
				n = -1;
				for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
					for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
						for(j = jend+1; j<=jend+2; j++){
							outBuffer[p+1] = lhs[lhsPosition5D(i,j,k,n+4,c)];
							outBuffer[p+2] = lhs[lhsPosition5D(i,j,k,n+5,c)];
							for(int m = 0; m<3; m++){
								outBuffer[p+3+m] = rhs[position5D(i,j,k,m,c)];
							}
							p += 5;
						}
					}
				}

				for(int m = 3; m<5; m++){
					n = (m-2)*5-1;
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							for(j = jend+1; j<=jend+2; j++){
								outBuffer[p+1] = lhs[lhsPosition5D(i,j,k,n+4,c)];
								outBuffer[p+2] = lhs[lhsPosition5D(i,j,k,n+5,c)];
								outBuffer[p+3] = rhs[position5D(i,j,k,m,c)];
								p += 3;
							}
						}
					}
				}

				/*
				 * Send data to next phase
				 * Can't receive data yet because buffer size will be wrong
				 */
				if(detailedTiming)
					timer.start(T_COMM);
				requests[1] = commSolve.Isend(outBuffer, 0, 22*bufferSize, MPI.DOUBLE, successor[1], DEFAULT_TAG);
				if(detailedTiming)
					timer.stop(T_COMM);

			}

		}

		/* Now go in the reverse direction */

		/* BACKSUBSTITUTION */
		for(int stage = ncells-1; stage>=0; stage--){
			int c = slice[1][stage];

			int jstart = 0;
			int jend = cellSize[1][c]-1;

			int isize = cellSize[0][c];
			int ksize = cellSize[2][c];

			int bufferSize = (isize-start[0][c]-end[0][c])*(ksize-start[2][c]-end[2][c]);

			if(stage != ncells-1){
				/*
				 * If this is not the starting cell in this row of cells,
				 * wait for a message to be received, containing the
				 * solution of the previous two stations
				 */
				if(detailedTiming)
					timer.start(T_COMM);
				requests[0] = commSolve.Irecv(inBuffer, 0, 10*bufferSize, MPI.DOUBLE, successor[1], DEFAULT_TAG);
				if(detailedTiming)
					timer.stop(T_COMM);

				/*
				 * Communication has already been started
				 * while waiting, do the block-diagonal inversion for the
				 * cell that was just finished
				 */
				pinvr(slice[1][stage+1]);

				/* Wait for pending communication to complete */
				if(detailedTiming)
					timer.start(T_COMM);
				Request.Waitall(requests);
				if(detailedTiming)
					timer.stop(T_COMM);

				/* Unpack the buffer for the first three factors */
				n = -1;
				int p = -1;
				int j = jend+2;
				int j1 = j-1;
				for(int m = 0; m<3; m++){
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							double sm1 = inBuffer[p+1];
							double sm2 = inBuffer[p+2];
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*sm1-lhs[lhsPosition5D(i,j,k,n+5,c)]*sm2;
							rhs[position5D(i,j1,k,m,c)] = rhs[position5D(i,j1,k,m,c)] - lhs[lhsPosition5D(i,j1,k,n+4,c)]*rhs[position5D(i,j,k,m,c)]-lhs[lhsPosition5D(i,j1,k,n+5,c)]*sm1;
							p += 2;
						}
					}
				}

				/* Now unpack the buffer for the remaining two factors */
				for(int m = 3; m<5; m++){
					n = (m-2)*5-1;
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							double sm1 = inBuffer[p+1];
							double sm2 = inBuffer[p+2];
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*sm1-lhs[lhsPosition5D(i,j,k,n+5,c)]*sm2;
							rhs[position5D(i,j1,k,m,c)] = rhs[position5D(i,j1,k,m,c)] - lhs[lhsPosition5D(i,j1,k,n+4,c)]*rhs[position5D(i,j,k,m,c)]-lhs[lhsPosition5D(i,j1,k,n+5,c)]*sm1;
							p += 2;
						}
					}
				}

			}
			else{
				/*
				 * Now we know this is the first grid block on the back sweep,
				 * so we don't need a message to start the substitution
				 */
				int j = jend+1;
				int j1 = jend+2;
				n = -1;
				for(int m = 0; m<3; m++){
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*rhs[position5D(i,j1,k,m,c)];
						}
					}
				}

				for(int m = 3; m<5; m++){
					n = (m-2)*5-1;
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*rhs[position5D(i,j1,k,m,c)];
						}
					}
				}

			}

			/*
			 * Whether or not this is the last processor, we always have
			 * to complete the back-substitution
			 */

			/* The first three factors */
			n = -1;
			for(int m = 0; m<3; m++){
				for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
					for(int j = jend; j>=jstart+2; j--){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							int j1 = j+1;
							int j2 = j+2;
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*rhs[position5D(i,j1,k,m,c)]-lhs[lhsPosition5D(i,j,k,n+5,c)]*rhs[position5D(i,j2,k,m,c)];
						}
					}
				}
			}

			/* And the remaining two */

			for(int m = 3; m<5; m++){
				n = (m-2)*5-1;
				for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
					for(int j = jend; j>=jstart+2; j--){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							int j1 = j+1;
							int j2 = j+2;
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*rhs[position5D(i,j1,k,m,c)]-lhs[lhsPosition5D(i,j,k,n+5,c)]*rhs[position5D(i,j2,k,m,c)];
						}
					}
				}
			}

			/* Send on information to the previous processor, if needed */
			if(stage != 0){
				int j = jstart+2;
				int j1 = jstart+3;
				int p = -1;
				for(int m = 0; m<5; m++){
					for(int k = start[2][c]+2; k<=ksize-end[2][c]+1; k++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							outBuffer[p+1] = rhs[position5D(i,j,k,m,c)];
							outBuffer[p+2] = rhs[position5D(i,j1,k,m,c)];
							p += 2;
						}
					}
				}

				/* Pack and send the buffer */
				if(detailedTiming)
					timer.start(T_COMM);
				requests[1] = commSolve.Isend(outBuffer, 0, 10*bufferSize, MPI.DOUBLE, predecessor[1], DEFAULT_TAG);
				if(detailedTiming)
					timer.stop(T_COMM);

			}

			/* If this was the last stage, do the block-diagonal inversion */
			if(stage == 0)
				pinvr(c);

		}

	}

	private void zThomasForward(int c, int kstart, int kend, int isize, int jsize){
		int n = -1;
		final int S = DIMI_DIMJ_DIMK, D = DIMI_DIMJ;
		final int o1 = (n+1)*S, o2 = (n+2)*S, o3 = (n+3)*S, o4 = (n+4)*S, o5 = (n+5)*S;
		for(int k = kstart+2; k<=kend; k++){
			for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
				for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
					/* every operand is read once into a local; the stores are those of the original */
					final int l = lhsPosition5D(i,j,k,0,c);
					final int r = position5D(i,j,k,0,c);
					final double fac1 = 1.0d/lhs[l+o3];
					final double c4 = fac1*lhs[l+o4];
					lhs[l+o4] = c4;
					final double c5 = fac1*lhs[l+o5];
					lhs[l+o5] = c5;
					final double q0 = fac1*rhs[r+0];
					rhs[r+0] = q0;
					final double q1 = fac1*rhs[r+S];
					rhs[r+S] = q1;
					final double q2 = fac1*rhs[r+2*S];
					rhs[r+2*S] = q2;
					final double b2 = lhs[l+D+o2];
					lhs[l+D+o3] = lhs[l+D+o3] - b2*c4;
					lhs[l+D+o4] = lhs[l+D+o4] - b2*c5;
					rhs[r+D+0] = rhs[r+D+0] - b2*q0;
					rhs[r+D+S] = rhs[r+D+S] - b2*q1;
					rhs[r+D+2*S] = rhs[r+D+2*S] - b2*q2;
					final double e1 = lhs[l+2*D+o1];
					lhs[l+2*D+o2] = lhs[l+2*D+o2] - e1*c4;
					lhs[l+2*D+o3] = lhs[l+2*D+o3] - e1*c5;
					rhs[r+2*D+0] = rhs[r+2*D+0] - e1*q0;
					rhs[r+2*D+S] = rhs[r+2*D+S] - e1*q1;
					rhs[r+2*D+2*S] = rhs[r+2*D+2*S] - e1*q2;
				}
			}
		}
	}

	private void zThomasForwardLast2(int c, int kstart, int kend, int isize, int jsize){
		int n = -1;
		int k = kend+1;
		int k1 = kend+2;
		for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
			for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
				double fac1 = 1.0d/lhs[lhsPosition5D(i,j,k,n+3,c)];
				lhs[lhsPosition5D(i,j,k,n+4,c)] = fac1*lhs[lhsPosition5D(i,j,k,n+4,c)];
				lhs[lhsPosition5D(i,j,k,n+5,c)] = fac1*lhs[lhsPosition5D(i,j,k,n+5,c)];
				for(int m = 0; m<3; m++){
					rhs[position5D(i,j,k,m,c)] = fac1*rhs[position5D(i,j,k,m,c)];
				}
				lhs[lhsPosition5D(i,j,k1,n+3,c)] = lhs[lhsPosition5D(i,j,k1,n+3,c)] - lhs[lhsPosition5D(i,j,k1,n+2,c)]*lhs[lhsPosition5D(i,j,k,n+4,c)];
				lhs[lhsPosition5D(i,j,k1,n+4,c)] = lhs[lhsPosition5D(i,j,k1,n+4,c)] - lhs[lhsPosition5D(i,j,k1,n+2,c)]*lhs[lhsPosition5D(i,j,k,n+5,c)];
				for(int m = 0; m<3; m++){
					rhs[position5D(i,j,k1,m,c)] = rhs[position5D(i,j,k1,m,c)] - lhs[lhsPosition5D(i,j,k1,n+2,c)]*rhs[position5D(i,j,k,m,c)];
				}
				/*
				 * Scale the last row immediately (some o this is
				 * overkill in case this is the last cell)
				 */
				double fac2 = 1.0d/lhs[lhsPosition5D(i,j,k1,n+3,c)];
				lhs[lhsPosition5D(i,j,k1,n+4,c)] = fac2*lhs[lhsPosition5D(i,j,k1,n+4,c)];
				lhs[lhsPosition5D(i,j,k1,n+5,c)] = fac2*lhs[lhsPosition5D(i,j,k1,n+5,c)];
				for(int m = 0; m<3; m++){
					rhs[position5D(i,j,k1,m,c)] = fac2*rhs[position5D(i,j,k1,m,c)];
				}
			}
		}
	}

	private void zUCFactors(int m, int c, int kstart, int kend, int isize, int jsize){
		int k, k1;
		int n = (m-2)*5-1;
		final int S = DIMI_DIMJ_DIMK, D = DIMI_DIMJ;
		final int o1 = (n+1)*S, o2 = (n+2)*S, o3 = (n+3)*S, o4 = (n+4)*S, o5 = (n+5)*S;
		final int om = m*S;
		for(k = kstart+2; k<=kend; k++){
			for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
				for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
					/* every operand is read once into a local; the stores are those of the original */
					final int l = lhsPosition5D(i,j,k,0,c);
					final int r = position5D(i,j,k,0,c);
					final double fac1 = 1.0d/lhs[l+o3];
					final double c4 = fac1*lhs[l+o4];
					lhs[l+o4] = c4;
					final double c5 = fac1*lhs[l+o5];
					lhs[l+o5] = c5;
					final double q0 = fac1*rhs[r+om];
					rhs[r+om] = q0;
					final double b2 = lhs[l+D+o2];
					lhs[l+D+o3] = lhs[l+D+o3] - b2*c4;
					lhs[l+D+o4] = lhs[l+D+o4] - b2*c5;
					rhs[r+D+om] = rhs[r+D+om] - b2*q0;
					final double e1 = lhs[l+2*D+o1];
					lhs[l+2*D+o2] = lhs[l+2*D+o2] - e1*c4;
					lhs[l+2*D+o3] = lhs[l+2*D+o3] - e1*c5;
					rhs[r+2*D+om] = rhs[r+2*D+om] - e1*q0;
				}
			}
		}
	}

	private void zUCFactorsLast2(int m, int c, int kstart, int kend, int isize, int jsize){
		int k = kend+1;
		int k1 = kend+2;
		int n = (m-2)*5-1;
		for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
			for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
				double fac1 = 1.0d/lhs[lhsPosition5D(i,j,k,n+3,c)];
				lhs[lhsPosition5D(i,j,k,n+4,c)] = fac1*lhs[lhsPosition5D(i,j,k,n+4,c)];
				lhs[lhsPosition5D(i,j,k,n+5,c)] = fac1*lhs[lhsPosition5D(i,j,k,n+5,c)];
				rhs[position5D(i,j,k,m,c)] = fac1*rhs[position5D(i,j,k,m,c)];
				lhs[lhsPosition5D(i,j,k1,n+3,c)] = lhs[lhsPosition5D(i,j,k1,n+3,c)] - lhs[lhsPosition5D(i,j,k1,n+2,c)]*lhs[lhsPosition5D(i,j,k,n+4,c)];
				lhs[lhsPosition5D(i,j,k1,n+4,c)] = lhs[lhsPosition5D(i,j,k1,n+4,c)] - lhs[lhsPosition5D(i,j,k1,n+2,c)]*lhs[lhsPosition5D(i,j,k,n+5,c)];
				rhs[position5D(i,j,k1,m,c)] = rhs[position5D(i,j,k1,m,c)] - lhs[lhsPosition5D(i,j,k1,n+2,c)]*rhs[position5D(i,j,k,m,c)];
				/*
				 * Scale the last row immediately (some o this is
				 * overkill in case this is the last cell)
				 */
				double fac2 = 1.0d/lhs[lhsPosition5D(i,j,k1,n+3,c)];
				lhs[lhsPosition5D(i,j,k1,n+4,c)] = fac2*lhs[lhsPosition5D(i,j,k1,n+4,c)];
				lhs[lhsPosition5D(i,j,k1,n+5,c)] = fac2*lhs[lhsPosition5D(i,j,k1,n+5,c)];
				rhs[position5D(i,j,k1,m,c)] = fac2*rhs[position5D(i,j,k1,m,c)];
			}
		}
	}

	/*
	 * This method performs the solution of the approximate factorization
	 * step in the z-direction for all five matrix components
	 * simultaneously. The Thomas algorithm is employed to solve the
	 * systems for the z-lines. Boundary conditions are non-periodic
	 */
	private void zSolve(){
		Request requests[] = new Request[2];
		double s[] = new double[5];
		double d, e, r1, r2;
		int n;

		/*
		 * Now do a sweep on a layer-by-layer basis, i.e. sweeping through cells
		 * on this node in the direction of increasing i for the forward sweep,
		 * and afer that reversing the direction for the backsubstitution.
		 */

		/* Forward elimination */
		for(int stage = 0; stage<ncells; stage++){
			int c = slice[2][stage];

			int kstart = 0;
			int kend = cellSize[2][c]-1;

			int isize = cellSize[0][c];
			int jsize = cellSize[1][c];

			int bufferSize = (isize-start[0][c]-end[0][c])*(jsize-start[1][c]-end[1][c]);

			if(stage != 0){

				/*
				 * If this is not the first processor in this row of cells,
				 * receive data from predecessor containing the right hand
				 * sides and the upper diagonal elements of the previous two rows
				 */
				if(detailedTiming)
					timer.start(T_COMM);
				requests[0] = commSolve.Irecv(inBuffer, 0, 22*bufferSize, MPI.DOUBLE, predecessor[2], DEFAULT_TAG);
				if(detailedTiming)
					timer.stop(T_COMM);

				/*
				 * Communication has already been started.
				 * Compute the left hand side while waiting for the msg
				 */
				lhsz(c);

				/*
				 * Wait for pending communication to complete.
				 * This waits on the current receive and on the send
				 * from the previous stage. They always come in pairs.
				 */
				if(detailedTiming)
					timer.start(T_COMM);
				Request.Waitall(requests);
				if(detailedTiming)
					timer.stop(T_COMM);

				/* Unpack the buffer */
				int k = kstart+2;
				int k1 = k+1;
				n = -1;

				/* Create a running pointer */
				int p = -1;
				for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
					for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
						lhs[lhsPosition5D(i,j,k,n+2,c)] = lhs[lhsPosition5D(i,j,k,n+2,c)] - inBuffer[p+1]*lhs[lhsPosition5D(i,j,k,n+1,c)];
						lhs[lhsPosition5D(i,j,k,n+3,c)] = lhs[lhsPosition5D(i,j,k,n+3,c)] - inBuffer[p+2]*lhs[lhsPosition5D(i,j,k,n+1,c)];
						for(int m = 0; m<3; m++){
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - inBuffer[p+3+m]*lhs[lhsPosition5D(i,j,k,n+1,c)];
						}
						d = inBuffer[p+6];
						e = inBuffer[p+7];
						for(int m = 0; m<3; m++){
							s[m] = inBuffer[p+8+m];
						}
						r1 = lhs[lhsPosition5D(i,j,k,n+2,c)];
						lhs[lhsPosition5D(i,j,k,n+3,c)] = lhs[lhsPosition5D(i,j,k,n+3,c)] - d*r1;
						lhs[lhsPosition5D(i,j,k,n+4,c)] = lhs[lhsPosition5D(i,j,k,n+4,c)] - e*r1;
						for(int m = 0; m<3; m++){
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - s[m]*r1;
						}
						r2 = lhs[lhsPosition5D(i,j,k1,n+1,c)];
						lhs[lhsPosition5D(i,j,k1,n+2,c)] = lhs[lhsPosition5D(i,j,k1,n+2,c)] - d*r2;
						lhs[lhsPosition5D(i,j,k1,n+3,c)] = lhs[lhsPosition5D(i,j,k1,n+3,c)] - e*r2;
						for(int m = 0; m<3; m++){
							rhs[position5D(i,j,k1,m,c)] = rhs[position5D(i,j,k1,m,c)] - s[m]*r2;
						}

						p += 10;
					}
				}

				for(int m = 3; m<5; m++){
					n = (m-2)*5-1;
					for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							lhs[lhsPosition5D(i,j,k,n+2,c)] = lhs[lhsPosition5D(i,j,k,n+2,c)] - inBuffer[p+1]*lhs[lhsPosition5D(i,j,k,n+1,c)];
							lhs[lhsPosition5D(i,j,k,n+3,c)] = lhs[lhsPosition5D(i,j,k,n+3,c)] - inBuffer[p+2]*lhs[lhsPosition5D(i,j,k,n+1,c)];
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - inBuffer[p+3]*lhs[lhsPosition5D(i,j,k,n+1,c)];
							d = inBuffer[p+4];
							e = inBuffer[p+5];
							s[m] = inBuffer[p+6];
							r1 = lhs[lhsPosition5D(i,j,k,n+2,c)];
							lhs[lhsPosition5D(i,j,k,n+3,c)] = lhs[lhsPosition5D(i,j,k,n+3,c)] - d*r1;
							lhs[lhsPosition5D(i,j,k,n+4,c)] = lhs[lhsPosition5D(i,j,k,n+4,c)] - e*r1;
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - s[m]*r1;
							r2 = lhs[lhsPosition5D(i,j,k1,n+1,c)];
							lhs[lhsPosition5D(i,j,k1,n+2,c)] = lhs[lhsPosition5D(i,j,k1,n+2,c)] - d*r2;
							lhs[lhsPosition5D(i,j,k1,n+3,c)] = lhs[lhsPosition5D(i,j,k1,n+3,c)] - e*r2;
							rhs[position5D(i,j,k1,m,c)] = rhs[position5D(i,j,k1,m,c)] - s[m]*r2;
							p += 6;
						}
					}

				}

			}
			else{
				/* If this is the first cell, we still compute the lhs */
				lhsz(c);
			}

			/* Perform the Thomas algorithm; first, FORWARD ELIMINATION */
			zThomasForward(c, kstart, kend, isize, jsize);

			/*
			 * The last two rows in this grid block are a bit different,
			 * since they do not have two more rows available for the
			 * elimination of off-diagonal entries
			 */
			zThomasForwardLast2(c, kstart, kend, isize, jsize);

			/* Do the u+c and the u-c factors */
			for(int m = 3; m<5; m++){
				zUCFactors(m, c, kstart, kend, isize, jsize);

				/* And again the last two rows separately */
				zUCFactorsLast2(m, c, kstart, kend, isize, jsize);
			}

			/*
			 * Send information to the next processor, except when this
			 * is the last grid block
			 */
			if(stage != ncells-1){
				int k;
				/* Create a running pointer for the send buffer */
				int p = -1;
				n = -1;
				for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
					for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
						for(k = kend+1; k<=kend+2; k++){
							outBuffer[p+1] = lhs[lhsPosition5D(i,j,k,n+4,c)];
							outBuffer[p+2] = lhs[lhsPosition5D(i,j,k,n+5,c)];
							for(int m = 0; m<3; m++){
								outBuffer[p+3+m] = rhs[position5D(i,j,k,m,c)];
							}
							p += 5;
						}
					}
				}

				for(int m = 3; m<5; m++){
					n = (m-2)*5-1;
					for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							for(k = kend+1; k<=kend+2; k++){
								outBuffer[p+1] = lhs[lhsPosition5D(i,j,k,n+4,c)];
								outBuffer[p+2] = lhs[lhsPosition5D(i,j,k,n+5,c)];
								outBuffer[p+3] = rhs[position5D(i,j,k,m,c)];
								p += 3;
							}
						}
					}
				}

				/*
				 * Send data to next phase
				 * Can't receive data yet because buffer size will be wrong
				 */
				if(detailedTiming)
					timer.start(T_COMM);
				requests[1] = commSolve.Isend(outBuffer, 0, 22*bufferSize, MPI.DOUBLE, successor[2], DEFAULT_TAG);
				if(detailedTiming)
					timer.stop(T_COMM);

			}

		}

		/* Now go in the reverse direction */

		/* BACKSUBSTITUTION */
		for(int stage = ncells-1; stage>=0; stage--){
			int c = slice[2][stage];

			int kstart = 0;
			int kend = cellSize[2][c]-1;

			int isize = cellSize[0][c];
			int jsize = cellSize[1][c];

			int bufferSize = (isize-start[0][c]-end[0][c])*(jsize-start[1][c]-end[1][c]);

			if(stage != ncells-1){
				/*
				 * If this is not the starting cell in this row of cells,
				 * wait for a message to be received, containing the
				 * solution of the previous two stations
				 */
				if(detailedTiming)
					timer.start(T_COMM);
				requests[0] = commSolve.Irecv(inBuffer, 0, 10*bufferSize, MPI.DOUBLE, successor[2], DEFAULT_TAG);
				if(detailedTiming)
					timer.stop(T_COMM);

				/*
				 * Communication has already been started
				 * while waiting, do the block-diagonal inversion for the
				 * cell that was just finished
				 */
				tzetar(slice[2][stage+1]);

				/* Wait for pending communication to complete */
				if(detailedTiming)
					timer.start(T_COMM);
				Request.Waitall(requests);
				if(detailedTiming)
					timer.stop(T_COMM);

				/* Unpack the buffer for the first three factors */
				n = -1;
				int p = -1;
				int k = kend+2;
				int k1 = k-1;
				for(int m = 0; m<3; m++){
					for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							double sm1 = inBuffer[p+1];
							double sm2 = inBuffer[p+2];
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*sm1-lhs[lhsPosition5D(i,j,k,n+5,c)]*sm2;
							rhs[position5D(i,j,k1,m,c)] = rhs[position5D(i,j,k1,m,c)] - lhs[lhsPosition5D(i,j,k1,n+4,c)]*rhs[position5D(i,j,k,m,c)]-lhs[lhsPosition5D(i,j,k1,n+5,c)]*sm1;
							p += 2;
						}
					}
				}

				/* Now unpack the buffer for the remaining two factors */
				for(int m = 3; m<5; m++){
					n = (m-2)*5-1;
					for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							double sm1 = inBuffer[p+1];
							double sm2 = inBuffer[p+2];
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*sm1-lhs[lhsPosition5D(i,j,k,n+5,c)]*sm2;
							rhs[position5D(i,j,k1,m,c)] = rhs[position5D(i,j,k1,m,c)] - lhs[lhsPosition5D(i,j,k1,n+4,c)]*rhs[position5D(i,j,k,m,c)]-lhs[lhsPosition5D(i,j,k1,n+5,c)]*sm1;
							p += 2;
						}
					}
				}

			}
			else{
				/*
				 * Now we know this is the first grid block on the back sweep,
				 * so we don't need a message to start the substitution
				 */
				int k = kend+1;
				int k1 = kend+2;
				n = -1;
				for(int m = 0; m<3; m++){
					for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*rhs[position5D(i,j,k1,m,c)];
						}
					}
				}

				for(int m = 3; m<5; m++){
					n = (m-2)*5-1;
					for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*rhs[position5D(i,j,k1,m,c)];
						}
					}
				}

			}

			/*
			 * Whether or not this is the last processor, we always have
			 * to complete the back-substitution
			 */

			/* The first three factors */
			n = -1;
			for(int m = 0; m<3; m++){
				for(int k = kend; k>=kstart+2; k--){
					for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							int k1 = k+1;
							int k2 = k+2;
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*rhs[position5D(i,j,k1,m,c)]-lhs[lhsPosition5D(i,j,k,n+5,c)]*rhs[position5D(i,j,k2,m,c)];
						}
					}
				}
			}

			/* And the remaining two */

			for(int m = 3; m<5; m++){
				n = (m-2)*5-1;
				for(int k = kend; k>=kstart+2; k--){
					for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							int k1 = k+1;
							int k2 = k+2;
							rhs[position5D(i,j,k,m,c)] = rhs[position5D(i,j,k,m,c)] - lhs[lhsPosition5D(i,j,k,n+4,c)]*rhs[position5D(i,j,k1,m,c)]-lhs[lhsPosition5D(i,j,k,n+5,c)]*rhs[position5D(i,j,k2,m,c)];
						}
					}
				}
			}

			/* Send on information to the previous processor, if needed */
			if(stage != 0){
				int k = kstart+2;
				int k1 = kstart+3;
				int p = -1;
				for(int m = 0; m<5; m++){
					for(int j = start[1][c]+2; j<=jsize-end[1][c]+1; j++){
						for(int i = start[0][c]+2; i<=isize-end[0][c]+1; i++){
							outBuffer[p+1] = rhs[position5D(i,j,k,m,c)];
							outBuffer[p+2] = rhs[position5D(i,j,k1,m,c)];
							p += 2;
						}
					}
				}

				/* Pack and send the buffer */
				if(detailedTiming)
					timer.start(T_COMM);
				requests[1] = commSolve.Isend(outBuffer, 0, 10*bufferSize, MPI.DOUBLE, predecessor[2], DEFAULT_TAG);
				if(detailedTiming)
					timer.stop(T_COMM);

			}

			/* If this was the last stage, do the block-diagonal inversion */
			if(stage == 0)
				tzetar(c);

		}

	}

	private void add(){
		for(int c = 0; c<ncells; c++){
			for(int m = 0; m<5; m++){
				for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
					for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
						for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
							u[position5D(i,j,k,m,c)] = u[position5D(i,j,k,m,c)]+rhs[position5D(i,j,k,m,c)];
						}
					}
				}
			}
		}
	}

	private void firstLoop(int c){
		double aux, rhoInv;
		for(int k = 0; k<=cellSize[2][c]+1;k++){
			int k1 = k+1;
			for(int j = 0; j<=cellSize[1][c]+1;j++){
				int j1 = j+1;
				for(int i = 0; i<=cellSize[0][c]+1;i++){
					int i1 = i+1;

					final int p = position4D(i,j,k,c);
					final int q = position5D(i1,j1,k1,0,c);
					final int S = DIMI_DIMJ_DIMK;
					final double u1 = u[q+S], u2 = u[q+2*S], u3 = u[q+3*S], u4 = u[q+4*S];
					rhoInv = 1.0d/u[q];
					rhoI[p] = rhoInv;
					us[p] = u1 * rhoInv;
					vs[p] = u2 * rhoInv;
					ws[p] = u3 * rhoInv;
					final double sq = (0.5*(
							u1*u1+
							u2*u2+
							u3*u3))*rhoInv;
					square[p] = sq;
					qs[p] = sq*rhoInv;
					/* don't need speed and ainx until the lhs computation */
					aux = c1c2*rhoInv*(u4 - sq);
					aux = Math.sqrt(aux);
					speed[p] = aux;
					ainv[p] = 1.0d/aux;
				}
			}
		}
	}

	/*
	 * Note on loopXi, loopEta and loopZeta: Fortran evaluates `rhs = rhs + A + B - C` from left to right,
	 * ((rhs + A) + B) - C. The `rhs = rhs + ...` form below keeps that association; `rhs += A + B - C` would
	 * add the whole sum to rhs at the end and round differently (1 ulp, amplified over the iterations).
	 */
	private void loopXi(int c){
		for(int k = start[2][c]+1; k<=cellSize[2][c]-end[2][c]; k++){
			int k1 = k+1;
			for(int j = start[1][c]+1; j<=cellSize[1][c]-end[1][c]; j++){
				int j1 = j+1;
				for(int i = start[0][c]+1; i<=cellSize[0][c]-end[0][c]; i++){
					int i1 = i+1;
					double uijk = us[position4D(i,j,k,c)];
					double up1 = us[position4D(i1,j,k,c)];
					double um1 = us[position4D(i-1,j,k,c)];

					rhs[position5D(i1,j1,k1,0,c)] = rhs[position5D(i1,j1,k1,0,c)] + dx1tx1 * (u[position5D(i1+1,j1,k1,0,c)] - 2*u[position5D(i1,j1,k1,0,c)] + u[position5D(i,j1,k1,0,c)]) -
					tx2*(u[position5D(i1+1,j1,k1,1,c)] - u[position5D(i1-1,j1,k1,1,c)]);

					rhs[position5D(i1,j1,k1,1,c)] = rhs[position5D(i1,j1,k1,1,c)] + dx2tx1 * (u[position5D(i1+1,j1,k1,1,c)] - 2*u[position5D(i1,j1,k1,1,c)] + u[position5D(i,j1,k1,1,c)]) +
					xxcon2*con43*(up1 - 2*uijk + um1) -
					tx2*(u[position5D(i1+1,j1,k1,1,c)]*up1 -
							u[position5D(i1-1,j1,k1,1,c)]*um1+
							(u[position5D(i1+1,j1,k1,4,c)] - square[position4D(i1,j,k,c)] -
									u[position5D(i1-1,j1,k1,4,c)] + square[position4D(i-1,j,k,c)])*c2);

					rhs[position5D(i1,j1,k1,2,c)] = rhs[position5D(i1,j1,k1,2,c)] + dx3tx1 * (u[position5D(i1+1,j1,k1,2,c)] - 2*u[position5D(i1,j1,k1,2,c)] + u[position5D(i,j1,k1,2,c)]) +
					xxcon2*(vs[position4D(i1,j,k,c)] - 2*vs[position4D(i,j,k,c)] + vs[position4D(i-1,j,k,c)]) -
					tx2*(u[position5D(i1+1,j1,k1,2,c)]*up1 - u[position5D(i,j1,k1,2,c)]*um1);

					rhs[position5D(i1,j1,k1,3,c)] = rhs[position5D(i1,j1,k1,3,c)] + dx4tx1 * (u[position5D(i1+1,j1,k1,3,c)] - 2*u[position5D(i1,j1,k1,3,c)] + u[position5D(i,j1,k1,3,c)]) +
					xxcon2*(ws[position4D(i1,j,k,c)] - 2*ws[position4D(i,j,k,c)] + ws[position4D(i-1,j,k,c)]) -
					tx2*(u[position5D(i1+1,j1,k1,3,c)]*up1 - u[position5D(i,j1,k1,3,c)]*um1);

					rhs[position5D(i1,j1,k1,4,c)] = rhs[position5D(i1,j1,k1,4,c)] + dx5tx1 * (u[position5D(i1+1,j1,k1,4,c)] - 2*u[position5D(i1,j1,k1,4,c)] + u[position5D(i,j1,k1,4,c)]) +
					xxcon3*(qs[position4D(i1,j,k,c)] - 2*qs[position4D(i,j,k,c)] + qs[position4D(i-1,j,k,c)]) +
					xxcon4*(up1*up1 - 2*uijk*uijk + um1*um1) +
					xxcon5*(u[position5D(i1+1,j1,k1,4,c)]*rhoI[position4D(i1,j,k,c)] -
							2*u[position5D(i1,j1,k1,4,c)]*rhoI[position4D(i,j,k,c)] +
							u[position5D(i,j1,k1,4,c)]*rhoI[position4D(i-1,j,k,c)]) -
							tx2*((c1*u[position5D(i1+1,j1,k1,4,c)] - c2*square[position4D(i1,j,k,c)])*up1 -
									(c1*u[position5D(i,j1,k1,4,c)] - c2*square[position4D(i-1,j,k,c)])*um1);
				}
			}
		}
	}

	private void loopEta(int c){
		for(int k = start[2][c]+1; k<=cellSize[2][c]-end[2][c]; k++){
			int k1 = k+1;
			for(int j = start[1][c]+1; j<=cellSize[1][c]-end[1][c]; j++){
				int j1 = j+1;
				for(int i = start[0][c]+1; i<=cellSize[0][c]-end[0][c]; i++){
					int i1 = i+1;
					double vijk = vs[position4D(i,j,k,c)];
					double vp1 = vs[position4D(i,j1,k,c)];
					double vm1 = vs[position4D(i,j-1,k,c)];

					rhs[position5D(i1,j1,k1,0,c)] = rhs[position5D(i1,j1,k1,0,c)] + dy1ty1 * (u[position5D(i1,j1+1,k1,0,c)] - 2*u[position5D(i1,j1,k1,0,c)] + u[position5D(i1,j,k1,0,c)]) -
					ty2*(u[position5D(i1,j1+1,k1,2,c)] - u[position5D(i1,j1-1,k1,2,c)]);

					rhs[position5D(i1,j1,k1,1,c)] = rhs[position5D(i1,j1,k1,1,c)] + dy2ty1 * (u[position5D(i1,j1+1,k1,1,c)] - 2*u[position5D(i1,j1,k1,1,c)] + u[position5D(i1,j,k1,1,c)]) +
					yycon2*(us[position4D(i,j1,k,c)] - 2*us[position4D(i,j,k,c)] + us[position4D(i,j-1,k,c)]) -
					ty2*(u[position5D(i1,j1+1,k1,1,c)]*vp1 - u[position5D(i1,j,k1,1,c)]*vm1);

					rhs[position5D(i1,j1,k1,2,c)] = rhs[position5D(i1,j1,k1,2,c)] + dy3ty1 * (u[position5D(i1,j1+1,k1,2,c)] - 2*u[position5D(i1,j1,k1,2,c)] + u[position5D(i1,j,k1,2,c)]) +
					yycon2*con43*(vp1 - 2*vijk + vm1) -
					ty2*(u[position5D(i1,j1+1,k1,2,c)]*vp1 -
							u[position5D(i1,j1-1,k1,2,c)]*vm1+
							(u[position5D(i1,j1+1,k1,4,c)] - square[position4D(i,j1,k,c)] -
									u[position5D(i1,j1-1,k1,4,c)] + square[position4D(i,j-1,k,c)])*c2);

					rhs[position5D(i1,j1,k1,3,c)] = rhs[position5D(i1,j1,k1,3,c)] + dy4ty1 * (u[position5D(i1,j1+1,k1,3,c)] - 2*u[position5D(i1,j1,k1,3,c)] + u[position5D(i1,j,k1,3,c)]) +
					yycon2*(ws[position4D(i,j1,k,c)] - 2*ws[position4D(i,j,k,c)] + ws[position4D(i,j-1,k,c)]) -
					ty2*(u[position5D(i1,j1+1,k1,3,c)]*vp1 - u[position5D(i1,j,k1,3,c)]*vm1);

					rhs[position5D(i1,j1,k1,4,c)] = rhs[position5D(i1,j1,k1,4,c)] + dy5ty1 * (u[position5D(i1,j1+1,k1,4,c)] - 2*u[position5D(i1,j1,k1,4,c)] + u[position5D(i1,j,k1,4,c)]) +
					yycon3*(qs[position4D(i,j1,k,c)] - 2*qs[position4D(i,j,k,c)] + qs[position4D(i,j-1,k,c)]) +
					yycon4*(vp1*vp1 - 2*vijk*vijk + vm1*vm1) +
					yycon5*(u[position5D(i1,j1+1,k1,4,c)]*rhoI[position4D(i,j1,k,c)] -
							2*u[position5D(i1,j1,k1,4,c)]*rhoI[position4D(i,j,k,c)] +
							u[position5D(i1,j,k1,4,c)]*rhoI[position4D(i,j-1,k,c)]) -
							ty2*((c1*u[position5D(i1,j1+1,k1,4,c)] - c2*square[position4D(i,j1,k,c)])*vp1 -
									(c1*u[position5D(i1,j,k1,4,c)] - c2*square[position4D(i,j-1,k,c)])*vm1);
				}
			}
		}
	}

	private void loopZeta(int c){
		for(int k = start[2][c]+1; k<=cellSize[2][c]-end[2][c]; k++){
			int k1 = k+1;
			for(int j = start[1][c]+1; j<=cellSize[1][c]-end[1][c]; j++){
				int j1 = j+1;
				for(int i = start[0][c]+1; i<=cellSize[0][c]-end[0][c]; i++){
					int i1 = i+1;
					double wijk = ws[position4D(i,j,k,c)];
					double wp1 = ws[position4D(i,j,k1,c)];
					double wm1 = ws[position4D(i,j,k-1,c)];

					rhs[position5D(i1,j1,k1,0,c)] = rhs[position5D(i1,j1,k1,0,c)] + dz1tz1 * (u[position5D(i1,j1,k1+1,0,c)] - 2*u[position5D(i1,j1,k1,0,c)] + u[position5D(i1,j1,k,0,c)]) -
					tz2*(u[position5D(i1,j1,k1+1,3,c)] - u[position5D(i1,j1,k1-1,3,c)]);

					rhs[position5D(i1,j1,k1,1,c)] = rhs[position5D(i1,j1,k1,1,c)] + dz2tz1 * (u[position5D(i1,j1,k1+1,1,c)] - 2*u[position5D(i1,j1,k1,1,c)] + u[position5D(i1,j1,k,1,c)]) +
					zzcon2*(us[position4D(i,j,k1,c)] - 2*us[position4D(i,j,k,c)] + us[position4D(i,j,k-1,c)]) -
					tz2*(u[position5D(i1,j1,k1+1,1,c)]*wp1 - u[position5D(i1,j1,k,1,c)]*wm1);

					rhs[position5D(i1,j1,k1,2,c)] = rhs[position5D(i1,j1,k1,2,c)] + dz3tz1 * (u[position5D(i1,j1,k1+1,2,c)] - 2*u[position5D(i1,j1,k1,2,c)] + u[position5D(i1,j1,k,2,c)]) +
					zzcon2*(vs[position4D(i,j,k1,c)] - 2*vs[position4D(i,j,k,c)] + vs[position4D(i,j,k-1,c)]) -
					tz2*(u[position5D(i1,j1,k1+1,2,c)]*wp1 - u[position5D(i1,j1,k,2,c)]*wm1);

					rhs[position5D(i1,j1,k1,3,c)] = rhs[position5D(i1,j1,k1,3,c)] + dz4tz1 * (u[position5D(i1,j1,k1+1,3,c)] - 2*u[position5D(i1,j1,k1,3,c)] + u[position5D(i1,j1,k,3,c)]) +
					zzcon2*con43*(wp1 - 2*wijk + wm1) -
					tz2*(u[position5D(i1,j1,k1+1,3,c)]*wp1 -
							u[position5D(i1,j1,k1-1,3,c)]*wm1+
							(u[position5D(i1,j1,k1+1,4,c)] - square[position4D(i,j,k1,c)] -
									u[position5D(i1,j1,k1-1,4,c)] + square[position4D(i,j,k-1,c)])*c2);

					rhs[position5D(i1,j1,k1,4,c)] = rhs[position5D(i1,j1,k1,4,c)] + dz5tz1 * (u[position5D(i1,j1,k1+1,4,c)] - 2*u[position5D(i1,j1,k1,4,c)] + u[position5D(i1,j1,k,4,c)]) +
					zzcon3*(qs[position4D(i,j,k1,c)] - 2*qs[position4D(i,j,k,c)] + qs[position4D(i,j,k-1,c)]) +
					zzcon4*(wp1*wp1 - 2*wijk*wijk + wm1*wm1) +
					zzcon5*(u[position5D(i1,j1,k1+1,4,c)]*rhoI[position4D(i,j,k1,c)] -
							2*u[position5D(i1,j1,k1,4,c)]*rhoI[position4D(i,j,k,c)] +
							u[position5D(i1,j1,k,4,c)]*rhoI[position4D(i,j,k-1,c)]) -
							tz2*((c1*u[position5D(i1,j1,k1+1,4,c)] - c2*square[position4D(i,j,k1,c)])*wp1 -
									(c1*u[position5D(i1,j1,k,4,c)] - c2*square[position4D(i,j,k-1,c)])*wm1); 
				}
			}
		}
	}

	private void computeRHS(){

		/* Loop over all cells owned by this node */
		for(int c = 0; c<ncells; c++){

			/*
			 * Compute the reciprocal of density, and the kinetic energy,
			 * and the speed of sound
			 */
			firstLoop(c);
			rhsForcing(c);


			/* Compute xi-direction fluxes */
			loopXi(c);
			rhsDissipXi(c);


			/* Compute eta-direction fluxes */
			loopEta(c);
			rhsDissipEta(c);


			/* Compute zeta-direction fluxes */
			loopZeta(c);
			rhsDissipZeta(c);


			rhsScale(c);

		}

	}

	/* copy of the exact forcing term to rhs of the cell c (part of compute_rhs) */
	private void rhsForcing(int c){

		/*
		 * Copy the exact forcing term to the right hand side; because
		 * this forcing term is known, we can store it on the whole of every
		 * cell, including the boundary
		 */
		for(int m = 0; m<5; m++){
			for(int k = 0; k<=cellSize[2][c]-1; k++){
				int k2 = k+2;
				for(int j = 0; j<=cellSize[1][c]-1; j++){
					int j2 = j+2;
					for(int i = 0; i<=cellSize[0][c]-1; i++){
						rhs[position5D(i+2,j2,k2,m,c)] = forcing[forcPosition5D(i,j,k,m,c)];
					}	
				}	
			}
		}
	}

	/* fourth order xi-direction dissipation of the cell c (part of compute_rhs) */
	private void rhsDissipXi(int c){

		/*
		 * Add fourth order xi-direction dissipation
		 */
		if(start[0][c] > 0){

			int i = 3;
			for(int m = 0; m<5; m++){
				for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
					for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(5*u[position5D(i,j,k,m,c)] - 4*u[position5D(i+1,j,k,m,c)] + u[position5D(i+2,j,k,m,c)]); 
					}
				}
			}

			i = 4;
			for(int m = 0; m<5; m++){
				for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
					for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(-4*u[position5D(i-1,j,k,m,c)] + 6*u[position5D(i,j,k,m,c)] - 4*u[position5D(i+1,j,k,m,c)] +u[position5D(i+2,j,k,m,c)]);
					}
				}
			}

		}

		for(int m = 0; m<5; m++){
			for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
				for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
				/* u[i-2..i+2] travel in locals: rhs and u have the same layout and alignment, so reloading u[i-1]
				 * and u[i-2] right after the store to rhs of the previous points stalls on a false dependency */
				final int ilo = 3*start[0][c]+2, ihi = cellSize[0][c]-3*end[0][c]+1;
				if(ilo <= ihi){
					int p = position5D(ilo,j,k,m,c);
					double um2 = u[p-2], um1 = u[p-1], u0 = u[p], up1 = u[p+1];
					for(int i = ilo; i<=ihi; i++){
						final double up2 = u[p+2];
						rhs[p] -= dssp*(um2 - 4*um1
								+6*u0 - 4*up1 + up2);
						um2 = um1; um1 = u0; u0 = up1; up1 = up2;
						p++;
					}
				}
				}
			}
		}

		if(end[0][c] > 0){

			int i = cellSize[0][c]-1;
			for(int m = 0; m<5; m++){
				for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
					for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(u[position5D(i-2,j,k,m,c)] - 4*u[position5D(i-1,j,k,m,c)] + 6*u[position5D(i,j,k,m,c)] - 4*u[position5D(i+1,j,k,m,c)]);
					}
				}
			}

			i = cellSize[0][c];
			for(int m = 0; m<5; m++){
				for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
					for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(u[position5D(i-2,j,k,m,c)] - 4*u[position5D(i-1,j,k,m,c)] + 5*u[position5D(i,j,k,m,c)]);
					}
				}
			}

		}
	}

	/* fourth order eta-direction dissipation of the cell c (part of compute_rhs) */
	private void rhsDissipEta(int c){

		/*
		 * Add fourth order eta-direction dissipation
		 */
		if(start[1][c] > 0){

			int j = 3;
			for(int m = 0; m<5; m++){
				for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
					for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(5*u[position5D(i,j,k,m,c)] - 4*u[position5D(i,j+1,k,m,c)] + u[position5D(i,j+2,k,m,c)]);
					}
				}
			}

			j = 4;
			for(int m = 0; m<5; m++){
				for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
					for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(-4*u[position5D(i,j-1,k,m,c)] + 6*u[position5D(i,j,k,m,c)] - 4*u[position5D(i,j+1,k,m,c)] +u[position5D(i,j+2,k,m,c)]);
					}
				}
			}

		}

		for(int m = 0; m<5; m++){
			for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
				for(int j = 3*start[1][c]+2; j<=cellSize[1][c]-3*end[1][c]+1; j++){
					for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(u[position5D(i,j-2,k,m,c)] - 4*u[position5D(i,j-1,k,m,c)]
						                                                                     +6*u[position5D(i,j,k,m,c)] - 4*u[position5D(i,j+1,k,m,c)] + u[position5D(i,j+2,k,m,c)]);
					}
				}
			}
		}

		if(end[1][c] > 0){

			int j = cellSize[1][c]-1;
			for(int m = 0; m<5; m++){
				for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
					for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(u[position5D(i,j-2,k,m,c)] - 4*u[position5D(i,j-1,k,m,c)] + 6*u[position5D(i,j,k,m,c)] - 4*u[position5D(i,j+1,k,m,c)]);
					}
				}
			}

			j = cellSize[1][c];
			for(int m = 0; m<5; m++){
				for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
					for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(u[position5D(i,j-2,k,m,c)] - 4*u[position5D(i,j-1,k,m,c)] + 5*u[position5D(i,j,k,m,c)]);
					}
				}
			}

		}
	}

	/* fourth order zeta-direction dissipation of the cell c (part of compute_rhs) */
	private void rhsDissipZeta(int c){

		/*
		 * Add fourth order zeta-direction dissipation
		 */
		if(start[2][c] > 0){

			int k = 3;
			for(int m = 0; m<5; m++){
				for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
					for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(5*u[position5D(i,j,k,m,c)] - 4*u[position5D(i,j,k+1,m,c)] + u[position5D(i,j,k+2,m,c)]); 
					}
				}
			}

			k = 4;
			for(int m = 0; m<5; m++){
				for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
					for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(-4*u[position5D(i,j,k-1,m,c)] + 6*u[position5D(i,j,k,m,c)] - 4*u[position5D(i,j,k+1,m,c)] +u[position5D(i,j,k+2,m,c)]);
					}
				}
			}

		}

		for(int m = 0; m<5; m++){
			for(int k = 3*start[2][c]+2; k<=cellSize[2][c]-3*end[2][c]+1; k++){
				for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
					for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(u[position5D(i,j,k-2,m,c)] - 4*u[position5D(i,j,k-1,m,c)]
						                                                                     +6*u[position5D(i,j,k,m,c)] - 4*u[position5D(i,j,k+1,m,c)] + u[position5D(i,j,k+2,m,c)]);
					}
				}
			}
		}

		if(end[2][c] > 0){

			int k = cellSize[2][c]-1;
			for(int m = 0; m<5; m++){
				for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
					for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(u[position5D(i,j,k-2,m,c)] - 4*u[position5D(i,j,k-1,m,c)] + 6*u[position5D(i,j,k,m,c)] - 4*u[position5D(i,j,k+1,m,c)]);
					}
				}
			}

			k = cellSize[2][c];
			for(int m = 0; m<5; m++){
				for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
					for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
						rhs[position5D(i,j,k,m,c)] -= dssp*(u[position5D(i,j,k-2,m,c)] - 4*u[position5D(i,j,k-1,m,c)] + 5*u[position5D(i,j,k,m,c)]);
					}
				}
			}

		}
	}

	/* final scaling of rhs by dt of the cell c (part of compute_rhs) */
	private void rhsScale(int c){
		for(int m = 0; m<5; m++){
			for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
				for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
					for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
						rhs[position5D(i,j,k,m,c)] *= dt;
					}
				}
			}
		}

	}

	/*
	 * This function computes the left hand side for the three x-factors
	 */
	private void lhsx(int c){

		/* Treat only cell c */

		/* First fill the lhs for the u-eigenvalue */
		for(int k = start[2][c]+1; k<=cellSize[2][c]-end[2][c]; k++){
			int k1 = k+1;
			for(int j = start[1][c]+1; j<=cellSize[1][c]-end[1][c]; j++){
				int j1 = j+1;
				for(int i = start[0][c]; i<=cellSize[0][c]-end[0][c]+1; i++){
					int i1 = i+1;
					double ru1 = c3c4*rhoI[position4D(i,j,k,c)];
					cv[i1] = us[position4D(i,j,k,c)];
					rhon[i1] = dmax1(dx2+con43*ru1,dx5+c1c5*ru1,dxmax+ru1,dx1); 
				}

				for(int i = start[0][c]+1; i<=cellSize[0][c]-end[0][c]; i++){
					int i1 = i+1;
					lhs[lhsPosition5D(i1,j1,k1,0,c)] = 0;
					lhs[lhsPosition5D(i1,j1,k1,1,c)] = -dttx2*cv[i] - dttx1*rhon[i];
					lhs[lhsPosition5D(i1,j1,k1,2,c)] = 1 + c2dttx1*rhon[i1];
					lhs[lhsPosition5D(i1,j1,k1,3,c)] = dttx2*cv[i1+1] - dttx1*rhon[i1+1];
					lhs[lhsPosition5D(i1,j1,k1,4,c)] = 0;
				}
			}
		}

		/* Add fourth order dissipation */
		if(start[0][c] > 0){
			int i = 3;
			for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
				for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
					lhs[lhsPosition5D(i,j,k,2,c)] +=  comz5;
					lhs[lhsPosition5D(i,j,k,3,c)] = lhs[lhsPosition5D(i,j,k,3,c)] - comz4;
					lhs[lhsPosition5D(i,j,k,4,c)] +=  comz1;

					lhs[lhsPosition5D(i+1,j,k,1,c)] = lhs[lhsPosition5D(i+1,j,k,1,c)] - comz4;
					lhs[lhsPosition5D(i+1,j,k,2,c)] +=  comz6;
					lhs[lhsPosition5D(i+1,j,k,3,c)] = lhs[lhsPosition5D(i+1,j,k,3,c)] - comz4;
					lhs[lhsPosition5D(i+1,j,k,4,c)] +=  comz1;
				}
			}
		}

		for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
			for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
				for(int i = 3*start[0][c]+2; i<=cellSize[0][c]-3*end[0][c]+1; i++){
					lhs[lhsPosition5D(i,j,k,0,c)] +=  comz1;
					lhs[lhsPosition5D(i,j,k,1,c)] = lhs[lhsPosition5D(i,j,k,1,c)] - comz4;
					lhs[lhsPosition5D(i,j,k,2,c)] +=  comz6;
					lhs[lhsPosition5D(i,j,k,3,c)] = lhs[lhsPosition5D(i,j,k,3,c)] - comz4;
					lhs[lhsPosition5D(i,j,k,4,c)] +=  comz1;
				}
			}
		}

		if(end[0][c] > 0){
			int i = cellSize[0][c]-1;
			for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
				for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
					lhs[lhsPosition5D(i,j,k,0,c)] +=  comz1;
					lhs[lhsPosition5D(i,j,k,1,c)] = lhs[lhsPosition5D(i,j,k,1,c)] - comz4;
					lhs[lhsPosition5D(i,j,k,2,c)] +=  comz6;
					lhs[lhsPosition5D(i,j,k,3,c)] = lhs[lhsPosition5D(i,j,k,3,c)] - comz4;

					lhs[lhsPosition5D(i+1,j,k,0,c)] +=  comz1;
					lhs[lhsPosition5D(i+1,j,k,1,c)] = lhs[lhsPosition5D(i+1,j,k,1,c)] - comz4;
					lhs[lhsPosition5D(i+1,j,k,2,c)] +=  comz5;
				}
			}
		}

		/*
		 * Subsequently, fill the other factors (u+c), (u-c) by a4ing
		 * to the first
		 */
		for(int k = start[2][c]+1; k<=cellSize[2][c]-end[2][c]; k++){
			int k1 = k+1;
			for(int j = start[1][c]+1; j<=cellSize[1][c]-end[1][c]; j++){
				int j1 = j+1;
				for(int i = start[0][c]+1; i<=cellSize[0][c]-end[0][c]; i++){
					int i1 = i+1;
					lhs[lhsPosition5D(i1,j1,k1,5,c)] = lhs[lhsPosition5D(i1,j1,k1,0,c)];
					lhs[lhsPosition5D(i1,j1,k1,6,c)] = lhs[lhsPosition5D(i1,j1,k1,1,c)] - dttx2*speed[position4D(i-1,j,k,c)];
					lhs[lhsPosition5D(i1,j1,k1,7,c)] = lhs[lhsPosition5D(i1,j1,k1,2,c)];
					lhs[lhsPosition5D(i1,j1,k1,8,c)] = lhs[lhsPosition5D(i1,j1,k1,3,c)] + dttx2*speed[position4D(i+1,j,k,c)];
					lhs[lhsPosition5D(i1,j1,k1,9,c)] = lhs[lhsPosition5D(i1,j1,k1,4,c)];

					lhs[lhsPosition5D(i1,j1,k1,10,c)] = lhs[lhsPosition5D(i1,j1,k1,0,c)];
					lhs[lhsPosition5D(i1,j1,k1,11,c)] = lhs[lhsPosition5D(i1,j1,k1,1,c)] + dttx2*speed[position4D(i-1,j,k,c)];
					lhs[lhsPosition5D(i1,j1,k1,12,c)] = lhs[lhsPosition5D(i1,j1,k1,2,c)];
					lhs[lhsPosition5D(i1,j1,k1,13,c)] = lhs[lhsPosition5D(i1,j1,k1,3,c)] - dttx2*speed[position4D(i+1,j,k,c)];
					lhs[lhsPosition5D(i1,j1,k1,14,c)] = lhs[lhsPosition5D(i1,j1,k1,4,c)];
				}
			}
		}

	}

	/*
	 * Block-diagonal matrix-vector multiplication
	 */
	private void ninvr(int c){

		double r1, r2, r3, r4, r5, t1, t2;

		/* Treat only one cell */
		for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
			for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
				for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){

					r1 = rhs[position5D(i,j,k,0,c)];
					r2 = rhs[position5D(i,j,k,1,c)];
					r3 = rhs[position5D(i,j,k,2,c)];
					r4 = rhs[position5D(i,j,k,3,c)];
					r5 = rhs[position5D(i,j,k,4,c)];

					t1 = bt*r3;
					t2 = 0.5*(r4+r5);

					rhs[position5D(i,j,k,0,c)] = -r2;
					rhs[position5D(i,j,k,1,c)] = r1;
					rhs[position5D(i,j,k,2,c)] = bt*(r4-r5);
					rhs[position5D(i,j,k,3,c)] = -t1+t2;
					rhs[position5D(i,j,k,4,c)] = t1+t2;

				}
			}
		}

	}

	/*
	 * This function computes the left hand side for the three y-factors
	 */
	private void lhsy(int c){

		/* Treat only cell c */

		/* First fill the lhs for the u-eigenvalue */
		for(int k = start[2][c]+1; k<=cellSize[2][c]-end[2][c]; k++){
			int k1 = k+1;
			for(int i = start[0][c]+1; i<=cellSize[0][c]-end[0][c]; i++){
				int i1 = i+1;
				for(int j = start[1][c]; j<=cellSize[1][c]-end[1][c]+1; j++){
					int j1 = j+1;
					double ru1 = c3c4*rhoI[position4D(i,j,k,c)];
					cv[j1] = vs[position4D(i,j,k,c)];
					rhoq[j1] = dmax1(dy3+con43*ru1,dy5+c1c5*ru1,dymax+ru1,dy1); 
				}

				for(int j = start[1][c]+1; j<=cellSize[1][c]-end[1][c]; j++){
					int j1 = j+1;
					lhs[lhsPosition5D(i1,j1,k1,0,c)] = 0;
					lhs[lhsPosition5D(i1,j1,k1,1,c)] = -dtty2*cv[j] - dtty1*rhoq[j];
					lhs[lhsPosition5D(i1,j1,k1,2,c)] = 1 + c2dtty1*rhoq[j1];
					lhs[lhsPosition5D(i1,j1,k1,3,c)] = dtty2*cv[j1+1] - dtty1*rhoq[j1+1];
					lhs[lhsPosition5D(i1,j1,k1,4,c)] = 0;
				}
			}
		}

		/* Add fourth order dissipation */
		if(start[1][c] > 0){
			int j = 3;
			for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
				for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
					lhs[lhsPosition5D(i,j,k,2,c)] +=  comz5;
					lhs[lhsPosition5D(i,j,k,3,c)] = lhs[lhsPosition5D(i,j,k,3,c)] - comz4;
					lhs[lhsPosition5D(i,j,k,4,c)] +=  comz1;

					lhs[lhsPosition5D(i,j+1,k,1,c)] = lhs[lhsPosition5D(i,j+1,k,1,c)] - comz4;
					lhs[lhsPosition5D(i,j+1,k,2,c)] +=  comz6;
					lhs[lhsPosition5D(i,j+1,k,3,c)] = lhs[lhsPosition5D(i,j+1,k,3,c)] - comz4;
					lhs[lhsPosition5D(i,j+1,k,4,c)] +=  comz1;
				}
			}
		}

		for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
			for(int j = 3*start[1][c]+2; j<=cellSize[1][c]-3*end[1][c]+1; j++){
				for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
					lhs[lhsPosition5D(i,j,k,0,c)] +=  comz1;
					lhs[lhsPosition5D(i,j,k,1,c)] = lhs[lhsPosition5D(i,j,k,1,c)] - comz4;
					lhs[lhsPosition5D(i,j,k,2,c)] +=  comz6;
					lhs[lhsPosition5D(i,j,k,3,c)] = lhs[lhsPosition5D(i,j,k,3,c)] - comz4;
					lhs[lhsPosition5D(i,j,k,4,c)] +=  comz1;
				}
			}
		}

		if(end[1][c] > 0){
			int j = cellSize[1][c]-1;
			for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
				for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
					lhs[lhsPosition5D(i,j,k,0,c)] +=  comz1;
					lhs[lhsPosition5D(i,j,k,1,c)] = lhs[lhsPosition5D(i,j,k,1,c)] - comz4;
					lhs[lhsPosition5D(i,j,k,2,c)] +=  comz6;
					lhs[lhsPosition5D(i,j,k,3,c)] = lhs[lhsPosition5D(i,j,k,3,c)] - comz4;

					lhs[lhsPosition5D(i,j+1,k,0,c)] +=  comz1;
					lhs[lhsPosition5D(i,j+1,k,1,c)] = lhs[lhsPosition5D(i,j+1,k,1,c)] - comz4;
					lhs[lhsPosition5D(i,j+1,k,2,c)] +=  comz5;
				}
			}
		}

		/*
		 * Subsequently, fill the other factors (u+c), (u-c) by a4ing
		 * to the first
		 */
		for(int k = start[2][c]+1; k<=cellSize[2][c]-end[2][c]; k++){
			int k1 = k+1;
			for(int j = start[1][c]+1; j<=cellSize[1][c]-end[1][c]; j++){
				int j1 = j+1;
				for(int i = start[0][c]+1; i<=cellSize[0][c]-end[0][c]; i++){
					int i1 = i+1;
					lhs[lhsPosition5D(i1,j1,k1,5,c)] = lhs[lhsPosition5D(i1,j1,k1,0,c)];
					lhs[lhsPosition5D(i1,j1,k1,6,c)] = lhs[lhsPosition5D(i1,j1,k1,1,c)] - dtty2*speed[position4D(i,j-1,k,c)];
					lhs[lhsPosition5D(i1,j1,k1,7,c)] = lhs[lhsPosition5D(i1,j1,k1,2,c)];
					lhs[lhsPosition5D(i1,j1,k1,8,c)] = lhs[lhsPosition5D(i1,j1,k1,3,c)] + dtty2*speed[position4D(i,j+1,k,c)];
					lhs[lhsPosition5D(i1,j1,k1,9,c)] = lhs[lhsPosition5D(i1,j1,k1,4,c)];

					lhs[lhsPosition5D(i1,j1,k1,10,c)] = lhs[lhsPosition5D(i1,j1,k1,0,c)];
					lhs[lhsPosition5D(i1,j1,k1,11,c)] = lhs[lhsPosition5D(i1,j1,k1,1,c)] + dtty2*speed[position4D(i,j-1,k,c)];
					lhs[lhsPosition5D(i1,j1,k1,12,c)] = lhs[lhsPosition5D(i1,j1,k1,2,c)];
					lhs[lhsPosition5D(i1,j1,k1,13,c)] = lhs[lhsPosition5D(i1,j1,k1,3,c)] - dtty2*speed[position4D(i,j+1,k,c)];
					lhs[lhsPosition5D(i1,j1,k1,14,c)] = lhs[lhsPosition5D(i1,j1,k1,4,c)];
				}
			}
		}

	}

	/*
	 * Block-diagonal matrix-vector multiplication
	 */
	private void pinvr(int c){

		double r1, r2, r3, r4, r5, t1, t2;

		/* Treat only one cell */
		for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
			for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
				for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){

					r1 = rhs[position5D(i,j,k,0,c)];
					r2 = rhs[position5D(i,j,k,1,c)];
					r3 = rhs[position5D(i,j,k,2,c)];
					r4 = rhs[position5D(i,j,k,3,c)];
					r5 = rhs[position5D(i,j,k,4,c)];

					t1 = bt*r1;
					t2 = 0.5*(r4+r5);

					rhs[position5D(i,j,k,0,c)] = bt*(r4-r5);
					rhs[position5D(i,j,k,1,c)] = -r3;
					rhs[position5D(i,j,k,2,c)] = r2;
					rhs[position5D(i,j,k,3,c)] = -t1+t2;
					rhs[position5D(i,j,k,4,c)] = t1+t2;

				}
			}
		}

	}

	/*
	 * This function computes the left hand side for the three z-factors
	 */
	private void lhsz(int c){

		/* Treat only cell c */

		/* First fill the lhs for the u-eigenvalue */
		for(int j = start[1][c]+1; j<=cellSize[1][c]-end[1][c]; j++){
			int j1 = j+1;
			for(int i = start[0][c]+1; i<=cellSize[0][c]-end[0][c]; i++){
				int i1 = i+1;
				for(int k = start[2][c]; k<=cellSize[2][c]-end[2][c]+1; k++){
					int k1 = k+1;
					double ru1 = c3c4*rhoI[position4D(i,j,k,c)];
					cv[k1] = ws[position4D(i,j,k,c)];
					rhos[k1] = dmax1(dz4+con43*ru1,dz5+c1c5*ru1,dzmax+ru1,dz1); 
				}

				for(int k = start[2][c]+1; k<=cellSize[2][c]-end[2][c]; k++){
					int k1 = k+1;
					lhs[lhsPosition5D(i1,j1,k1,0,c)] = 0;
					lhs[lhsPosition5D(i1,j1,k1,1,c)] = -dttz2*cv[k] - dttz1*rhos[k];
					lhs[lhsPosition5D(i1,j1,k1,2,c)] = 1 + c2dttz1*rhos[k1];
					lhs[lhsPosition5D(i1,j1,k1,3,c)] = dttz2*cv[k1+1] - dttz1*rhos[k1+1];
					lhs[lhsPosition5D(i1,j1,k1,4,c)] = 0;
				}
			}
		}

		/* Add fourth order dissipation */
		if(start[2][c] > 0){
			int k = 3;
			for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
				for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
					lhs[lhsPosition5D(i,j,k,2,c)] +=  comz5;
					lhs[lhsPosition5D(i,j,k,3,c)] = lhs[lhsPosition5D(i,j,k,3,c)] - comz4;
					lhs[lhsPosition5D(i,j,k,4,c)] +=  comz1;

					lhs[lhsPosition5D(i,j,k+1,1,c)] = lhs[lhsPosition5D(i,j,k+1,1,c)] - comz4;
					lhs[lhsPosition5D(i,j,k+1,2,c)] +=  comz6;
					lhs[lhsPosition5D(i,j,k+1,3,c)] = lhs[lhsPosition5D(i,j,k+1,3,c)] - comz4;
					lhs[lhsPosition5D(i,j,k+1,4,c)] +=  comz1;
				}
			}
		}

		for(int k = 3*start[2][c]+2; k<=cellSize[2][c]-3*end[2][c]+1; k++){
			for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
				for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
					lhs[lhsPosition5D(i,j,k,0,c)] +=  comz1;
					lhs[lhsPosition5D(i,j,k,1,c)] = lhs[lhsPosition5D(i,j,k,1,c)] - comz4;
					lhs[lhsPosition5D(i,j,k,2,c)] +=  comz6;
					lhs[lhsPosition5D(i,j,k,3,c)] = lhs[lhsPosition5D(i,j,k,3,c)] - comz4;
					lhs[lhsPosition5D(i,j,k,4,c)] +=  comz1;
				}
			}
		}

		if(end[2][c] > 0){
			int k = cellSize[2][c]-1;
			for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
				for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
					lhs[lhsPosition5D(i,j,k,0,c)] +=  comz1;
					lhs[lhsPosition5D(i,j,k,1,c)] = lhs[lhsPosition5D(i,j,k,1,c)] - comz4;
					lhs[lhsPosition5D(i,j,k,2,c)] +=  comz6;
					lhs[lhsPosition5D(i,j,k,3,c)] = lhs[lhsPosition5D(i,j,k,3,c)] - comz4;

					lhs[lhsPosition5D(i,j,k+1,0,c)] +=  comz1;
					lhs[lhsPosition5D(i,j,k+1,1,c)] = lhs[lhsPosition5D(i,j,k+1,1,c)] - comz4;
					lhs[lhsPosition5D(i,j,k+1,2,c)] +=  comz5;
				}
			}
		}

		/*
		 * Subsequently, fill the other factors (u+c), (u-c) by a4ing
		 * to the first
		 */
		for(int k = start[2][c]+1; k<=cellSize[2][c]-end[2][c]; k++){
			int k1 = k+1;
			for(int j = start[1][c]+1; j<=cellSize[1][c]-end[1][c]; j++){
				int j1 = j+1;
				for(int i = start[0][c]+1; i<=cellSize[0][c]-end[0][c]; i++){
					int i1 = i+1;
					lhs[lhsPosition5D(i1,j1,k1,5,c)] = lhs[lhsPosition5D(i1,j1,k1,0,c)];
					lhs[lhsPosition5D(i1,j1,k1,6,c)] = lhs[lhsPosition5D(i1,j1,k1,1,c)] - dttz2*speed[position4D(i,j,k-1,c)];
					lhs[lhsPosition5D(i1,j1,k1,7,c)] = lhs[lhsPosition5D(i1,j1,k1,2,c)];
					lhs[lhsPosition5D(i1,j1,k1,8,c)] = lhs[lhsPosition5D(i1,j1,k1,3,c)] + dttz2*speed[position4D(i,j,k+1,c)];
					lhs[lhsPosition5D(i1,j1,k1,9,c)] = lhs[lhsPosition5D(i1,j1,k1,4,c)];

					lhs[lhsPosition5D(i1,j1,k1,10,c)] = lhs[lhsPosition5D(i1,j1,k1,0,c)];
					lhs[lhsPosition5D(i1,j1,k1,11,c)] = lhs[lhsPosition5D(i1,j1,k1,1,c)] + dttz2*speed[position4D(i,j,k-1,c)];
					lhs[lhsPosition5D(i1,j1,k1,12,c)] = lhs[lhsPosition5D(i1,j1,k1,2,c)];
					lhs[lhsPosition5D(i1,j1,k1,13,c)] = lhs[lhsPosition5D(i1,j1,k1,3,c)] - dttz2*speed[position4D(i,j,k+1,c)];
					lhs[lhsPosition5D(i1,j1,k1,14,c)] = lhs[lhsPosition5D(i1,j1,k1,4,c)];
				}
			}
		}

	}

	/*
	 * Block-diagonal matrix-vector multiplication
	 */
	private void tzetar(int c){

		double r1, r2, r3, r4, r5, t1, t2, t3, xvel, yvel, zvel;
		double ac, acinv, ac2u, uzik1, btuz;

		/* Treat only one cell */
		for(int k = start[2][c]+2; k<=cellSize[2][c]-end[2][c]+1; k++){
			int k1 = k-1;
			for(int j = start[1][c]+2; j<=cellSize[1][c]-end[1][c]+1; j++){
				int j1 = j-1;
				for(int i = start[0][c]+2; i<=cellSize[0][c]-end[0][c]+1; i++){
					int i1 = i-1;

					xvel = us[position4D(i1,j1,k1,c)];
					yvel = vs[position4D(i1,j1,k1,c)];
					zvel = ws[position4D(i1,j1,k1,c)];
					ac = speed[position4D(i1,j1,k1,c)];
					acinv = ainv[position4D(i1,j1,k1,c)];

					ac2u = ac*ac;

					r1 = rhs[position5D(i,j,k,0,c)];
					r2 = rhs[position5D(i,j,k,1,c)];
					r3 = rhs[position5D(i,j,k,2,c)];
					r4 = rhs[position5D(i,j,k,3,c)];
					r5 = rhs[position5D(i,j,k,4,c)];

					uzik1 = u[position5D(i,j,k,0,c)];
					btuz = bt*uzik1;

					/* As in the original: multiply by the stored reciprocal 1/ac; btuz/ac*(r4+r5) rounds differently. */
					t1 = btuz*acinv*(r4+r5);
					t2 = r3+t1;
					t3 = btuz*(r4-r5);

					rhs[position5D(i,j,k,0,c)] = t2;
					rhs[position5D(i,j,k,1,c)] = -uzik1*r2 + xvel*t2;
					rhs[position5D(i,j,k,2,c)] =  uzik1*r1 + yvel*t2;
					rhs[position5D(i,j,k,3,c)] = zvel*t2 + t3;
					rhs[position5D(i,j,k,4,c)] = uzik1*(-xvel*r2 + yvel*r1) + qs[position4D(i1,j1,k1,c)]*t2 + c2iv*ac2u*t1 + zvel*t3;
				}
			}
		}

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

		copyFaces();

		rhsNorm(xcr);

		for(int m = 0; m<5; m++){
			xcr[m] = xcr[m]/dt;
		}

		if(myRank != 0)
			return verified;

		CLASS = 'U';

		verified = 1;

		for(int m = 0; m<5; m++){
			xcrref[m] = 1.0;
			xceref[m] = 1.0;
		}

		/* reference data for 12X12X12 grids after 100 time steps, with DT = 1.50d-02 */
		if( gridPoints[0] == 12 &&
				gridPoints[1] == 12 &&
				gridPoints[2] == 12 &&
				noTimeSteps == 100 ){
			CLASS = 'S';
			dtref = 1.5E-2;

			/* Reference values of RMS-norms of residual */
			xcrref[0] = 2.7470315451339479E-2;
			xcrref[1] = 1.0360746705285417E-2;
			xcrref[2] = 1.6235745065095532E-2;
			xcrref[3] = 1.5840557224455615E-2;
			xcrref[4] = 3.4849040609362460E-2;

			/* Reference values of RMS-norms of solution error */
			xceref[0] = 2.7289258557377227E-5;
			xceref[1] = 1.0364446640837285E-5;
			xceref[2] = 1.6154798287166471E-5;
			xceref[3] = 1.5750704994480102E-5;
			xceref[4] = 3.4177666183390531E-5;

		}
		/* reference data for 36X36X36 grids after 400 time steps, with DT = 1.50d-03 */
		else if( gridPoints[0] == 36 &&
				gridPoints[1] == 36 &&
				gridPoints[2] == 36 &&
				noTimeSteps == 400 ){
			CLASS = 'W';
			dtref = 1.5E-3;

			/* Reference values of RMS-norms of residual */
			xcrref[0] = 0.1893253733584E-02;
			xcrref[1] = 0.1717075447775E-03;
			xcrref[2] = 0.2778153350936E-03;
			xcrref[3] = 0.2887475409984E-03;
			xcrref[4] = 0.3143611161242E-02;

			/* Reference values of RMS-norms of solution error */
			xceref[0] = 0.7542088599534E-04;
			xceref[1] = 0.6512852253086E-05;
			xceref[2] = 0.1049092285688E-04;
			xceref[3] = 0.1128838671535E-04;
			xceref[4] = 0.1212845639773E-03;

		}
		/* reference data for 64X64X64 grids after 400 time steps, with DT = 1.50d-03 */
		else if( gridPoints[0] == 64 &&
				gridPoints[1] == 64 &&
				gridPoints[2] == 64 &&
				noTimeSteps == 400 ){
			CLASS = 'A';
			dtref = 1.5E-3;

			/* Reference values of RMS-norms of residual */
			xcrref[0] = 2.4799822399300195e0;
			xcrref[1] = 1.1276337964368832e0;
			xcrref[2] = 1.5028977888770491e0;
			xcrref[3] = 1.4217816211695179e0;
			xcrref[4] = 2.1292113035138280e0;

			/* Reference values of RMS-norms of solution error */
			xceref[0] = 1.0900140297820550e-04;
			xceref[1] = 3.7343951769282091e-05;
			xceref[2] = 5.0092785406541633e-05;
			xceref[3] = 4.7671093939528255e-05;
			xceref[4] = 1.3621613399213001e-04;

		}
		/* reference data for 102X102X102 grids after 400 time steps, with DT = 1.0e-03 */
		else if( gridPoints[0] == 102 &&
				gridPoints[1] == 102 &&
				gridPoints[2] == 102 &&
				noTimeSteps == 400 ){
			CLASS = 'B';
			dtref = 1.0E-3;

			/* Reference values of RMS-norms of residual */
			xcrref[0] = 0.6903293579998e+02;
			xcrref[1] = 0.3095134488084e+02;
			xcrref[2] = 0.4103336647017e+02;
			xcrref[3] = 0.3864769009604e+02;
			xcrref[4] = 0.5643482272596e+02;

			/* Reference values of RMS-norms of solution error */
			xceref[0] = 0.9810006190188e-02;
			xceref[1] = 0.1022827905670e-02;
			xceref[2] = 0.1720597911692e-02;
			xceref[3] = 0.1694479428231e-02;
			xceref[4] = 0.1847456263981e-01;

		}
		/* reference data for 162X162X162 grids after 400 time steps, with DT = 0.67e-03 */
		else if( gridPoints[0] == 162 &&
				gridPoints[1] == 162 &&
				gridPoints[2] == 162 &&
				noTimeSteps == 400 ){
			CLASS = 'C';
			dtref = 0.67E-3;

			/* Reference values of RMS-norms of residual */
			xcrref[0] = 0.5881691581829e+03;
			xcrref[1] = 0.2454417603569e+03;
			xcrref[2] = 0.3293829191851e+03;
			xcrref[3] = 0.3081924971891e+03;
			xcrref[4] = 0.4597223799176e+03;

			/* Reference values of RMS-norms of solution error */
			xceref[0] = 0.2598120500183e+00;
			xceref[1] = 0.2590888922315e-01;
			xceref[2] = 0.5132886416320e-01;
			xceref[3] = 0.4806073419454e-01;
			xceref[4] = 0.5483377491301e+00;

		}
		/* reference data for 408X408X408 grids after 500 time steps, with DT = 0.3e-03 */
		else if( gridPoints[0] == 408 &&
				gridPoints[1] == 408 &&
				gridPoints[2] == 408 &&
				noTimeSteps == 500 ){
			CLASS = 'D';
			dtref = 0.3E-3;

			/* Reference values of RMS-norms of residual */
			xcrref[0] = 0.1044696216887e+05;
			xcrref[1] = 0.3204427762578e+04;
			xcrref[2] = 0.4648680733032e+04;
			xcrref[3] = 0.4238923283697e+04;
			xcrref[4] = 0.7588412036136e+04;

			/* Reference values of RMS-norms of solution error */
			xceref[0] = 0.5089471423669e+01;
			xceref[1] = 0.5323514855894e+00;
			xceref[2] = 0.1187051008971e+01;
			xceref[3] = 0.1083734951938e+01;
			xceref[4] = 0.1164108338568e+02;

		}
		/* reference data for 1020X1020X1020 grids after 500 time steps, with DT = 0.1e-03 */
		else if( gridPoints[0] == 1020 &&
				gridPoints[1] == 1020 &&
				gridPoints[2] == 1020 &&
				noTimeSteps == 500 ){
			CLASS = 'E';
			dtref = 0.1E-3;

			/* Reference values of RMS-norms of residual */
			xcrref[0] = 0.6255387422609e+05;
			xcrref[1] = 0.1495317020012e+05;
			xcrref[2] = 0.2347595750586e+05;
			xcrref[3] = 0.2091099783534e+05;
			xcrref[4] = 0.4770412841218e+05;

			/* Reference values of RMS-norms of solution error */
			xceref[0] = 0.6742735164909e+02;
			xceref[1] = 0.5390656036938e+01;
			xceref[2] = 0.1680647196477e+02;
			xceref[3] = 0.1536963126457e+02;
			xceref[4] = 0.1575330146156e+03;

		}
		else{
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
				System.out.println("\t"+m+" "+xcr[m]);
			}
			else if(xcrdif[m] <= epsilon){
				System.out.println("\t"+m+" "+xcr[m]+" "+xcrref[m]+" "+xcrdif[m]);
			}
			else{
				verified = 0;
				System.out.println("FAILURE: \t"+m+" "+xcr[m]+" "+xcrref[m]+" "+xcrdif[m]);
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
				System.out.println("\t"+m+" "+xce[m]);
			}
			else if(xcedif[m] <= epsilon){
				System.out.println("\t"+m+" "+xce[m]+" "+xceref[m]+" "+xcedif[m]);
			}
			else{
				verified = 0;
				System.out.println("FAILURE: \t"+m+" "+xce[m]+" "+xceref[m]+" "+xcedif[m]);
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

		double rmsWork[] = new double[5];
		double uExact[] = new double[5];

		for(int m = 0; m<5; m++){
			rmsWork[m] = 0.0;
		}

		for(int c = 0; c<ncells; c++){
			int kk = 2;
			for(int k = cellLow[2][c]; k<=cellHigh[2][c]; k++){
				double zeta = k*dnzm1;
				int jj = 2;
				for(int j = cellLow[1][c]; j<=cellHigh[1][c]; j++){
					double eta = j*dnym1;
					int ii = 2;
					for(int i = cellLow[0][c]; i<=cellHigh[0][c]; i++){
						double xi = i*dnxm1;
						exactSolution(xi,eta,zeta,uExact);
						for(int m = 0; m<5; m++){
							double add = u[position5D(ii,jj,kk,m,c)]-uExact[m];
							rmsWork[m] +=  add*add;
						}
						ii++;
					}
					jj++;
				}
				kk++;
			}
		}

		commWork.Allreduce(rmsWork, 0, rms, 0, 5, MPI.DOUBLE, MPI.SUM);

		for(int m = 0; m<5; m++){
			for(int d = 0; d<3; d++){
				rms[m] = rms[m]/(gridPoints[d]-2);
			}
			rms[m] = Math.sqrt(rms[m]);
		}

	}

	private void rhsNorm(double rms[]){

		double rmsWork[] = new double[5];

		for(int m = 0; m<5; m++){
			rmsWork[m] = 0.0;
		}

		for(int c = 0; c<ncells; c++){
			for(int k = start[2][c]+2; k<=cellSize[2][c]+2; k++){
				for(int j = start[1][c]+2; j<=cellSize[1][c]+2; j++){
					for(int i = start[0][c]+2; i<=cellSize[0][c]+2; i++){
						for(int m = 0; m<5; m++){
							double add = rhs[position5D(i,j,k,m,c)];
							rmsWork[m] +=  add*add;
						}
					}
				}
			}
		}

		commWork.Allreduce(rmsWork, 0, rms, 0, 5, MPI.DOUBLE, MPI.SUM);

		for(int m = 0; m<5; m++){
			for(int d = 0; d<3; d++){
				rms[m] = rms[m]/(gridPoints[d]-2);
			}
			rms[m] = Math.sqrt(rms[m]);
		}
	}

	public double dmax1(double a, double b){
		if(a<b) return b; else return a;
	}

	public double dmax1(double a, double b, double c, double d ){
		return dmax1(dmax1(a,b), dmax1(c,d) );
	}

	/*
	 * This methods organizes the memory in the same way that fortran does,
	 * to facilitate the code translation and efficiency.
	 */
	private int position5D(int i, int j, int k, int m, int c) {
		return i + j * DIMI + k * DIMI_DIMJ + m * DIMI_DIMJ_DIMK
		+ c * DIMI_DIMJ_DIMK_DIMM;
	}

	private int forcPosition5D(int i, int j, int k, int m, int c) {
		return i + j * DIMI_FORC + k * DIMI_DIMJ_FORC + m * DIMI_DIMJ_DIMK_FORC
		+ c * DIMI_DIMJ_DIMK_DIMM_FORC;
	}

	private int lhsPosition5D(int i, int j, int k, int m, int c) {
		return i + j * DIMI + k * DIMI_DIMJ + m * DIMI_DIMJ_DIMK
		+ c * DIMI_DIMJ_DIMK_DIMM_LHS;
	}

	private int position4D(int i, int j, int k, int m) {
		return i + j * DIMI_4D + k * DIMI_DIMJ_4D + m * DIMI_DIMJ_DIMK_4D;
	}	
}
