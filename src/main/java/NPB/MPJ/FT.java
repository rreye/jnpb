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

import NPB.Benchmark;
import NPB.common.io.BMArgs;
import NPB.common.io.BMResults;
import NPB.common.numerics.Misc;
import NPB.common.numerics.Random;
import NPB.common.timer.Timer;
import NPB.exceptions.BadClassException;
import NPB.exceptions.IncorrectNumProcessException;
import NPB.exceptions.NPBException;
import NPB.exceptions.BadInputException;


public class FT implements Benchmark{

	private static final String BM_NAME = "FT";
	private char CLASS;
	private boolean detailedTiming;
	private final boolean moreDetailedTiming = false;

	private final int LAYOUT_0D = 0, LAYOUT_1D = 1, LAYOUT_2D = 2;
	private int nx, ny, nz, maxDim, niter, layoutType, np1, np2, ntdivnp;
	private double ntotalF;

	private int dims[][] = new int[3][3];
	private int xstart[] = new int[3];
	private int ystart[] = new int[3];
	private int zstart[] = new int[3];
	private int xend[] = new int[3];
	private int yend[] = new int[3];
	private int zend[] = new int[3];
	private double u[];

	private double chk[] = new double[2];
	private double allchk[] = new double[2];
	private double aux[] = new double[2];
	
	private final int fftBlockDefault = 16;
	private final int fftBlockPadDefault = 18;
	private final int transBlock = 32;
	private final int transBlockPad = 34;
	private int fftBlock;
	private int fftBlockPad;

	/* Real and imaginary parts in complex vectors. */
	private final int REAL = 0;
	private final int IMAG = 1;
	private final double pi = 3.141592653589793238;
	private final double alpha = 1E-6;
	private final double seed = 314159265;
	private final double a = 1220703125;
	private final double ap = -4 * alpha * pi * pi;
	private double sums[];

	private Random rng;
	private Timer timer;
	private BMResults res;

	/*
	 * Different timers
	 */
	private final int tTotal = 0;
	private final int tSetup = 1;
	private final int tFft = 2;
	private final int tEvolve = 3;
	private final int tChecksum = 4;
	private final int tFftlow = 5;
	private final int tFftcopy = 6;
	private final int tTranspose = 7;
	private final int tTransxzloc = 8;
	private final int tTransxzglo = 9;
	private final int tTransxzfin = 10;
	private final int tTransxyloc = 11;
	private final int tTransxyglo = 12;
	private final int tTransxyfin = 13;
	private final int tSynch = 14;
	private final int T_COMM = 15;

	/*
	 * MPJ variables
	 */
	private int myRank;
	private int numProcess; /* active processes (a power of two / a square) */
	private int npTotal; /* processes launched */
	private boolean active = true; /* false for the excess processes masked out */
	private Intracomm commWork; /* communicator of the active processes */
	private int myRank1, myRank2;

	private Intracomm commSlice1;
	private Intracomm commSlice2;

	public static void main(String[] args) throws NPBException {
		FT ft = null;

		BMArgs.parseCmdLineArgs(args, BM_NAME);
		try {
			ft = new FT(BMArgs.CLASS, BMArgs.DETAILED_TIMING);
		} catch (OutOfMemoryError e) {
			BMArgs.outOfMemoryMessage();
			throw e;
		}
		ft.runBenchmark(args);
	}

	public BMResults getResults(){
		return res;
	}
	
	public FT(char classSize, boolean dtiming){

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

		switch (CLASS) {
		case 'S':
			nx = 64;
			ny = 64;
			nz = 64;
			maxDim = Math.max(Math.max(nx, ny), nz);
			niter = 6;
			break;
		case 'W':
			nx = 128;
			ny = 128;
			nz = 32;
			maxDim = Math.max(Math.max(nx, ny), nz);
			niter = 6;
			break;
		case 'A':
			nx = 256;
			ny = 256;
			nz = 128;
			maxDim = Math.max(Math.max(nx, ny), nz);
			niter = 6;
			break;
		case 'B':
			nx = 512;
			ny = 256;
			nz = 256;
			maxDim = Math.max(Math.max(nx, ny), nz);
			niter = 20;
			break;
		case 'C':
			nx = 512;
			ny = 512;
			nz = 512;
			maxDim = Math.max(Math.max(nx, ny), nz);
			niter = 20;
			break;
		case 'D':
			nx = 2048;
			ny = 1024;
			nz = 1024;
			maxDim = Math.max(Math.max(nx, ny), nz);
			niter = 25;
			break;
		case 'E':
			nx = 4096;
			ny = 2048;
			nz = 2048;
			maxDim = Math.max(Math.max(nx, ny), nz);
			niter = 25;
			break;
		default:
			terminate();
			throw new BadClassException("Specified class (" + CLASS + ") not valid for this benchmark\n"
					+ "Valid classes are S, W, A, B, C, D and E");
		}

		/*
		 * Per-process partition (as in NPB-MPI: ntdivnp = ((nx*ny)/np)*nz). Computed in long
		 * so that a too small number of processes is reported clearly instead of overflowing
		 * the int (class D has 2^31 points: it needs more than one process; Java arrays are
		 * limited to 2^31-1 elements: two interleaved doubles per point).
		 */
		long ntdivnpLong = ((long) nx * ny / numProcess) * nz;
		if (2L * ntdivnpLong > Integer.MAX_VALUE) {
			terminate();
			throw new IncorrectNumProcessException("Class " + CLASS + " needs more processes: "
					+ "the partition of " + ntdivnpLong + " points per process is too large for a Java array");
		}
		ntdivnp = (int) ntdivnpLong;
		u = new double[2 * nx];
		ntotalF = (double) nx * ny * nz; /* as ntotal_f in NPB-MPI: nx*ny*nz overflows int for class D */
		sums = new double[2 * niter];
		rng = new Random();
	}

	public void runBenchmark(String[] args) throws NPBException {
		
		this.build(args);

		/* excess process masked out (NPB_NPROCS_STRICT off): nothing to do */
		if (!active) {
			terminate();
			return;
		}
		
		timer = new Timer();

		double twiddle[] = new double[ntdivnp];
		double u0[] = new double[2 * ntdivnp];
		double u1[] = new double[2 * ntdivnp];
		double u2[] = new double[2 * ntdivnp];

		setup();
		computeIndexmap(twiddle, dims[0][2], dims[1][2], dims[2][2]);
		computeInitialConditions(u1, dims[0][0], dims[1][0], dims[2][0]);

		fftInit(dims[0][0]);
		fft(1, u1, u0);

		/*
		 * Start over from the beginning. Note that all operations must be
		 * timed, in contrast to other benchmarks.
		 */
		timer.resetAll();
		commWork.Barrier();

		timer.start(tTotal);
		if (moreDetailedTiming) timer.start(tSetup);

		computeIndexmap(twiddle, dims[0][2], dims[1][2], dims[2][2]);
		computeInitialConditions(u1, dims[0][0], dims[1][0], dims[2][0]);
		fftInit(dims[0][0]);

		if (moreDetailedTiming) {
			synchup();
			timer.stop(tSetup);
			timer.start(tFft);
		}

		fft(1, u1, u0);

		if (moreDetailedTiming) timer.stop(tFft);

		for (int iter = 1; iter <= niter; iter++) {
			
			if (moreDetailedTiming) timer.start(tEvolve);
			evolve(u0, u1, twiddle, dims[0][0], dims[1][0], dims[2][0]);
			if (moreDetailedTiming) {
				timer.stop(tEvolve);
				timer.start(tFft);
			}
			fft(-1, u1, u2);
			if (moreDetailedTiming) {
				timer.stop(tFft);
				synchup();
				timer.start(tChecksum);
			}
			checksum(iter, u2, dims[0][0], dims[1][0], dims[2][0]);
			if (moreDetailedTiming) timer.stop(tChecksum);			
		}

		boolean verified = false;

		verified = verify(nx, ny, nz, niter, CLASS);
		timer.stop(tTotal);

		double times[] = new double[2];
		times[1] = timer.read(T_COMM);

		commWork.Reduce(times, 0, times, 0, 2, MPI.DOUBLE, MPI.MAX, 0);

		if (myRank == 0) {
			double totalTime = timer.read(tTotal);
			double MFLOPS;

			if (totalTime != 0)
				MFLOPS = 1E-6 * ntotalF *
				(14.8157 + 7.19641 * Math.log(ntotalF) +
						(5.23518 + 7.21113 * Math.log(ntotalF))*niter) / totalTime;
			else
				MFLOPS = 0;
			
			int nVerified = 0;
			if (verified)
				nVerified = 1;
			
			BMResults.printVerificationStatus(CLASS, nVerified);

			times[0] = totalTime;

			res = new BMResults(BM_NAME, CLASS, nx, ny, nz, niter,
					times, MFLOPS, "floating point", nVerified, npTotal, numProcess,
					detailedTiming);
			
			res.printMPJ();

			if (moreDetailedTiming)
				printTimers();
		}
		terminate();
	}

	private void terminate() {
		MPI.Finalize();
	}

	private void setup() {

		boolean inData = true;

		if (myRank == 0) {
			
			BMArgs.bannerMPJ(BM_NAME, CLASS, npTotal);

			try
			{
				BufferedReader input = new BufferedReader(new FileReader("inputft.data"));

				System.out.println(" Reading from input file inputft.data");

				String line[] = input.readLine().split(" ");
				niter = Integer.valueOf(line[0]);

				line = input.readLine().split(" ");
				layoutType = Integer.valueOf(line[0]);

				line = input.readLine().split(" ");
				np1 = Integer.valueOf(line[0]);
				np2 = Integer.valueOf(line[1]);

				input.close();

				/*
				 * Check to make sure input data is consistent
				 */

				/*
				 * Product of processors grid dims must equal number of
				 * processors
				 */
				if (np1 * np2 != numProcess) {
					System.err.println(" np1 and np2 given in input file are not valid.");
					System.err.println(" Product is " + (np1 * np2) + " and should be " + numProcess);
					commWork.Abort(-2);
				}

				/* Layout type must be valid */
				if (layoutType != LAYOUT_0D && layoutType != LAYOUT_1D && layoutType != LAYOUT_2D) {
					System.err.println(" Layout type specified in input file is not valid");
					commWork.Abort(-3);
				}

				/* 0D layout must be 1x1 grid */
				if (layoutType == LAYOUT_0D && (np1 != 1 || np2 != 1)) {
					System.err.println(" For 0D layout, both np1 and np2 must be 1");
					commWork.Abort(-4);
				}

				/* 1D layout must be 1xN grid */
				if (layoutType == LAYOUT_1D && np1 != 1) {
					System.err.println(" For 1D layout, np1 must be 1");
					commWork.Abort(-5);
				}

			}
			catch (FileNotFoundException e) {
				inData = false;
			}
			catch (IOException e) {
				System.err.println(" Error reading input file: "+ e.getMessage());
				commWork.Abort(-7);
			}
			if (!inData) {
				System.out.println(" No input file inputft.data. Using compiled defaults");
				if (numProcess == 1) {
					np1 = np2 = 1;
					layoutType = LAYOUT_0D;
				} else if (numProcess <= nz) {
					np1 = 1;
					np2 = numProcess;
					layoutType = LAYOUT_1D;
				} else {
					np1 = nz;
					np2 = numProcess / nz;
					layoutType = LAYOUT_2D;
				}
			}

			System.out.println(" Size                :  " + nx + "x  " + ny + "x  " + nz);
			System.out.println(" Iterations          :  " + niter);
			System.out.println(" Number of processes :  " + npTotal);
			if (numProcess != npTotal)
				System.out.println(" WARNING: Number of processes is not power of two (" + numProcess + " active)");
			System.out.println(" Processor array     :  " + np1 + "x" + np2);
			if (layoutType == LAYOUT_0D)
				System.out.println(" Layout type         :  0D");
			else if (layoutType == LAYOUT_1D)
				System.out.println(" Layout type         :  1D");
			else if (layoutType == LAYOUT_2D)
				System.out.println(" Layout type         :  2D");
		}

		int buff[] = { np1, np2, layoutType, niter };
		commWork.Bcast(buff, 0, 4, MPI.INT, 0);
		np1 = buff[0];
		np2 = buff[1];
		layoutType = buff[2];
		niter = buff[3];

		if (layoutType == LAYOUT_0D) {
			for (int i = 0; i < 3; i++) {
				dims[0][i] = nx;
				dims[1][i] = ny;
				dims[2][i] = nz;
			}
		} else if (layoutType == LAYOUT_1D) {

			dims[0][0] = nx;
			dims[1][0] = ny;
			dims[2][0] = nz;

			dims[0][1] = nx;
			dims[1][1] = ny;
			dims[2][1] = nz;

			dims[0][2] = nz;
			dims[1][2] = nx;
			dims[2][2] = ny;

		} else if (layoutType == LAYOUT_2D) {

			dims[0][0] = nx;
			dims[1][0] = ny;
			dims[2][0] = nz;

			dims[0][1] = ny;
			dims[1][1] = nx;
			dims[2][1] = nz;

			dims[0][2] = nz;
			dims[1][2] = nx;
			dims[2][2] = ny;
		}

		for (int i = 0; i < 3; i++) {
			dims[1][i] = dims[1][i] / np1;
			dims[2][i] = dims[2][i] / np2;
		}

		/*
		 * Determine processor coordinates of this processor Processor grid is
		 * np1xnp2 Arrays are always (n1, n2/np1, n3/np2) Processor coords are
		 * zero-based
		 */
		myRank2 = myRank % np2;
		myRank1 = myRank / np2;

		/*
		 * Communicators for rows/columns of processor grid. commSlice1 is
		 * communicator of all procs with same myRank1, ranked as myRank2
		 * commSlice2 is communicator of all procs with same myRank2, ranked as
		 * myRank1
		 */
		commSlice1 = commWork.Split(myRank1, myRank2);
		commSlice2 = commWork.Split(myRank2, myRank1);

		if (moreDetailedTiming) synchup();

		/* Determine which section of the grid is owned by this processor */
		if (layoutType == LAYOUT_0D) {
			for (int i = 0; i < 3; i++) {
				xstart[i] = 1;
				xend[i] = nx;
				ystart[i] = 1;
				yend[i] = ny;
				zstart[i] = 1;
				zend[i] = nz;
			}
		} else if (layoutType == LAYOUT_1D) {

			xstart[0] = 1;
			xend[0] = nx;
			ystart[0] = 1;
			yend[0] = ny;
			zstart[0] = 1 + myRank2 * nz / np2;
			zend[0] = (myRank2 + 1) * nz / np2;

			xstart[1] = 1;
			xend[1] = nx;
			ystart[1] = 1;
			yend[1] = ny;
			zstart[1] = 1 + myRank2 * nz / np2;
			zend[1] = (myRank2 + 1) * nz / np2;

			xstart[2] = 1;
			xend[2] = nx;
			ystart[2] = 1 + myRank2 * ny / np2;
			yend[2] = (myRank2 + 1) * ny / np2;
			zstart[2] = 1;
			zend[2] = nz;

		} else if (layoutType == LAYOUT_2D) {

			xstart[0] = 1;
			xend[0] = nx;
			ystart[0] = 1 + myRank1 * ny / np1;
			yend[0] = (myRank1 + 1) * ny / np1;
			zstart[0] = 1 + myRank2 * nz / np2;
			zend[0] = (myRank2 + 1) * nz / np2;

			xstart[1] = 1 + myRank1 * nx / np1;
			xend[1] = (myRank1 + 1) * nx / np1;
			ystart[1] = 1;
			yend[1] = ny;
			zstart[1] = zstart[0];
			zend[1] = zend[0];

			xstart[2] = xstart[1];
			xend[2] = xend[1];
			ystart[2] = 1 + myRank2 * ny / np2;
			yend[2] = (myRank2 + 1) * ny / np2;
			zstart[2] = 1;
			zend[2] = nz;

		}

		/*
		 * Set up info for blocking of ffts and transposes. This improves
		 * performance on cache-based systems. Blocking involves working on a
		 * chunk of the problem at a time, taking chunks along the first,
		 * second, or third dimension.
		 *  - In cffts1 blocking is on 2nd dimension (with fft on 1st dim) - In
		 * cffts2/3 blocking is on 1st dimension (with fft on 2nd and 3rd dims)
		 * 
		 * Since 1st dim is always in processor, we'll assume it's long enough
		 * (default blocking factor is 16 so min size for 1st dim is 16) The
		 * only case we have to worry about is cffts1 in a 2d decomposition. so
		 * the blocking factor should not be larger than the 2nd dimension.
		 */
		fftBlock = fftBlockDefault;
		fftBlockPad = fftBlockPadDefault;

		if (layoutType == LAYOUT_2D) {
			if (dims[1][0] < fftBlock)
				fftBlock = dims[1][0];
			if (dims[1][1] < fftBlock)
				fftBlock = dims[1][1];
			if (dims[1][2] < fftBlock)
				fftBlock = dims[1][2];
		}

		if (fftBlock != fftBlockDefault)
			fftBlockPad = fftBlock + 3;

	}

	private void synchup() {
		timer.start(tSynch);
		if (detailedTiming) timer.start(T_COMM);
		commWork.Barrier();
		if (detailedTiming) timer.stop(T_COMM);
		timer.stop(tSynch);
	}

	/*
	 * Compute method from local (i,j,k) to ibar^2+jbar^2+kbar^2 for time
	 * evolution exponent.
	 * 
	 * This method is very different depending on whether we are in the
	 * 0D, 1D or 2D layout. Compute separately. Basically we want to convert
	 * the indices 1 2 3 4 5 6 7 8 to 0 1 2 3 -4 -3 -2 -1 The following
	 * magic formula does the trick: ((i-1+n/2) % n) - n/2
	 */
	private void computeIndexmap(double twiddle[], int d1, int d2, int d3) throws NPBException {

		int i, ii, ii2, j, jj, ij2, k, kk;

		if (layoutType == LAYOUT_0D) {
			for (i = 1; i <= dims[0][2]; i++) {
				ii = ((i + xstart[2] - 2 + (nx >> 1)) % nx) - (nx >> 1);
				ii2 = ii * ii;
				
				for (j = 1; j <= dims[1][2]; j++) {
					jj = ((j + ystart[2] - 2 + (ny >> 1)) % ny) - (ny >> 1);
					ij2 = jj * jj + ii2;
					
					for (k = 1; k <= dims[2][2]; k++) {
						kk = ((k + zstart[2] - 2 + (nz >> 1)) % nz) - (nz >> 1);
						twiddle[position3D(i, j, k, d1, d2, d3) >> 1] = Math.exp(ap * (kk * kk + ij2));
					}
				}
			}
		} else if (layoutType == LAYOUT_1D || layoutType == LAYOUT_2D) {
			for (i = 1; i <= dims[1][2]; i++) {
				ii = ((i + xstart[2] - 2 + (nx >> 1)) % nx) - (nx >> 1);
				ii2 = ii * ii;
				
				for (j = 1; j <= dims[2][2]; j++) {
					jj = ((j + ystart[2] - 2 + (ny >> 1)) % ny) - (ny >> 1);
					ij2 = jj * jj + ii2;
					
					for (k = 1; k <= dims[0][2]; k++) {
						kk = ((k + zstart[2] - 2 + (nz >> 1)) % nz) - (nz >> 1);
						twiddle[position3D(k, i, j, d1, d2, d3) >> 1] = Math.exp(ap * (kk * kk + ij2));
					}
				}
			}
		} else {
			throw new BadInputException("Unknown layout type");
		}
	}

	/*
	 * This method organizes the memory in the same way that fortran does,
	 * to facilitate the code translation and efficiency.
	 */
	private int position3D(int x, int y, int z, int dimx, int dimy, int dimz) {
		return (x - 1 + (y - 1) * dimx + (z - 1) * dimx * dimy) << 1;
	}

	/*
	 * Fill in array u0 with initial conditions from random number generator
	 * 
	 * 0D and 1D layouts are easy because each processor gets a contiguous
	 * chunk of the array, in the Fortran ordering sense (and in Java using
	 * the position method). For 2D layout, it's a bit more complicated. We
	 * always have entire x-lines (contiguous) in processor. We can do
	 * ny/np1 of them at a time since we have ny/np1 contiguous in
	 * y-direction. But then we jump by z-planes (nz/np2 of them, total).
	 * For the 0D and 1D layouts we could do larger chunks, but this turns
	 * out to have no measurable impact on performance.
	 */
	private void computeInitialConditions(double u0[], int d1, int d2, int d3) {

		double x0, start;
		int k;

		start = seed;

		/* Jump to the starting element for our first plane. */
		double an = ipow46(a, 2 * nx, (zstart[0] - 1) * ny + (ystart[0] - 1));
		double values[] = rng.vrandlc(start, an);
		start = values[0];
		an = ipow46(a, 2 * nx, ny);

		/* Go through by z planes filling in one square at a time */
		for (k = 1; k <= dims[2][0]; k++) {
			x0 = start;
			x0 = rng.complexvranlc((double) (nx * dims[1][0]), x0, a, u0, position3D(1, 1, k, d1, d2, d3) >> 1);
			if (k != dims[2][0]) {
				values = rng.vrandlc(start, an);
				start = values[0];
			}
		}
	}

	/*
	 * Compute a^exponent mod 2^46
	 * 
	 * Use: a^n = a^(n/2)*a^(n/2) if n even else a^n = a*a^(n-1) if n odd
	 */
	private double ipow46(double a, int exp1, int exp2) {	
		double result = 1;
		int n2;
		double values[];

		if (exp2 == 0 || exp1 == 0) {
			return result;
		}

		double q = a;
		double r = 1;
		int n = exp1;
		boolean twoPow = true;

		while (twoPow) {
			n2 = n >> 1;
			if (n2 << 1 == n) {
				values = rng.vrandlc(q, q);
				q = values[0];
				n = n2;
			} else {
				n = n * exp2;
				twoPow = false;
			}
		}

		while (n > 1) {
			n2 = n >> 1;
			if (n2 << 1 == n) {
				values = rng.vrandlc(q, q);
				q = values[0];
				n = n2;
			} else {
				values = rng.vrandlc(r, q);
				r = values[0];
				n--;
			}
		}

		values = rng.vrandlc(r, q);
		result = values[0];

		return result;
	}

	/*
	 * Compute the roots-of-unity array that will be used for subsequent
	 * FFTs
	 * 
	 * Initialize the "u" array with sines and cosines in a manner that
	 * permits stride one access at each FFT iteration.
	 */
	private void fftInit(int n) {
		double t, ti;
		int m = ilog2(n);
		u[0] = m;
		int ku = 1;
		int ln = 1;
		int i,j;
		
		for (j = 0; j < m; j++) {
			t = pi / ln;

			for (i = 0; i < ln; i++) {
				ti = i * t;
				u[(i + ku) << 1] = Math.cos(ti);
				u[IMAG + ((i + ku) << 1)] = Math.sin(ti);
			}
			ku += ln;
			ln = ln << 1;
		}
	}

	/*
	 * This method is similar to the method of the same name in common.Misc
	 * class. However this method exist to provide exactly the same
	 * behaviour than the function ilog2 in MPI version of FT.
	 */
	private int ilog2(int n) {
		if (n == 1) {
			return 0;
		}
		int lg = 1;
		int nn = 2;
		while (nn < n) {
			nn = nn << 1;
			lg++;
		}
		return lg;
	}

	/*
	 * note: args x1, x2 must be different complex arrays note: args for
	 * cfftsx are (direction, layout, xin, xout, scratch) xin/xout may be
	 * the same and it can be somewhat faster if they are note: args for
	 * transpose are (layout1, layout2, xin, xout) xin/xout must be
	 * different
	 */
	private void fft(int dir, double x1[], double x2[]) throws BadInputException {

		double scratch[] = new double[2 * (Math.min(256, Math.max(fftBlockDefault, 65536 / numProcess / Math.min(nx, Math.min(ny, nz)))) + 2) * maxDim * 2];

		if (dir == 1) {
			if (layoutType == LAYOUT_0D) {
				cffts1(1, dims[0][0], dims[1][0], dims[2][0], x1, x1, scratch);
				cffts2(1, dims[0][1], dims[1][1], dims[2][1], x1, x1, scratch);
				cffts3(1, dims[0][2], dims[1][2], dims[2][2], x1, x2, scratch);
			} else if (layoutType == LAYOUT_1D) {
				cffts1(1, dims[0][0], dims[1][0], dims[2][0], x1, x1, scratch);
				cffts2(1, dims[0][1], dims[1][1], dims[2][1], x1, x1, scratch);
				if (moreDetailedTiming) timer.start(tTranspose);
				transpose_xy_z(1, 2, x1, x2);
				if (moreDetailedTiming) timer.stop(tTranspose);
				cffts1(1, dims[0][2], dims[1][2], dims[2][2], x2, x2, scratch);
			} else if (layoutType == LAYOUT_2D) {
				cffts1(1, dims[0][0], dims[1][0], dims[2][0], x1, x1, scratch);
				if (moreDetailedTiming) timer.start(tTranspose);
				transposeXY(0, 1, x1, x2);
				if (moreDetailedTiming) timer.stop(tTranspose);
				cffts1(1, dims[0][1], dims[1][1], dims[2][1], x2, x2, scratch);
				if (moreDetailedTiming) timer.start(tTranspose);
				transposeXZ(1, 2, x2, x1);
				if (moreDetailedTiming) timer.stop(tTranspose);
				cffts1(1, dims[0][2], dims[1][2], dims[2][2], x1, x2, scratch);
			}
		} else {
			if (layoutType == LAYOUT_0D) {
				cffts3(-1, dims[0][2], dims[1][2], dims[2][2], x1, x1, scratch);
				cffts2(-1, dims[0][1], dims[1][1], dims[2][1], x1, x1, scratch);
				cffts1(-1, dims[0][0], dims[1][0], dims[2][0], x1, x2, scratch);
			} else if (layoutType == LAYOUT_1D) {
				cffts1(-1, dims[0][2], dims[1][2], dims[2][2], x1, x1, scratch);
				if (moreDetailedTiming) timer.start(tTranspose);
				transpose_x_yz(2, 1, x1, x2);
				if (moreDetailedTiming) timer.stop(tTranspose);
				cffts2(-1, dims[0][1], dims[1][1], dims[2][1], x2, x2, scratch);
				cffts1(-1, dims[0][0], dims[1][0], dims[2][0], x2, x2, scratch);
			} else if (layoutType == LAYOUT_2D) {
				cffts1(-1, dims[0][2], dims[1][2], dims[2][2], x1, x1, scratch);
				if (moreDetailedTiming) timer.start(tTranspose);
				transposeXZ(2, 1, x1, x2);
				if (moreDetailedTiming) timer.stop(tTranspose);
				cffts1(-1, dims[0][1], dims[1][1], dims[2][1], x2, x2, scratch);
				if (moreDetailedTiming) timer.start(tTranspose);
				transposeXY(1, 0, x2, x1);
				if (moreDetailedTiming) timer.stop(tTranspose);
				cffts1(-1, dims[0][0], dims[1][0], dims[2][0], x1, x2, scratch);
			}
		}
	}

	/*
	 * Number of FFTs of length n done at a time (the dimension that is blocked has nblk elements). Larger blocks than
	 * the default 16 are faster as long as the scratch block (fftBlock * n complex numbers, twice) stays in the cache
	 * of this process; the processes share the cache, so the budget is divided among them (with many processes this
	 * gives the default again). The block must not exceed (and, being a power of two, it then divides) nblk.
	 */
	private void setFftBlock(int n, int nblk) {
		fftBlock = Math.min(Math.min(256, Math.max(fftBlockDefault, 65536 / numProcess / n)), nblk);
		fftBlockPad = fftBlock + 2;
	}

	private void cffts1(int is, int d1, int d2, int d3, double x[], double xout[], double y[]) throws BadInputException {
		
		int i, k, jj, j, posx, posy, posxout;	
		int logd1 = ilog2(d1);
		setFftBlock(d1, d2);

		for (k = 1; k <= d3; k++) {
			for (jj = 0; jj <= d2 - fftBlock; jj += fftBlock) {
				if (moreDetailedTiming) timer.start(tFftcopy);
				for (j = 1; j <= fftBlock; j++) {
					for (i = 1; i <= d1; i++) {
						posy = position3D(j, i, 1, fftBlockPad, d1, 2);
						posx = position3D(i, j + jj, k, d1, d2, d3);
						y[posy] = x[posx];
						y[IMAG + posy] = x[IMAG + posx];
					}
				}

				if (moreDetailedTiming) {
					timer.stop(tFftcopy);
					timer.start(tFftlow);
				}
				final int ro = cfftz(is, logd1, d1, y, y, position3D(1, 1, 2, fftBlockPad, d1, 2));
				if (moreDetailedTiming) {
					timer.stop(tFftlow);
					timer.start(tFftcopy);
				}
				for (j = 1; j <= fftBlock; j++) {
					for (i = 1; i <= d1; i++) {
						posxout = position3D(i, j + jj, k, d1, d2, d3);
						posy = position3D(j, i, 1, fftBlockPad, d1, 2);
						xout[posxout] = y[ro + posy];
						xout[IMAG + posxout] = y[ro + IMAG + posy];
					}
				}
				if (moreDetailedTiming) timer.stop(tFftcopy);
			}
		}
	}

	private void cffts2(int is, int d1, int d2, int d3, double x[], double xout[], double y[]) throws BadInputException {
		
		int k, ii, j, i, posy, posx, posxout;
		int logd2 = ilog2(d2);
		setFftBlock(d2, d1);

		for (k = 1; k <= d3; k++) {
			for (ii = 0; ii <= d1 - fftBlock; ii += fftBlock) {
				if (moreDetailedTiming) timer.start(tFftcopy);
				for (j = 1; j <= d2; j++) {
					for (i = 1; i <= fftBlock; i++) {
						posy = position3D(i, j, 1, fftBlockPad, d2, 2);
						posx = position3D(i + ii, j, k, d1, d2, d3);
						y[posy] = x[posx];
						y[IMAG + posy] = x[IMAG + posx];
					}
				}
				if (moreDetailedTiming) {
					timer.stop(tFftcopy);
					timer.start(tFftlow);
				}
				final int ro = cfftz(is, logd2, d2, y, y, position3D(1, 1, 2, fftBlockPad, d2, 2));
				if (moreDetailedTiming) {
					timer.stop(tFftlow);
					timer.start(tFftcopy);
				}
				for (j = 1; j <= d2; j++) {
					for (i = 1; i <= fftBlock; i++) {
						posxout = position3D(i + ii, j, k, d1, d2, d3);
						posy = position3D(i, j, 1, fftBlockPad, d2, 2);
						xout[posxout] = y[ro + posy];
						xout[IMAG + posxout] = y[ro + IMAG + posy];
					}
				}
				if (moreDetailedTiming) timer.stop(tFftcopy);
			}
		}
	}

	private void cffts3(int is, int d1, int d2, int d3, double x[], double xout[], double y[]) throws BadInputException {
		
		int k, ii, j, i, posy, posx, posxout;
		int logd3 = ilog2(d3);
		setFftBlock(d3, d1);

		for (j = 1; j <= d2; j++) {
			for (ii = 0; ii <= d1 - fftBlock; ii += fftBlock) {
				if (moreDetailedTiming) timer.start(tFftcopy);
				for (k = 1; k <= d3; k++) {
					for (i = 1; i <= fftBlock; i++) {
						posy = position3D(i, k, 1, fftBlockPad, d3, 2);
						posx = position3D(i + ii, j, k, d1, d2, d3);
						y[posy] = x[posx];
						y[IMAG + posy] = x[IMAG + posx];
					}
				}
				if (moreDetailedTiming) {
					timer.stop(tFftcopy);
					timer.start(tFftlow);
				}
				final int ro = cfftz(is, logd3, d3, y, y, position3D(1, 1, 2, fftBlockPad, d3, 2));
				if (moreDetailedTiming) {
					timer.stop(tFftlow);
					timer.start(tFftcopy);
				}
				for (k = 1; k <= d3; k++) {
					for (i = 1; i <= fftBlock; i++) {
						posxout = position3D(i + ii, j, k, d1, d2, d3);
						posy = position3D(i, k, 1, fftBlockPad, d3, 2);
						xout[posxout] = y[ro + posy];
						xout[IMAG + posxout] = y[ro + IMAG + posy];
					}
				}
				if (moreDetailedTiming) timer.stop(tFftcopy);
			}
		}
	}

	private void transpose_xy_z(int l1, int l2, double xin[], double xout[]) {
		transpose2Local(dims[0][l1] * dims[1][l1], dims[2][l1], xin, xout);
		transpose2Global(xout, xin);
		transpose2Finish(dims[0][l1] * dims[1][l1], dims[2][l1], xin, xout);
	}

	/*
	 * xy transpose is a little tricky, since we don't want to touch 3rd
	 * axis. But alltoall must involve 3rd axis (most slowly varying) to be
	 * efficient. So we do (nx, ny/np1, nz/np2) -> (ny/np1, nz/np2, nx)
	 * (local) (ny/np1, nz/np2, nx) -> ((ny/np1*nz/np2)*np1, nx/np1)
	 * (global) then local finish.
	 */
	private void transposeXY(int l1, int l2, double xin[], double xout[]) {
		transposeXYLocal(dims[0][l1], dims[1][l1], dims[2][l1], xin, xout);
		transposeXYGlobal(dims[0][l1], dims[1][l1], dims[2][l1], xout, xin);
		transposeXYFinish(dims[0][l1], dims[1][l1], dims[2][l1], xin, xout);
	}

	private void transposeXZ(int l1, int l2, double xin[], double xout[]) {
		transposeXZLocal(dims[0][l1], dims[1][l1], dims[2][l1], xin, xout);
		transposeXZGlobal(dims[0][l1], dims[1][l1], dims[2][l1], xout, xin);
		transposeXZFinish(dims[0][l2], dims[1][l2], dims[2][l2], xin, xout);
	}

	private void transpose_x_yz(int l1, int l2, double xin[], double xout[]) {
		transpose2Local(dims[0][l1], dims[1][l1] * dims[2][l1], xin, xout);
		transpose2Global(xout, xin);
		transpose2Finish(dims[0][l1], dims[1][l1] * dims[2][l1], xin, xout);
	}

	/*
	 * Computes NY N-point complex-to-complex FFTs of X using an algorithm
	 * due to Swarztrauber. X is both the input and the output array, while
	 * Y is a scratch array. It is assumed that N = 2^M. Before calling
	 * CFFTZ to perform FFTs, the array U must be initialized by calling
	 * CFFTZ with IS set to 0 and M set to MX, where MX is the maximum value
	 * of M for any subsequent call.
	 * 
	 * Check if input parameters are invalid
	 */
	private int cfftz(int is, int m, int n, double x[], double y[], int offset) throws BadInputException {

		int l, j, i, pos;
		int mx = (int) u[0];
		
		if ((is != 1 && is != -1) || m < 1 || m > mx) {
			throw new BadInputException("CFFTZ: Either U has not been initialized,"
					+ "or else one of the input parameters is invalid " + is
					+ " " + m + " " + mx);
		}

		/* Perform one variant of the Stockham FFT, several stages per pass; x and y are disjoint halves. */
		boolean inX = true;
		l = 1;
		while (l <= m) {
			/* number of radix-2 stages done in this pass */
			final int R = (m - l >= 1) ? 2 : 1;
			if (R == 2) {
				if (inX) fftz4(is, l, m, n, fftBlock, fftBlockPad, u, x, 0, y, offset);
				else     fftz4(is, l, m, n, fftBlock, fftBlockPad, u, y, offset, x, 0);
			} else {
				if (inX) fftz2(is, l, m, n, fftBlock, fftBlockPad, u, x, 0, y, offset);
				else     fftz2(is, l, m, n, fftBlock, fftBlockPad, u, y, offset, x, 0);
			}
			l += R;
			inX = !inX;
		}
		/* the result is in x after an even number of passes, in y (at offset) otherwise */
		return inX ? 0 : offset;
	}

	/**
	 * Stages l to l+1 of the Stockham FFT in one pass (lk = 2^(l-1), li = 2^(m-l) >= 2). For every element it
	 * does the same operations as the 2 radix-2 stages (each one a sum, a difference and a product by its root of
	 * unity), so the rounding is identical; the intermediate values just stay in registers instead of going through
	 * the scratch array.
	 */
	private void fftz4(int is, int l, int m, int n, int ny, int ny1, double u[], double x[], int xoffset, double y[], int yoffset) {
		int i, j, k, tw;
		final int lk = 1 << (l - 1), li = 1 << (m - l);
		final int lis = li >> 1, na = ((n >> 2) * ny1) << 1, lkb = (lk * ny1) << 1, ny2 = ny << 1;

		for (i = 0; i < lis; i++) {
			tw = (li >> 0) + i;
			final double u0_0r = u[2 * tw], u0_0i = (is >= 1) ? u[2 * tw + 1] : -u[2 * tw + 1];
			tw = (li >> 0) + i + 1 * lis;
			final double u0_1r = u[2 * tw], u0_1i = (is >= 1) ? u[2 * tw + 1] : -u[2 * tw + 1];
			tw = (li >> 1) + i;
			final double u1_0r = u[2 * tw], u1_0i = (is >= 1) ? u[2 * tw + 1] : -u[2 * tw + 1];

			for (k = 0; k < lk; k++) {
				final int p0 = (((i * lk + k) * ny1) << 1) + xoffset; final int p1 = p0 + 1 * na; final int p2 = p0 + 2 * na; final int p3 = p0 + 3 * na;
				final int q0 = (((i * 4 * lk + k) * ny1) << 1) + yoffset; final int q1 = q0 + 1 * lkb; final int q2 = q0 + 2 * lkb; final int q3 = q0 + 3 * lkb;

				for (j = 0; j < ny2; j += 2) {
					final double v0_0_0r = x[p0 + j], v0_0_0i = x[p0 + j + 1];
					final double v0_1_0r = x[p1 + j], v0_1_0i = x[p1 + j + 1];
					final double v0_2_0r = x[p2 + j], v0_2_0i = x[p2 + j + 1];
					final double v0_3_0r = x[p3 + j], v0_3_0i = x[p3 + j + 1];
					final double v1_0_0r = v0_0_0r + v0_2_0r, v1_0_0i = v0_0_0i + v0_2_0i;
					final double d0_0_0r = v0_0_0r - v0_2_0r, d0_0_0i = v0_0_0i - v0_2_0i;
					final double v1_0_1r = u0_0r * d0_0_0r - u0_0i * d0_0_0i, v1_0_1i = u0_0r * d0_0_0i + u0_0i * d0_0_0r;
					final double v1_1_0r = v0_1_0r + v0_3_0r, v1_1_0i = v0_1_0i + v0_3_0i;
					final double d0_1_0r = v0_1_0r - v0_3_0r, d0_1_0i = v0_1_0i - v0_3_0i;
					final double v1_1_1r = u0_1r * d0_1_0r - u0_1i * d0_1_0i, v1_1_1i = u0_1r * d0_1_0i + u0_1i * d0_1_0r;
					y[q0 + j] = v1_0_0r + v1_1_0r;
					y[q0 + j + 1] = v1_0_0i + v1_1_0i;
					final double d1_0_0r = v1_0_0r - v1_1_0r, d1_0_0i = v1_0_0i - v1_1_0i;
					y[q2 + j] = u1_0r * d1_0_0r - u1_0i * d1_0_0i;
					y[q2 + j + 1] = u1_0r * d1_0_0i + u1_0i * d1_0_0r;
					y[q1 + j] = v1_0_1r + v1_1_1r;
					y[q1 + j + 1] = v1_0_1i + v1_1_1i;
					final double d1_0_1r = v1_0_1r - v1_1_1r, d1_0_1i = v1_0_1i - v1_1_1i;
					y[q3 + j] = u1_0r * d1_0_1r - u1_0i * d1_0_1i;
					y[q3 + j + 1] = u1_0r * d1_0_1i + u1_0i * d1_0_1r;
				}
			}
		}
	}

	/*
	 * Performs the L-th iteration of the second variant of the Stockham
	 * FFT.
	 * 
	 * Set initial parameters.
	 */
	private void fftz2(int is, int l, int m, int n, int ny, int ny1, double u[], double x[], int xoffset, double y[], int yoffset) {

		int i, i11, i12, i21, i22, k, j, op2kui, pos1, pos2, pos3, pos4;
		int n1 = n >> 1;
		/*
		 * int lk = (int)Math.pow(2,l-1); int li = (int)Math.pow(2,m-l);
		 */
		int lk;
		int li;
		if (l - 1 == 0)
			lk = 1;
		else
			lk = 2 << (l - 2);
		if (m - l == 0)
			li = 1;
		else
			li = 2 << (m - l - 1);
		int lj = lk << 1;
		int ku = li;

		int pos;
		int jx2;

		/* Complex variables */
		double real, imag, x11real, x11imag, x21real, x21imag;

		for (i = 0; i < li; i++) {
			i11 = i * lk + 1;
			i12 = i11 + n1;
			i21 = i * lj + 1;
			i22 = i21 + lk;
			op2kui = (ku + i) << 1;
			if (is >= 1) {
				real = u[op2kui];
				imag = u[IMAG + op2kui];
			} else {
				real = u[op2kui];
				imag = -u[IMAG + op2kui];
			}

			/* This loop is vectorizable. */
			for (k = 0; k < lk; k++) {
				pos1 = position2D(0, i11 + k, ny1, n) + xoffset;
				pos2 = position2D(0, i12 + k, ny1, n) + xoffset;
				pos3 = position2D(0, i21 + k, ny1, n) + yoffset;
				pos4 = position2D(0, i22 + k, ny1, n) + yoffset;
				
				for (j = 1; j <= ny; j++) {
					jx2 = j << 1;
					pos = pos1 + jx2;
					x11real = x[pos++];
					x11imag = x[pos];

					pos = pos2 + jx2;
					x21real = x[pos++];
					x21imag = x[pos];

					pos = pos3 + jx2;
					y[pos++] = x11real + x21real;
					y[pos] = x11imag + x21imag;

					pos = pos4 + jx2;
					x11real -= x21real;
					x11imag -= x21imag;
					y[pos++] = x11real * real - x11imag * imag;
					y[pos] = x11real * imag + x11imag * real;
				}
			}
		}
	}

	/*
	 * This method organizes the memory in the same way that fortran does,
	 * to facilitate the code translation and efficiency.
	 */
	private int position2D(int x, int y, int dimx, int dimy) {
		return (x - 1 + (y - 1) * dimx) << 1;
	}

	private void complexRealMultiplyWOCopy(double op1[], int op1offset, double op2) {
		op1[REAL + op1offset] *= op2;
		op1[IMAG + op1offset] *= op2;
	}

	private void transpose2Local(int n1, int n2, double xin[], double xout[]) {

		int i, j, ii, iii, jj, jjj, posxout, posxin, posz;
		double z[] = new double[2 * transBlockPad * transBlock];

		if (moreDetailedTiming) timer.start(tTransxzloc);
		
		if (n1 < transBlock || n2 < transBlock) {
			if (n1 >= n2) {
				for (j = 1; j <= n2; j++) {
					for (i = 1; i <= n1; i++) {
						posxout = position2D(j, i, n2, n1);
						posxin = position2D(i, j, n1, n2);
						xout[posxout] = xin[posxin];
						xout[IMAG + posxout] = xin[IMAG + posxin];
					}
				}
			} else {
				for (i = 1; i <= n1; i++) {
					for (j = 1; j <= n2; j++) {
						posxout = position2D(j, i, n2, n1);
						posxin = position2D(i, j, n1, n2);
						xout[posxout] = xin[posxin];
						xout[IMAG + posxout] = xin[IMAG + posxin];
					}
				}
			}
		} else {
			for (j = 0; j <= n2 - 1; j += transBlock) {
				for (i = 0; i <= n1 - 1; i += transBlock) {

					for (jj = 1; jj <= transBlock; jj++) {
						jjj = j + jj;
						for (ii = 1; ii <= transBlock; ii++) {
							posz = position2D(jj, ii, transBlockPad, transBlock);
							posxin = position2D(i + ii, jjj, n1, n2);
							z[posz] = xin[posxin];
							z[IMAG + posz] = xin[IMAG + posxin];
						}
					}

					for (ii = 1; ii <= transBlock; ii++) {
						iii = i + ii;
						for (jj = 1; jj <= transBlock; jj++) {
							posxout = position2D(j + jj, iii, n2, n1);
							posz = position2D(jj, ii, transBlockPad, transBlock);
							xout[posxout] = z[posz];
							xout[IMAG + posxout] = z[IMAG + posz];
						}
					}

				}
			}
		}
		if (moreDetailedTiming) timer.stop(tTransxzloc);
	}

	private void transpose2Global(double xin[], double xout[]) {
		if (moreDetailedTiming) {
			synchup();
			timer.start(tTransxzglo);
		}
		
		if (detailedTiming) timer.start(T_COMM);
		commSlice1.Alltoall(xin, 0, 2 * ntdivnp / numProcess, MPI.DOUBLE, xout, 0, 2 * ntdivnp / numProcess, MPI.DOUBLE);
		if (detailedTiming) timer.stop(T_COMM);
		
		if (moreDetailedTiming) timer.stop(tTransxzglo);
	}

	private void transpose2Finish(int n1, int n2, double xin[], double xout[]) {
		int p, ioff, j, i, posxout, posxin;

		if (moreDetailedTiming) timer.start(tTransxzfin);

		for (p = 0; p < np2; p++) {
			ioff = p * n2;
			for (j = 1; j <= n1 / np2; j++) {
				for (i = 1; i <= n2; i++) {
					posxout = position2D(i + ioff, j, n2 * np2, n1 / np2);
					posxin = position3D(i, j, p + 1, n2, n1 / np2, np2);
					xout[posxout] = xin[posxin];
					xout[IMAG + posxout] = xin[IMAG + posxin];
				}
			}
		}

		if (moreDetailedTiming) timer.stop(tTransxzfin);
	}

	private void transposeXYLocal(int d1, int d2, int d3, double xin[], double xout[]) {
		int i, j, k, posxout, posxin;

		if (moreDetailedTiming) timer.start(tTransxyloc);

		for (k = 1; k <= d3; k++) {
			for (i = 1; i <= d1; i++) {
				for (j = 1; j <= d2; j++) {
					posxout = position3D(j, k, i, d2, d3, d1);
					posxin = position3D(i, j, k, d1, d2, d3);
					xout[posxout] = xin[posxin];
					xout[IMAG + posxout] = xin[IMAG + posxin];
				}
			}
		}

		if (moreDetailedTiming) timer.stop(tTransxyloc);
	}

	private void transposeXYGlobal(int d1, int d2, int d3, double xin[], double xout[]) {
		
		if (moreDetailedTiming) {
			synchup();
			timer.start(tTransxyglo);
		}

		if (detailedTiming) timer.start(T_COMM);
		commSlice2.Alltoall(xin, 0, 2 * d1 * d2 * d3 / np1, MPI.DOUBLE, xout, 0, 2 * d1 * d2 * d3 / np1, MPI.DOUBLE);
		if (detailedTiming) timer.stop(T_COMM);

		if (moreDetailedTiming) timer.stop(tTransxyglo);
	}

	private void transposeXYFinish(int d1, int d2, int d3, double xin[], double xout[]) {
		int p, ioff, k, j, i, posxout, posxin;

		if (moreDetailedTiming) timer.start(tTransxyfin);
		
		for (p = 0; p < np1; p++) {
			ioff = p * d1 / np1;
			for (k = 1; k <= d3; k++) {
				for (j = 1; j <= d2; j++) {
					for (i = 1; i <= d1 / np1; i++) {
						posxout = position3D(i + ioff, j, k, d1, d2, d3);
						posxin = position4D(i, k, j, p + 1, d1 / np1, d3, d2, np1);
						xout[posxout] = xin[posxin];
						xout[IMAG + posxout] = xin[IMAG + posxin];
					}
				}
			}
		}
		
		if (moreDetailedTiming) timer.stop(tTransxyfin);
	}

	/*
	 * This method organizes the memory in the same way that fortran does,
	 * to facilitate the code translation and efficiency.
	 */
	private int position4D(int x, int y, int z, int k, int dimx, int dimy, int dimz, int dimk) {
		return (x - 1 + (y - 1) * dimx + (z - 1) * dimx * dimy + (k - 1) * dimx * dimy * dimz) << 1;
	}

	private void transposeXZLocal(int d1, int d2, int d3, double xin[], double xout[]) {
		int j, k, i, kk, k1, ii, i1, posxout, posxin, posbuf;
		int block1, block3;

		if (moreDetailedTiming) timer.start(tTransxzloc);

		block3 = d3;
		if (d1 < 32 || block3 == 1) {
			/* Basic transpose */
			for (j = 1; j <= d2; j++) {
				for (k = 1; k <= d3; k++) {
					for (i = 1; i <= d1; i++) {
						posxout = position3D(k, j, i, d3, d2, d1);
						posxin = position3D(i, j, k, d1, d2, d3);
						xout[posxout] = xin[posxin];
						xout[IMAG + posxout] = xin[IMAG + posxin];
					}
				}
			}
		} else {
			if (block3 > transBlock) {
				block3 = transBlock;
			}
			block1 = d1;
			if (block1 * block3 > transBlock * transBlock) {
				block1 = transBlock * transBlock / block3;
			}

			/* Blocked transpose */
			double buf[] = new double[2 * transBlockPad * maxDim];

			for (j = 1; j <= d2; j++) {
				for (kk = 0; kk <= d3 - block3; kk += block3) {
					for (ii = 0; ii <= d1 - block1; ii += block1) {

						for (k = 1; k <= block3; k++) {
							k1 = k + kk;
							for (i = 1; i <= block1; i++) {
								posbuf = position2D(k, i, transBlockPad, maxDim);
								posxin = position3D(i + ii, j, k1, d1, d2, d3);
								buf[posbuf] = xin[posxin];
								buf[IMAG + posbuf] = xin[IMAG + posxin];
							}
						}

						for (i = 1; i <= block1; i++) {
							i1 = i + ii;
							for (k = 1; k <= block3; k++) {
								posxout = position3D(k + kk, j, i1, d3, d2, d1);
								posbuf = position2D(k, i, transBlockPad, maxDim);
								xout[posxout] = buf[posbuf];
								xout[IMAG + posxout] = buf[IMAG + posbuf];
							}
						}

					}
				}
			}
		}

		if (moreDetailedTiming) timer.stop(tTransxzloc);
	}

	private void transposeXZGlobal(int d1, int d2, int d3, double xin[], double xout[]) {
		
		if (moreDetailedTiming) {
			synchup();
			timer.start(tTransxzglo);
		}

		if (detailedTiming) timer.start(T_COMM);
		commSlice1.Alltoall(xin, 0, 2 * d1 * d2 * d3 / np2, MPI.DOUBLE, xout, 0, 2 * d1 * d2 * d3 / np2, MPI.DOUBLE);
		if (detailedTiming) timer.stop(T_COMM);

		if (moreDetailedTiming) timer.stop(tTransxzglo);
	}

	private void transposeXZFinish(int d1, int d2, int d3, double xin[], double xout[]) {
		int p, ioff, k, j, i, posxout, posxin;

		if (moreDetailedTiming) timer.start(tTransxzfin);
		
		for (p = 0; p < np2; p++) {
			ioff = p * d1 / np2;
			for (k = 1; k <= d3; k++) {
				for (j = 1; j <= d2; j++) {
					for (i = 1; i <= d1 / np2; i++) {
						posxout = position3D(i + ioff, j, k, d1, d2, d3);
						posxin = position4D(i, j, k, p + 1, d1 / np2, d2, d3, np2);
						xout[posxout] = xin[posxin];
						xout[IMAG + posxout] = xin[IMAG + posxin];
					}
				}
			}
		}
		if (moreDetailedTiming)timer.stop(tTransxzfin);
	}

	private void evolve(double u0[], double u1[], double twiddle[], int d1, int d2, int d3) {
		int i, j, k, pos;
		
		for (k = 1; k <= d3; k++) {
			for (j = 1; j <= d2; j++) {
				for (i = 1; i <= d1; i++) {
					pos = position3D(i, j, k, d1, d2, d3);
					complexRealMultiplyWOCopy(u0, pos, twiddle[pos >> 1]);
					u1[pos] = u0[pos];
					u1[IMAG + pos] = u0[IMAG + pos];
				}
			}
		}
	}

	private void checksum(int i, double u1[], int d1, int d2, int d3) {

		aux[0] = ntotalF;
		chk[0] = chk[1] = allchk[0] = allchk[1] = aux[1] = 0;
		
		for (int j = 1; j <= 1024; j++) {
			int q = (j % nx) + 1;
			if (q >= xstart[0] && q <= xend[0]) {
				int r = ((3 * j) % ny) + 1;
				if (r >= ystart[0] && r <= yend[0]) {
					int s = ((5 * j) % nz) + 1;
					if (s >= zstart[0] && s <= zend[0]) {
						int pos = position3D(q - xstart[0] + 1, r - ystart[0] + 1, s - zstart[0] + 1, d1, d2, d3);
						chk[REAL] = chk[REAL] + u1[pos];
						chk[IMAG] = chk[IMAG] + u1[IMAG + pos];
					}
				}
			}
		}
		
		complexDivision(chk, aux);

		if (detailedTiming) timer.start(T_COMM);
		commWork.Reduce(chk, 0, allchk, 0, 2, MPI.DOUBLE, MPI.SUM, 0);
		if (detailedTiming) timer.stop(T_COMM);

		if (myRank == 0)
			System.out.format(" T =\t" + i + "     Checksum =  (" + allchk[REAL] + ", " + allchk[IMAG] + ")\n");

		if (i > 0) {
			sums[(i - 1) << 1] = allchk[REAL];
			sums[IMAG + ((i - 1) << 1)] = allchk[IMAG];
		}
	}

	private void complexDivision(double op1[], double op2[]) {
		double aux = op2[REAL] * op2[REAL] + op2[IMAG] * op2[IMAG];
		double x = (op1[REAL] * op2[REAL] + op1[IMAG] * op2[IMAG]) / aux;
		double y = (op1[IMAG] * op2[REAL] - op1[REAL] * op2[IMAG]) / aux;
		op1[REAL] = x;
		op1[IMAG] = y;
	}

	private boolean verify(int d1, int d2, int d3, int nt, char CLASS) {

		boolean verified = false;
		double vdata[][];
		double err;
		double epsilon = 1.0E-12;

		if(myRank != 0) return false;

		double vDataS[][] = {
				{ 5.546087004964E+02, 5.546385409189E+02, 5.546148406171E+02,
					5.545423607415E+02, 5.544255039624E+02,
					5.542683411902E+02 },
					{ 4.845363331978E+02, 4.865304269511E+02, 4.883910722336E+02,
						4.901273169046E+02, 4.917475857993E+02,
						4.932597244941E+02 } };

		double vDataW[][] = {
				{ 5.673612178944E+02, 5.631436885271E+02, 5.594024089970E+02,
					5.560698047020E+02, 5.530898991250E+02,
					5.504159734538E+02 },
					{ 5.293246849175E+02, 5.282149986629E+02, 5.270996558037E+02,
						5.260027904925E+02, 5.249400845633E+02,
						5.239212247086E+02 } };

		double vDataA[][] = {
				{ 5.046735008193E+02, 5.059412319734E+02, 5.069376896287E+02,
					5.077892868474E+02, 5.085233095391E+02,
					5.091487099959E+02 },
					{ 5.114047905510E+02, 5.098809666433E+02, 5.098144042213E+02,
						5.101336130759E+02, 5.104914655194E+02,
						5.107917842803E+02 } };

		double vDataB[][] = {
				{ 5.177643571579E+02, 5.154521291263E+02, 5.146409228649E+02,
					5.142378756213E+02, 5.139626667737E+02,
					5.137423460082E+02, 5.135547056878E+02,
					5.133910925466E+02, 5.132470705390E+02,
					5.131197729984E+02, 5.130070319283E+02,
					5.129070537032E+02, 5.128182883502E+02,
					5.127393733383E+02, 5.126691062020E+02,
					5.126064276004E+02, 5.125504076570E+02,
					5.125002331720E+02, 5.124551951846E+02,
					5.124146770029E+02 },
					{ 5.077803458597E+02, 5.088249431599E+02, 5.096208912659E+02,
						5.101023387619E+02, 5.103976610617E+02,
						5.105948019802E+02, 5.107404165783E+02,
						5.108576573661E+02, 5.109577278523E+02,
						5.110460304483E+02, 5.111252433800E+02,
						5.111968077718E+02, 5.112616233064E+02,
						5.113203605551E+02, 5.113735928093E+02,
						5.114218460548E+02, 5.114656139760E+02,
						5.115053595966E+02, 5.115415130407E+02,
						5.115744692211E+02 } };

		double vDataC[][] = {
				{ 5.195078707457E+02, 5.155422171134E+02, 5.144678022222E+02,
					5.140150594328E+02, 5.137550426810E+02,
					5.135811056728E+02, 5.134569343165E+02,
					5.133651975661E+02, 5.132955192805E+02,
					5.132410471738E+02, 5.131971141679E+02,
					5.131605205716E+02, 5.131290734194E+02,
					5.131012720314E+02, 5.130760908195E+02,
					5.130528295923E+02, 5.130310107773E+02,
					5.130103090133E+02, 5.129905029333E+02,
					5.129714421109E+02 },
					{ 5.149019699238E+02, 5.127578201997E+02, 5.122251847514E+02,
						5.121090289018E+02, 5.121143685824E+02,
						5.121496764568E+02, 5.121870921893E+02,
						5.122193250322E+02, 5.122454735794E+02,
						5.122663649603E+02, 5.122830879827E+02,
						5.122965869718E+02, 5.123075927445E+02,
						5.123166486553E+02, 5.123241541685E+02,
						5.123304037599E+02, 5.123356167976E+02,
						5.123399592211E+02, 5.123435588985E+02,
						5.123465164008E+02 } };

		double vDataD[][] = {
				{ 5.122230065252E+02, 5.120463975765E+02, 5.119865766760E+02,
					5.119518799488E+02, 5.119269088223E+02,
					5.119082416858E+02, 5.118943814638E+02,
					5.118842385057E+02, 5.118769435632E+02,
					5.118718203448E+02, 5.118683569061E+02,
					5.118661708593E+02, 5.118649768950E+02,
					5.118645605626E+02, 5.118647586618E+02,
					5.118654451572E+02, 5.118665212451E+02,
					5.118679083821E+02, 5.118695433664E+02,
					5.118713748264E+02, 5.118733606701E+02,
					5.118754661974E+02, 5.118776626738E+02,
					5.118799262314E+02, 5.118822370068E+02 },
					{ 5.118534037109E+02, 5.117061181082E+02, 5.117096364601E+02,
						5.117373863950E+02, 5.117680347632E+02,
						5.117967875532E+02, 5.118225281841E+02,
						5.118451629348E+02, 5.118649119387E+02,
						5.118820803844E+02, 5.118969781011E+02,
						5.119098918835E+02, 5.119210777066E+02,
						5.119307604484E+02, 5.119391362671E+02,
						5.119463757241E+02, 5.119526269238E+02,
						5.119580184108E+02, 5.119626617538E+02,
						5.119666538138E+02, 5.119700787219E+02,
						5.119730095953E+02, 5.119755100241E+02,
						5.119776353561E+02, 5.119794338060E+02 } };

		double vDataE[][] = {
				{ 5.121601045346E+02, 5.120905403678E+02, 5.120623229306E+02,
					5.120438418997E+02, 5.120311521872E+02, 5.120226088809E+02,
					5.120169296534E+02, 5.120131225172E+02, 5.120104767108E+02,
					5.120085127969E+02, 5.120069224127E+02, 5.120055158164E+02,
					5.120041820159E+02, 5.120028605402E+02, 5.120015223011E+02,
					5.120001570022E+02, 5.119987650555E+02, 5.119973525091E+02,
					5.119959279472E+02, 5.119945006558E+02, 5.119930795911E+02,
					5.119916728462E+02, 5.119902874185E+02, 5.119889291565E+02,
					5.119876028049E+02 },
					{ 5.117395998266E+02, 5.118614716182E+02, 5.119074203747E+02,
					5.119345900733E+02, 5.119551325550E+02, 5.119720179919E+02,
					5.119861371665E+02, 5.119979364402E+02, 5.120077674092E+02,
					5.120159443121E+02, 5.120227453670E+02, 5.120284096041E+02,
					5.120331373793E+02, 5.120370938679E+02, 5.120404138831E+02,
					5.120432068837E+02, 5.120455615860E+02, 5.120475499442E+02,
					5.120492304629E+02, 5.120506508902E+02, 5.120518503782E+02,
					5.120528612016E+02, 5.120537101195E+02, 5.120544194514E+02,
					5.120550079284E+02 } };

		CLASS = 'U';

		if (d1 == 64 && d2 == 64 && d3 == 64 && nt == 6)
			CLASS = 'S';
		else if (d1 == 128 && d2 == 128 && d3 == 32 && nt == 6)
			CLASS = 'W';
		else if (d1 == 256 && d2 == 256 && d3 == 128 && nt == 6)
			CLASS = 'A';
		else if (d1 == 512 && d2 == 256 && d3 == 256 && nt == 20)
			CLASS = 'B';
		else if (d1 == 512 && d2 == 512 && d3 == 512 && nt == 20)
			CLASS = 'C';
		else if (d1 == 2048 && d2 == 1024 && d3 == 1024 && nt == 25)
			CLASS = 'D';

		if (CLASS == 'S') {
			vdata = vDataS;
		} else if (CLASS == 'W') {
			vdata = vDataW;
		} else if (CLASS == 'A') {
			vdata = vDataA;
		} else if (CLASS == 'B') {
			vdata = vDataB;
		} else if (CLASS == 'C') {
			vdata = vDataC;
		} else if (CLASS == 'D') {
			vdata = vDataD;
		} else if (CLASS == 'E') {
			vdata = vDataE;
		} else {
			return false;
		}

		verified = true;

		for (int i = 0; i < nt; i++) {
			err = (sums[2 * i] - vdata[REAL][i]) / vdata[REAL][i];
			if (Double.isNaN(err) || Math.abs(err) > epsilon) {
				verified = false;
				break;
			}
			err = (sums[IMAG + 2 * i] - vdata[IMAG][i]) / vdata[IMAG][i];
			if (Double.isNaN(err) || Math.abs(err) > epsilon) {
				verified = false;
				break;
			}
		}

		return verified;
	}

	private void printTimers() {
		String tstrings[] = {
				"timer  1(total)         : ",
				"timer  2(setup)         : ",
				"timer  3(fft)           : ",
				"timer  4(evolve)        : ",
				"timer  5(checksum)      : ",
				"timer  6(fftlow)        : ",
				"timer  7(fftcopy)       : ",
				"timer  8(transpose)     : ",
				"timer  9(transpose1Loc) : ",
				"timer 10(transpose1Glo) : ",
				"timer 11(transpose1Fin) : ",
				"timer 12(transpose2Loc) : ",
				"timer 13(transpose2Glo) : ",
				"timer 14(transpose2Fin) : ",
				"timer 15(sync)          : " };
		for (int i = 0; i < tstrings.length; i++) {
			if (timer.read(i) != 0)
				System.out.format(java.util.Locale.ENGLISH, "%s%8.4f\n", tstrings[i], timer.read(i));
		}
	}
}
