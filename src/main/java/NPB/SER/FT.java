/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB 3.4.4 OpenMP (NPB3.4-OMP/FT)
 */
package NPB.SER;

import NPB.Benchmark;
import NPB.common.io.BMArgs;
import NPB.common.io.BMResults;
import NPB.common.numerics.Random;
import NPB.common.timer.Timer;
import NPB.exceptions.BadClassException;
import NPB.exceptions.NPBException;


public class FT implements Benchmark{

	private static final String BM_NAME = "FT";
	private char CLASS;
	private boolean detailedTiming;

	private int nx, ny, nz, maxDim, niter;
	/* nx*ny*nz is 2^31 for class D: it does not fit in an int. */
	private long ntotal;
	/* interleaved work block of the FFTs and, in its second half, their scratch */
	private double work[];

	private int CacheSize = 32768;
	private int BlockMax = 64;

	/* Real and imaginary parts in complex vectors. */
	private final int REAL = 0;
	private final int IMAG = 1;
	private final double pi = 3.141592653589793238;
	private final double alpha = 1E-6;
	private final double seed = 314159265;
	private final double a = 1220703125;
	private final double ap = -4 * alpha * pi * pi;

	private double sums[][];

	/*
	 * Chunked layout of the 3D arrays (xnt, y, twiddle). Java arrays are indexed with an int, and for class D
	 * (nx+1)*ny*nz = 2148532224 > 2^31-1 elements per real component, so the grid is stored as a sequence of chunks of
	 * whole z planes (a plane is (nx+1)*ny elements, laid out as in Fortran: x fastest). Complex grids are
	 * grid[chunk][REAL or IMAG][offset in chunk], twiddle is twiddle[chunk][offset in chunk]. Every FFT along x and y
	 * works inside one plane; only the FFT along z crosses chunks. The inner loops still run over plain double[].
	 */
	/** Default maximum number of doubles in one chunk (2 GB): classes S to C fit in a single chunk. */
	static final long DEFAULT_MAX_CHUNK_ELEMS = 1L << 28;
	/** System property that overrides the default maximum chunk size (in doubles) of the default constructor. */
	static final String CHUNK_ELEMS_PROPERTY = "npb.ft.chunkelems";
	private final long maxChunkElems;
	private int planeSize;      // (nx+1)*ny
	private int planesPerChunk; // z planes in every chunk but the last one
	private int nChunks;
	private int zChunk[];       // z plane (0-based) -> chunk
	private int zOff[];         // z plane (0-based) -> offset of the plane inside its chunk

	private Random rng;
	private Timer timer;
	private BMResults res;

	/*
	 * Different timers
	 */
	private final int T_1 = 1;
	private final int T_2 = 2;
	private final int T_3 = 3;
	private final int T_4 = 4;
	private final int T_7 = 7;
	private final int T_8 = 8;
	private final int T_9 = 9;
	private final int T_10 = 10;
	private final int T_11 = 11;
	private final int T_12 = 12;
	private final int T_13 = 13;
	private final int T_14 = 14;
	private final int T_15 = 15;

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
		this(classSize, dtiming, Long.getLong(CHUNK_ELEMS_PROPERTY, DEFAULT_MAX_CHUNK_ELEMS));
	}

	/**
	 * @param maxChunkElems maximum number of doubles in a chunk of the 3D arrays (see the chunked layout above); at
	 *                      least one z plane per chunk is always used.
	 */
	public FT(char classSize, boolean dtiming, long maxChunkElems){

		CLASS = classSize;
		detailedTiming = dtiming;
		this.maxChunkElems = maxChunkElems;
	}

	/**
	 * Number of z planes per chunk: as many as fit in maxElems doubles (and in a Java array), at least 1, at most nz.
	 */
	static int planesPerChunk(int planeSize, int nz, long maxElems) {
		long limit = Math.min(maxElems, Integer.MAX_VALUE - 8);
		long planes = limit / planeSize;
		return (int) Math.max(1, Math.min(nz, planes));
	}

	public void build(String[] args) throws NPBException {

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
		default:
			throw new BadClassException("Specified class (" + CLASS + ") not valid for this benchmark\n"
					+ "Valid classes are S, W, A, B, C and D");
		}

		ntotal = (long) nx * ny * nz;
		planeSize = (nx + 1) * ny;
		planesPerChunk = planesPerChunk(planeSize, nz, maxChunkElems);
		nChunks = (nz + planesPerChunk - 1) / planesPerChunk;
		zChunk = new int[nz];
		zOff = new int[nz];
		for (int p = 0; p < nz; p++) {
			zChunk[p] = p / planesPerChunk;
			zOff[p] = (p % planesPerChunk) * planeSize;
		}
		work = new double[2 * 2 * (BlockMax + 2) * maxDim];
		sums = new double[2][niter];
		rng = new Random();
	}

	/** Number of chunks the 3D arrays are split into (valid after build). */
	int getChunkCount() {
		return nChunks;
	}

	/** Chunk holding z plane p (0-based). Package-private: used by the tests (also with class D, which does not fit here). */
	int chunkOfPlane(int p) {
		return zChunk[p];
	}

	/** Offset of z plane p (0-based) inside its chunk. */
	int offsetOfPlane(int p) {
		return zOff[p];
	}

	/** Number of z planes. */
	int getNz() {
		return nz;
	}

	/** nx*ny*nz (long: 2^31 for class D). */
	long getNtotal() {
		return ntotal;
	}

	/** Number of doubles of a z plane, (nx+1)*ny. */
	int getPlaneSize() {
		return planeSize;
	}

	/** Number of z planes in chunk c. */
	int planesInChunk(int c) {
		return planesIn(c);
	}

	/** Number of z planes in chunk c. */
	private int planesIn(int c) {
		return Math.min(planesPerChunk, nz - c * planesPerChunk);
	}

	/** New complex grid: grid[chunk][REAL or IMAG][offset in chunk]. */
	private double[][][] newGrid() {
		double g[][][] = new double[nChunks][2][];
		for (int c = 0; c < nChunks; c++) {
			g[c][REAL] = new double[planesIn(c) * planeSize];
			g[c][IMAG] = new double[planesIn(c) * planeSize];
		}
		return g;
	}

	/** New real grid: grid[chunk][offset in chunk]. */
	private double[][] newRealGrid() {
		double g[][] = new double[nChunks][];
		for (int c = 0; c < nChunks; c++)
			g[c] = new double[planesIn(c) * planeSize];
		return g;
	}

	public void runBenchmark(String[] args) throws NPBException {

		int n12, n22, n32, i, ii, ii2, j, jj, k, kk, ik2, pos;
		double tw[];

		this.build(args);

		timer = new Timer();
		
		double twiddle[][] = newRealGrid();
		double xnt[][][]   = newGrid();
		double y[][][]     = newGrid();
		double exp1[][]  = new double[2][nx];
		double exp2[][]  = new double[2][ny];
		double exp3[][]  = new double[2][nz];

		BMArgs.bannerSER(BM_NAME, CLASS);
		System.out.println(" Size                :  " + nx + "x  " + ny + "x  " + nz);
		System.out.println(" Iterations          :  " + niter);
		System.out.println();

		if (detailedTiming) timer.start(T_2);
		computeInitialConditions(xnt, nx, ny, nz);
		CompExp(nx, exp1);
		CompExp(ny, exp2);
		CompExp(nz, exp3);
		fftXYZ(1, xnt, y, exp1, exp2, exp3, nx, ny, nz);
		if (detailedTiming) timer.stop(T_2);

		timer.start(T_1);

		if (detailedTiming) timer.start(T_13);

		n12 = nx >> 1;
		n22 = ny >> 1;
		n32 = nz >> 1;

		for (i = 1; i <= nz; i++) {
			ii = i-1-((i-1)/n32)*nz;
			ii2 = ii * ii;
			for (k = 1; k <= ny; k++) {
				kk = k-1-((k-1)/n22)*ny;
				ik2 = ii2 + kk*kk;
				tw = twiddle[zChunk[i-1]];
				pos = zOff[i-1] + (k-1)*(nx+1);
				for (j = 1; j <= nx; j++) {
					jj = j-1-((j-1)/n12)*nx;
					tw[pos] = Math.exp(ap*(jj*jj + ik2));
					pos++;
				}
			}
		}

		if (detailedTiming){
			timer.stop(T_13);
			timer.start(T_12);
		}

		computeInitialConditions(xnt, nx, ny, nz);

		if (detailedTiming) {
			timer.stop(T_12);
			timer.start(T_15);
		}

		fftXYZ(1, xnt, y, exp1, exp2, exp3, nx, ny, nz);

		if (detailedTiming) timer.stop(T_15);

		for (int iter = 1; iter <= niter; iter++) {
			if (detailedTiming) timer.start(T_11);
			evolve(xnt, y, twiddle, nx, ny, nz);
			if (detailedTiming) {
				timer.stop(T_11);
				timer.start(T_15);
			}
			fftXYZ(-1, xnt, xnt, exp1, exp2, exp3, nx, ny, nz);
			if (detailedTiming) {
				timer.stop(T_15);
				timer.start(T_10);
			}
			CalculateChecksum(iter, xnt, nx, ny, nz);
			if (detailedTiming) timer.stop(T_10);
		}

		boolean verified = false;

		if (detailedTiming) timer.start(T_14);
		verified = verify(nx, ny, nz, niter, CLASS);
		if (detailedTiming) timer.stop(T_14);

		timer.stop(T_1);

		double totalTime = timer.read(T_1);
		double MFLOPS = 0;

		if (totalTime != 0)
			MFLOPS = 1E-6 * ntotal * 
			(14.8157 + 7.19641 * Math.log(ntotal) +
					(5.23518 + 7.21113 * Math.log(ntotal))*niter) / totalTime;

		int nVerified = 0;
		if (verified)
			nVerified = 1;

		BMResults.printVerificationStatus(CLASS, nVerified);

		double times[] = { 0, 0 };
		times[0] = totalTime;

		res = new BMResults(BM_NAME, CLASS, nx, ny, nz, niter,
				times, MFLOPS, "floating point", nVerified, 1,
				detailedTiming);

		res.printSER();

		if (detailedTiming)
			printTimers();

		terminate();
	}

	private void terminate() {
	}

	/*
	 * This method organizes the memory in the same way that fortran does,
	 * to facilitate the code translation and efficiency.
	 */
	private int position2D(int x, int y, int dimx, int dimy) {
		return x - 1 + (y - 1) * dimx;
	}

	/*
	 * There is no position3D: the 3D arrays are chunked (see above) and the position of the element (x, y, z)
	 * (1-based, rows of nx+1 elements and planes of (nx+1)*ny) is chunk zChunk[z-1], offset
	 * zOff[z-1] + (y-1)*(nx+1) + (x-1) inside the chunk. A full-grid index does not fit in an int for class D.
	 */

	private void computeInitialConditions(double u0[][][], int d1, int d2, int d3) {

		double x0, start;
		double RanStarts[] = new double[maxDim];
		double tmp[][] = new double[2][maxDim];
		int i,j,k,pos;
		double u0r[], u0i[];

		start = seed;

		/* Jump to the starting element for our first plane. */
		double an = ipow46(a, 0);
		double values[] = rng.vrandlc(start, an);
		start = values[0];
		an = ipow46(a, 2*d1*d2);

		/* Go through by z planes filling in one square at a time */
		RanStarts[0] = start;
		for (k = 2; k <= d3; k++) {
			values = rng.vrandlc(start, an);
			start = values[0];
			RanStarts[k-1] = start;
		}

		for (k = 1; k <= d3; k++) {
			x0 = RanStarts[k-1];
			for (j = 1; j <= d2; j++) {
				x0 = rng.complexvranlc(d1, x0, a, tmp, 0);
				u0r = u0[zChunk[k-1]][REAL];
				u0i = u0[zChunk[k-1]][IMAG];
				pos = zOff[k-1] + (j-1)*(d1+1);
				for (i = 1; i <= d1; i++){
					u0r[pos] = tmp[REAL][i-1];
					u0i[pos] = tmp[IMAG][i-1];
					pos++;
				}
			}
		}
	}

	/*
	 * compute the roots-of-unity array that will be used for subsequent FFTs. 
	 */
	private void CompExp(int n, double exponent[][]){
		int m,ku,i,j,ln;
		double t, ti;

		m = ilog2(n);
		exponent[REAL][0] = m;
		exponent[IMAG][0] = m;
		ku = 1;
		ln = 1;

		for (j = 0; j < m; j++) {
			t = pi / ln;

			for (i = 0; i < ln; i++) {
				ti = i * t;
				exponent[REAL][i + ku] = Math.cos(ti);
				exponent[IMAG][i + ku] = Math.sin(ti);
			}
			ku += ln;
			ln = ln << 1;
		}
	}

	/*
	 * Compute a^exponent mod 2^46
	 * 
	 * Use: a^n = a^(n/2)*a^(n/2) if n even else a^n = a*a^(n-1) if n odd
	 */
	private double ipow46(double a, int exp) {
		double result = 1.0;
		int n2;
		double values[];

		if (exp == 0) {
			return result;
		}

		double q = a;
		double r = 1;
		int n = exp;

		while (n > 1) {
			n2 = n >> 1;
			if (n2 * 2 == n) {
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
			nn *= 2;
			lg++;
		}
		return lg;
	}

	/*
	 * 3D FFT. As in NPB (cffts1/2/3 in ft.f90) the forward transform goes through the dimensions in the order
	 * x, y, z and the inverse one in the order z, y, x, writing the result into xout in the last step. A DFT is
	 * separable, so the order does not change the result mathematically, but it does change the rounding: always
	 * using x, y, z gave checksums that differed from the official ones by a few ulp.
	 */
	private void fftXYZ(int sign, double x[][][], double xout[][][], double exp1[][], double exp2[][], double exp3[][], int n1, int n2, int n3) {

		if (detailedTiming) timer.start(T_3);

		if (sign == 1) {
			fftX(sign, x, x, exp1, n1, n2, n3);
			fftY(sign, x, exp2, n1, n2, n3);
			fftZ(sign, x, xout, exp3, n1, n2, n3);
		} else {
			fftZ(sign, x, x, exp3, n1, n2, n3);
			fftY(sign, x, exp2, n1, n2, n3);
			fftX(sign, x, xout, exp1, n1, n2, n3);
		}

		if (detailedTiming) timer.stop(T_3);
	}

	/** FFTs along x (cffts1): reads x, writes dst. */
	private void fftX(int sign, double x[][][], double dst[][][], double exp1[][], int n1, int n2, int n3) {
		int i, j, k, bls, len, pos, pp;
		double xr[] = null, xi[] = null, dr[] = null, di[] = null;
		/* FFTs done at a time; they are copied to an interleaved (re, im) block of blk*n1 complex numbers */
		final int blk = Math.min(CacheSize/n1, BlockMax), pad = blk + 2, pad2 = pad << 1, log = ilog2(n1);
		final double w[] = work;

		if (detailedTiming) timer.start(T_7);

		for (k = 0; k < n3; k++) {
			xr = x[zChunk[k]][REAL]; xi = x[zChunk[k]][IMAG];
			dr = dst[zChunk[k]][REAL]; di = dst[zChunk[k]][IMAG];
			for (bls = 0; bls < n2; bls += blk) {
				len = Math.min(blk, n2 - bls);

				for (j = 0; j < len; j++) {
					pos = zOff[k] + (bls + j)*(n1+1);
					pp = j << 1;
					for (i = 0; i < n1; i++) {
						w[pp] = xr[pos+i];
						w[pp+1] = xi[pos+i];
						pp += pad2;
					}
				}

				final int ro = cfftz(sign, log, n1, len, pad, exp1, w);

				for (j = 0; j < len; j++) {
					pos = zOff[k] + (bls + j)*(n1+1);
					pp = ro + (j << 1);
					for (i = 0; i < n1; i++) {
						dr[pos+i] = w[pp];
						di[pos+i] = w[pp+1];
						pp += pad2;
					}
				}
			}
		}

		if (detailedTiming) timer.stop(T_7);
	}

	/** FFTs along y (cffts2): in place on x. */
	private void fftY(int sign, double x[][][], double exp2[][], int n1, int n2, int n3) {
		int i, j, k, bls, len, pos, pp;
		double xr[] = null, xi[] = null, dr[] = null, di[] = null;
		/* FFTs done at a time; they are copied to an interleaved (re, im) block of blk*n2 complex numbers */
		final int blk = Math.min(CacheSize/n2, BlockMax), pad = blk + 2, pad2 = pad << 1, log = ilog2(n2);
		final double w[] = work;

		if (detailedTiming) timer.start(T_8);

		for (k = 0; k < n3; k++) {
			xr = x[zChunk[k]][REAL]; xi = x[zChunk[k]][IMAG];
			dr = xr; di = xi;
			for (bls = 0; bls < n1; bls += blk) {
				len = Math.min(blk, n1 - bls);

				for (j = 0; j < n2; j++) {
					pos = zOff[k] + j*(n1+1) + bls;
					pp = j * pad2;
					for (i = 0; i < len; i++) {
						w[pp + 2*i] = xr[pos+i];
						w[pp + 2*i + 1] = xi[pos+i];
					}
				}

				final int ro = cfftz(sign, log, n2, len, pad, exp2, w);

				for (j = 0; j < n2; j++) {
					pos = zOff[k] + j*(n1+1) + bls;
					pp = ro + j * pad2;
					for (i = 0; i < len; i++) {
						dr[pos+i] = w[pp + 2*i];
						di[pos+i] = w[pp + 2*i + 1];
					}
				}
			}
		}

		if (detailedTiming) timer.stop(T_8);
	}

	/** FFTs along z (cffts3): reads x, writes dst. */
	private void fftZ(int sign, double x[][][], double dst[][][], double exp3[][], int n1, int n2, int n3) {
		int i, j, k, bls, len, pos, pp;
		double xr[] = null, xi[] = null, dr[] = null, di[] = null;
		/* FFTs done at a time; they are copied to an interleaved (re, im) block of blk*n3 complex numbers */
		final int blk = Math.min(CacheSize/n3, BlockMax), pad = blk + 2, pad2 = pad << 1, log = ilog2(n3);
		final double w[] = work;

		if (detailedTiming) timer.start(T_9);

		for (k = 0; k < n2; k++) {
			for (bls = 0; bls < n1; bls += blk) {
				len = Math.min(blk, n1 - bls);

				for (j = 0; j < n3; j++) {
					xr = x[zChunk[j]][REAL]; xi = x[zChunk[j]][IMAG];
					pos = zOff[j] + k*(n1+1) + bls;
					pp = j * pad2;
					for (i = 0; i < len; i++) {
						w[pp + 2*i] = xr[pos+i];
						w[pp + 2*i + 1] = xi[pos+i];
					}
				}

				final int ro = cfftz(sign, log, n3, len, pad, exp3, w);

				for (j = 0; j < n3; j++) {
					dr = dst[zChunk[j]][REAL]; di = dst[zChunk[j]][IMAG];
					pos = zOff[j] + k*(n1+1) + bls;
					pp = ro + j * pad2;
					for (i = 0; i < len; i++) {
						dr[pos+i] = w[pp + 2*i];
						di[pos+i] = w[pp + 2*i + 1];
					}
				}
			}
		}

		if (detailedTiming) timer.stop(T_9);
	}

	/**
	 * ny FFTs of n = 2^m points (Stockham, due to Swarztrauber) on the interleaved block w, laid out as
	 * w[2*(v + ny1*t)] (v = FFT, t = point); the second half of w is the scratch. Two stages per pass.
	 */
	private int cfftz(int is, int m, int n, int ny, int ny1, double exponent[][], double w[]) {
		final double er[] = exponent[REAL], ei[] = exponent[IMAG];
		final int offset = (ny1 * n) << 1;
		boolean inX = true;
		int l = 1;

		if (detailedTiming) timer.start(T_4);

		while (l <= m) {
			if (m - l >= 1) {
				if (inX) fftz4(is, l, m, n, ny, ny1, er, ei, w, 0, w, offset);
				else     fftz4(is, l, m, n, ny, ny1, er, ei, w, offset, w, 0);
				l += 2;
			} else {
				if (inX) fftz2(is, l, m, n, ny, ny1, er, ei, w, 0, w, offset);
				else     fftz2(is, l, m, n, ny, ny1, er, ei, w, offset, w, 0);
				l += 1;
			}
			inX = !inX;
		}

		if (detailedTiming) timer.stop(T_4);

		/* the result is at the start of w after an even number of passes, in its second half otherwise */
		return inX ? 0 : offset;
	}

	/** One radix-2 stage (stage l) of the Stockham FFT on interleaved data: reads x, writes y (disjoint). */
	private static void fftz2(int is, int l, int m, int n, int ny, int ny1, double er[], double ei[], double x[], int xoffset, double y[], int yoffset) {
		int i, j, k;
		final int lk = 1 << (l - 1), li = 1 << (m - l), n2a = ((n >> 1) * ny1) << 1, lkb = (lk * ny1) << 1, ny2 = ny << 1;

		for (i = 0; i < li; i++) {
			final double ur = er[li + i], ui = (is >= 1) ? ei[li + i] : -ei[li + i];

			for (k = 0; k < lk; k++) {
				final int p0 = (((i * lk + k) * ny1) << 1) + xoffset, p1 = p0 + n2a;
				final int q0 = (((i * 2 * lk + k) * ny1) << 1) + yoffset, q1 = q0 + lkb;

				for (j = 0; j < ny2; j += 2) {
					final double ar = x[p0 + j], ai = x[p0 + j + 1], br = x[p1 + j], bi = x[p1 + j + 1];
					y[q0 + j] = ar + br;
					y[q0 + j + 1] = ai + bi;
					final double dr = ar - br, di = ai - bi;
					y[q1 + j] = ur * dr - ui * di;
					y[q1 + j + 1] = ur * di + ui * dr;
				}
			}
		}
	}

	/**
	 * Stages l to l+1 of the Stockham FFT in one pass (lk = 2^(l-1), li = 2^(m-l) >= 2). For every element it
	 * does the same operations as the 2 radix-2 stages (each one a sum, a difference and a product by its root of
	 * unity), so the rounding is identical; the intermediate values just stay in registers instead of going through
	 * the scratch array.
	 */
	private static void fftz4(int is, int l, int m, int n, int ny, int ny1, double er[], double ei[], double x[], int xoffset, double y[], int yoffset) {
		int i, j, k, tw;
		final int lk = 1 << (l - 1), li = 1 << (m - l);
		final int lis = li >> 1, na = ((n >> 2) * ny1) << 1, lkb = (lk * ny1) << 1, ny2 = ny << 1;

		for (i = 0; i < lis; i++) {
			tw = (li >> 0) + i;
			final double u0_0r = er[tw], u0_0i = (is >= 1) ? ei[tw] : -ei[tw];
			tw = (li >> 0) + i + 1 * lis;
			final double u0_1r = er[tw], u0_1i = (is >= 1) ? ei[tw] : -ei[tw];
			tw = (li >> 1) + i;
			final double u1_0r = er[tw], u1_0i = (is >= 1) ? ei[tw] : -ei[tw];

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

	private void evolve(double x[][][], double y[][][], double twiddle[][], int nx, int ny, int nz) {

		double aux[];
		int i,j,k,pos;
		double xr[], xi[], yr[], yi[], tw[];

		for (i = 1; i <= nz; i++) {
			xr = x[zChunk[i-1]][REAL];
			xi = x[zChunk[i-1]][IMAG];
			yr = y[zChunk[i-1]][REAL];
			yi = y[zChunk[i-1]][IMAG];
			tw = twiddle[zChunk[i-1]];
			for (k = 1; k <= ny; k++) {
				pos = zOff[i-1] + (k-1)*(nx+1);
				for (j = 1; j <= nx; j++) {
					double op1[] = { yr[pos], yi[pos] };
					double op2[] = { tw[pos], 0.0 };
					aux = complexMultiply(op1, op2);
					yr[pos] = aux[REAL];
					yi[pos] = aux[IMAG];
					xr[pos] = aux[REAL];
					xi[pos] = aux[IMAG];
					pos++;
				}
			}
		}
	}

	private void CalculateChecksum(int iterN, double u[][][], int d1, int d2, int d3) {
		int i, i1, ii, ji, ki, pos;
		double csum[] = { 0.0, 0.0 };

		for (i = 1; i <= 1024; i++) {
			i1 = i;
			ii = (i1 % d1) + 1;
			ji = ((3 * i1) % d2) + 1;
			ki = ((5 * i1) % d3) + 1;
			pos = zOff[ki-1] + (ji-1)*(d1+1) + (ii-1);
			csum[REAL] = csum[REAL] + u[zChunk[ki-1]][REAL][pos];
			csum[IMAG] = csum[IMAG] + u[zChunk[ki-1]][IMAG][pos];
		}

		double aux[] = { (double)ntotal, 0.0 };
		csum = complexDivision(csum, aux);

		System.out.format(" T =\t" + iterN + "     Checksum =  (" + csum[REAL] + ", " + csum[IMAG] + ")\n");

		if (iterN > 0) {
			sums[REAL][iterN - 1] = csum[REAL];
			sums[IMAG][iterN - 1] = csum[IMAG];
		}
	}

	private double[] complexMultiply(double op1[], double op2[]) {
		double complex1 = op1[REAL] * op2[REAL];
		double complex2 = op1[IMAG] * op2[IMAG];
		double result[] = { complex1 - complex2, 0 };

		complex1 = op1[REAL] * op2[IMAG];
		complex2 = op1[IMAG] * op2[REAL];
		result[IMAG] = complex1 + complex2;

		return result;
	}

	private double[] complexDivision(double op1[], double op2[]) {
		double aux = op2[REAL] * op2[REAL] + op2[IMAG] * op2[IMAG];
		double x = (op1[REAL] * op2[REAL] + op1[IMAG] * op2[IMAG]) / aux;
		double y = (op1[IMAG] * op2[REAL] - op1[REAL] * op2[IMAG]) / aux;
		double result[] = { x, y };
		return result;
	}

	private boolean verify(int d1, int d2, int d3, int nt, char CLASS) {
		boolean verified = false;

		double vdata[][];
		double err;
		double epsilon = 1.0E-12;

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
		} else {
			System.err.println("Invalid class. Verification not performed");
			return false;
		}

		verified = true;

		for (int i = 0; i < nt; i++) {
			err = (sums[REAL][i] - vdata[REAL][i]) / vdata[REAL][i];
			if (Double.isNaN(err) || Math.abs(err) > epsilon) {
				verified = false;
				break;
			}
			err = (sums[IMAG][i] - vdata[IMAG][i]) / vdata[IMAG][i];
			if (Double.isNaN(err) || Math.abs(err) > epsilon) {
				verified = false;
				break;
			}
		}

		return verified;
	}

	private void printTimers() {
		System.out.format(java.util.Locale.ENGLISH,"FT total                   : %8.4f\n",timer.read(T_1));
		System.out.format(java.util.Locale.ENGLISH,"WarmUp time                : %8.4f\n",timer.read(T_2));
		System.out.format(java.util.Locale.ENGLISH,"fftXYZ body                : %8.4f\n",timer.read(T_3));
		System.out.format(java.util.Locale.ENGLISH,"Swarztrauber               : %8.4f\n",timer.read(T_4));
		System.out.format(java.util.Locale.ENGLISH,"X time                     : %8.4f\n",timer.read(T_7));
		System.out.format(java.util.Locale.ENGLISH,"Y time                     : %8.4f\n",timer.read(T_8));
		System.out.format(java.util.Locale.ENGLISH,"Z time                     : %8.4f\n",timer.read(T_9));
		System.out.format(java.util.Locale.ENGLISH,"CalculateChecksum          : %8.4f\n",timer.read(T_10));
		System.out.format(java.util.Locale.ENGLISH,"evolve                     : %8.4f\n",timer.read(T_11));
		System.out.format(java.util.Locale.ENGLISH,"compute_initial_conditions : %8.4f\n",timer.read(T_12));
		System.out.format(java.util.Locale.ENGLISH,"twiddle                    : %8.4f\n",timer.read(T_13));
		System.out.format(java.util.Locale.ENGLISH,"verify                     : %8.4f\n",timer.read(T_14));
		System.out.format(java.util.Locale.ENGLISH,"fftXYZ                     : %8.4f\n",timer.read(T_15));
	}
}
