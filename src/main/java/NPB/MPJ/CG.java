/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB-MPJ by Damián Álvarez Mallón
 */
package NPB.MPJ;

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

public class CG implements Benchmark{

	private static final String BM_NAME = "CG";
	private char CLASS;
	private boolean detailedTiming;

	private int numProcCols;
	private int numProcRows;

	private int na, nonzer, niter, nz;
	private double shift, rcond, ratio;
	private double zeta, zetaVerifyValue;
	private int firstCol, firstRow, lastCol, lastRow;
	private int sendStart, sendLen;
	private int exchProc, exchRecvLength;
	private int l2npcols;

	private int reduceExchProc[];
	private int reduceSendStarts[];
	private int reduceSendLengths[];
	private int reduceRecvStarts[];
	private int reduceRecvLengths[];

	private Random rng;
	private Timer timer;
	private BMResults res;

	private double tran = 314159265;
	private final double amult = 1220703125;

	private int colidx[];
	private int acol[];
	private int arow[];
	private int rowstr[];
	private	int iv[];
	private double v[];
	private double aelt[];
	private double a[];
	private double x[];
	private double z[];
	private double p[];
	private double q[];
	private double r[];
	private double w[];

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

	public static void main(String[] args) throws NPBException {
		CG cg = null;

		BMArgs.parseCmdLineArgs(args, BM_NAME);
		try {
			cg = new CG(BMArgs.CLASS, BMArgs.DETAILED_TIMING);
		} catch (OutOfMemoryError e) {
			BMArgs.outOfMemoryMessage();
			throw e;
		}
		cg.runBenchmark(args);
	}

	public BMResults getResults(){
		return res;
	}

	public CG(char classSize, boolean dtiming){

		this.CLASS = classSize;
		this.detailedTiming = dtiming;
	}

	public void build(String[] args) throws NPBException {
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
			na = 1400;
			nonzer = 7;
			shift = 10;
			niter = 15;
			rcond = 0.1;
			zetaVerifyValue = 8.5971775078648;
			break;
		case 'W':
			na = 7000;
			nonzer = 8;
			shift = 12;
			niter = 15;
			rcond = 0.1;
			zetaVerifyValue = 10.362595087124;
			break;
		case 'A':
			na = 14000;
			nonzer = 11;
			shift = 20;
			niter = 15;
			rcond = 0.1;
			zetaVerifyValue = 17.130235054029;
			break;
		case 'B':
			na = 75000;
			nonzer = 13;
			shift = 60;
			niter = 75;
			rcond = 0.1;
			zetaVerifyValue = 22.712745482631;
			break;
		case 'C':
			na = 150000;
			nonzer = 15;
			shift = 110;
			niter = 75;
			rcond = 0.1;
			zetaVerifyValue = 28.973605592845;
			break;
		case 'D':
			na = 1500000;
			nonzer = 21;
			shift = 500;
			niter = 100;
			rcond = 0.1;
			zetaVerifyValue = 52.514532105794;
			break;
		case 'E':
			na = 9000000;
			nonzer = 26;
			shift = 1500;
			niter = 100;
			rcond = 0.1;
			zetaVerifyValue = 77.522164599383;
			break;
		default:
			terminate();
			throw new BadClassException("Specified class (" + CLASS + ") not valid for this benchmark\n"
					+ "Valid classes are S, W, A, B, C, D and E");
		}

		numProcCols = Misc.ilog2(numProcess) / 2;
		numProcRows = Misc.ilog2(numProcess) / 2;
		if (numProcCols + numProcRows != Misc.ilog2(numProcess))
			numProcCols += 1;
		numProcCols = Misc.ipow2(numProcCols);
		numProcRows = Misc.ipow2(numProcRows);

		/*
		 * Dimension of the arrays after the partition, as in NPB-MPI 3.4 (alloc_space):
		 * computed in 64 bits "to avoid integer overflow" (class E has 6.5e9 non-zeros
		 * in total); the arrays of a process must still fit in a Java array.
		 */
		long naz = na;
		naz = naz * (nonzer + 1) / numProcess * (nonzer + 1) + nonzer
				+ naz * (nonzer + 2 + numProcess / 256) / numProcCols;
		if (naz > Integer.MAX_VALUE) {
			terminate();
			throw new IncorrectNumProcessException("Class " + CLASS + " needs more processes: the partition of "
					+ naz + " non-zeros per process is too large for a Java array");
		}
		nz = (int) naz;

		reduceExchProc = new int[numProcCols];
		reduceSendStarts = new int[numProcCols];
		reduceSendLengths = new int[numProcCols];
		reduceRecvStarts = new int[numProcCols];
		reduceRecvLengths = new int[numProcCols];

		iv = new int[2 * na + 1];
		colidx = new int[nz];
		rowstr = new int[na + 1];
		acol = new int[nz];
		arow = new int[nz];
		aelt = new double[nz];
		a = new double[nz];
		v = new double[na + 1];
		x = new double[na / numProcRows + 2];
		z = new double[na / numProcRows + 2];
		p = new double[na / numProcRows + 2];
		q = new double[na / numProcRows + 2];
		r = new double[na / numProcRows + 2];
		w = new double[na / numProcRows + 2];

		ratio = Math.pow(rcond, 1 / (double) na);
		rng = new Random();
	}

	private void terminate() {
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
		
		if (myRank == 0) {
			BMArgs.bannerMPJ(BM_NAME, CLASS, npTotal);
			System.out.println(" Size:  " + na);
			System.out.println(" Iterations:   " + niter);
			System.out.println(" Total number of processes: " + npTotal);
			if (numProcess != npTotal)
				System.out.println(" WARNING: Number of processes is not a power of two (" + numProcess + " active)");
			System.out.println(" Number of nonzeroes per row: " + nonzer);
			System.out.println(" Eigenvalue shift: " + shift);
			System.out.println();
		}

		/* Set up processor info, such as whether sq num of procs, etc */
		setupProcInfo();

		/*
		 * Set up partition's submatrix info: firstcol, lastcol, firstrow,
		 * lastrow
		 */
		setupSubmatrixInfo();

		/* Initialize the random number generator */
		double values[] = { 0, 0 };
		values = rng.vrandlc(tran, amult);
		tran = values[0];
		zeta = values[1];

		/* Set up partition's sparse random matrix for given class size */
		makea();

		/*
		 * Note: as a result of the above call to makea: values of j used in
		 * indexing rowstr go from 1 --> lastrow-firstrow+1 values of colidx
		 * which are col indexes go from firstcol --> lastcol So: Shift the col
		 * index vals from actual (firstcol --> lastcol ) to local, i.e., (1 -->
		 * lastcol-firstcol+1)
		 */

		for (int i = 0; i <= lastRow - firstRow; i++) {
			for (int j = rowstr[i] - 1; j < rowstr[i + 1] - 1; j++) {
				colidx[j] = colidx[j] - firstCol + 1;
			}
		}

		/* set starting vector to (1, 1, .... 1) */
		for (int i = 0; i < na / numProcRows + 1; i++) {
			x[i] = 1;
		}

		zeta = 0;

		/*
		 * Do one iteration untimed to init all code and data page tables (then
		 * reinit, start timing, to niter its)
		 */
		doMainComputation(-1);

		/* set starting vector to (1, 1, .... 1) */
		for (int i = 0; i < na / numProcRows + 1; i++) {
			x[i] = 1;
		}

		zeta = 0;

		/* Synchronize and start timing */
		commWork.Barrier();
		timer.start(T_COMP);

		/* Main iteration for inverse power method */
		for (int i = 1; i <= niter; i++) {
			doMainComputation(i);
		}

		timer.stop(T_COMP);

		double times[] = { timer.read(T_COMP), timer.read(T_COMM) };
		double timesmax[] = { 0, 0 };

		commWork.Reduce(times, 0, timesmax, 0, 2, MPI.DOUBLE, MPI.MAX, 0);

		if (myRank == 0) {
			System.out.printf(" Benchmark completed \n");

			double epsilon = 1.0e-10;
			double err = Math.abs(zeta - zetaVerifyValue) / zetaVerifyValue;

			int verified = 0;
			if (err <= epsilon) {
				verified = 1;
			}

			BMResults.printVerificationStatus(CLASS, verified);
			System.out.println(" Zeta is: " + zeta);
			System.out.println(" Error is: " + err);

			double MOPS = (2 * niter * na)
			* ((3 + (nonzer * (nonzer + 1)) + 25
					* (5 + (nonzer * (nonzer + 1))) + 3) / timesmax[0]) / 1000000;

			res = new BMResults(BM_NAME, CLASS, na, 0, 0, niter,
					timesmax, MOPS, "Floating point", verified, npTotal, numProcess,
					detailedTiming);

			res.printMPJ();
		}
		terminate();
	}

	private void setupProcInfo() throws IncorrectNumProcessException {

		int i = numProcCols;
		while (i != 0) {
			if (i != 1 && i / 2 * 2 != i) {
				terminate();
				throw new IncorrectNumProcessException("Number of processors must be a power of two (1,2,4,...) for this benchmark\n" +
				"Something really bad happens: numProcCols isn't power of two");
			}
			i /= 2;
		}

		i = numProcRows;
		while (i != 0) {
			if (i != 1 && i / 2 * 2 != i) {
				terminate();
				throw new IncorrectNumProcessException("Number of processors must be a power of two (1,2,4,...) for this benchmark\n" +
				"Something really bad happens: numProcRows isn't power of two");
			}
			i /= 2;
		}

		i = numProcess;

		while (i != 0) {
			if (i != 1 && i / 2 * 2 != i) {
				terminate();
				throw new IncorrectNumProcessException("Number of processors must be a power of two (1,2,4,...) for this benchmark\n" +
				"Something really bad happens: numProccess isn't power of two");
			}
			i /= 2;
		}
	}

	private void setupSubmatrixInfo() {

		int colSize, rowSize;

		int procRow = myRank / numProcCols;
		int procCol = myRank - procRow * numProcCols;

		/*
		 * If na evenly divisible by numProcCols, then it is evenly divisible by
		 * numProcRows
		 */
		if (na / numProcCols * numProcCols == na) {
			colSize = na / numProcCols;
			firstCol = procCol * colSize + 1;
			lastCol = firstCol - 1 + colSize;
			rowSize = na / numProcRows;
			firstRow = procRow * rowSize + 1;
			lastRow = firstRow - 1 + rowSize;
		}

		/*
		 * If na not evenly divisible by numProcCols, then first subdivide for
		 * numProcRows and then, if numProcCols not equal to numProcRows (i.e.,
		 * not a sq number of procs), get col subdivisions by dividing by 2 each
		 * row subdivision.
		 */
		else {
			if (procRow < na - na / numProcRows * numProcRows) {
				rowSize = na / numProcRows + 1;
				firstRow = procRow * rowSize + 1;
				lastRow = firstRow - 1 + rowSize;
			} else {
				rowSize = na / numProcRows;
				firstRow = (na - na / numProcRows * numProcRows)
				* (rowSize + 1)
				+ (procRow - (na - na / numProcRows * numProcRows))
				* rowSize + 1;
				lastRow = firstRow - 1 + rowSize;
			}

			if (numProcCols == numProcRows) {
				if (procCol < na - na / numProcCols * numProcCols) {
					colSize = na / numProcCols + 1;
					firstCol = procCol * colSize + 1;
					lastCol = firstCol - 1 + colSize;
				} else {
					colSize = na / numProcCols;
					firstCol = (na - na / numProcCols * numProcCols)
					* (colSize + 1)
					+ (procCol - (na - na / numProcCols * numProcCols))
					* colSize + 1;
					lastCol = firstCol - 1 + colSize;
				}
			} else {
				if ((procCol / 2) < na - na / (numProcCols / 2)
						* (numProcCols / 2)) {
					colSize = na / (numProcCols / 2) + 1;
					firstCol = (procCol / 2) * colSize + 1;
					lastCol = firstCol - 1 + colSize;
				} else {
					colSize = na / (numProcCols / 2);
					firstCol = (na - na / (numProcCols / 2) * (numProcCols / 2))
					* (colSize + 1)
					+ ((procCol / 2) - (na - na / (numProcCols / 2)
							* (numProcCols / 2))) * colSize + 1;
					lastCol = firstCol - 1 + colSize;
				}
				if (myRank % 2 == 0) {
					lastCol = firstCol - 1 + (colSize - 1) / 2 + 1;
				} else {
					firstCol = firstCol + (colSize - 1) / 2 + 1;
					lastCol = firstCol - 1 + colSize / 2;
				}
			}

		}

		if (numProcCols == numProcRows) {
			sendStart = 1;
			sendLen = lastRow - firstRow + 1;
		} else {
			if (myRank % 2 == 0) {
				sendStart = 1;
				sendLen = (1 + lastRow - firstRow + 1) / 2;
			} else {
				sendStart = (1 + lastRow - firstRow + 1) / 2 + 1;
				sendLen = (lastRow - firstRow + 1) / 2;
			}
		}

		/* Transpose exchange processor */
		if (numProcCols == numProcRows) {
			exchProc = (myRank % numProcRows) * numProcRows + myRank
			/ numProcRows;
		} else {
			exchProc = 2
			* (((myRank / 2) % numProcRows) * numProcRows + myRank / 2
					/ numProcRows) + (myRank % 2);
		}

		int i = numProcCols / 2;
		l2npcols = 0;
		while (i > 0) {
			l2npcols++;
			i /= 2;
		}

		/* Set up the reduce phase schedules... */
		int divFactor = numProcCols;
		int j;
		for (i = 0; i < l2npcols; i++) {
			j = ((procCol + divFactor / 2) % divFactor) + procCol / divFactor
			* divFactor;
			reduceExchProc[i] = procRow * numProcCols + j;
			divFactor /= 2;
		}

		for (i = l2npcols - 1; i >= 0; i--) {
			if (numProcRows == numProcCols) {
				reduceSendStarts[i] = sendStart;
				reduceSendLengths[i] = sendLen;
				reduceRecvLengths[i] = lastRow - firstRow + 1;
			} else {
				reduceRecvLengths[i] = sendLen;
				if (i == l2npcols - 1) {
					reduceSendLengths[i] = lastRow - firstRow + 1 - sendLen;
					if (myRank / 2 * 2 == myRank) {
						reduceSendStarts[i] = sendStart + sendLen;
					} else {
						reduceSendStarts[i] = 1;
					}
				} else {
					reduceSendLengths[i] = sendLen;
					reduceSendStarts[i] = sendStart;
				}
			}
			reduceRecvStarts[i] = sendStart;
		}

		exchRecvLength = lastCol - firstCol + 1;
	}

	/*
	 * Generate the test problem for benchmark 6 makea generates a sparse matrix
	 * with a prescribed sparsity distribution
	 */
	private void makea() {
		double size = 1;
		int nnza = 0;

		/*
		 * Initialize iv(na...2na-1) to zero. Used by sprnvc to mark nonzero
		 * positions
		 */
		for (int i = 0; i < na; i++) {
			iv[na + i] = 0;
		}

		for (int i = 0; i < na; i++) {
			sprnvc(nonzer, v, colidx, iv, na);
			int nzv = nonzer;
			nzv = vecset(v, colidx, nzv, i + 1, 0.5);
			for (int j = 0; j < nzv; j++) {
				int jcol = colidx[j];
				if (jcol >= firstCol && jcol <= lastCol) {
					double scale = size * v[j];
					for (int k = 0; k < nzv; k++) {
						if (colidx[k] >= firstRow && colidx[k] <= lastRow) {
							acol[nnza] = jcol;
							arow[nnza] = colidx[k];
							aelt[nnza] = v[k] * scale;
							nnza++;
							if (nnza > nz)
								makeaError(nnza, i);
						}
					}
				}
			}
			size *= ratio;
		}

		/*
		 * ... add the identity * rcond to the generated matrix to bound the
		 * smallest eigenvalue from below by rcond
		 */
		for (int i = firstRow; i <= lastRow; i++) {
			if (i >= (firstCol) && i <= (lastCol)) {
				acol[nnza] = i;
				arow[nnza] = i;
				aelt[nnza] = rcond - shift;
				nnza++;
				if (nnza > nz)
					makeaError(nnza, na + i);
			}
		}

		sparse(a, colidx, rowstr, acol, arow, v, iv, na, nnza);
	}

	private void makeaError(int nnza, int iouter) {
		System.err.println("Space for matrix elements exceeded in makea");
		System.err.println("nnza, nzmax = " + nnza + " " + nz);
		System.err.println("iouter = " + iouter);
	}

	private void sprnvc(int nz, double v[], int iv[], int nzloc[], int mark) {
		int nzv = 0;
		int nzrow = 0;
		int nn1 = 1;

		do {
			nn1 *= 2;
		} while (nn1 < na);

		/* nn1 is the smallest power of two not less than na */
		double vecloc = 0;
		double vecelt = 0;
		double values[];
		int i = 0;

		while (nz > nzv) {
			values = rng.vrandlc(tran, amult);
			tran = values[0];
			vecelt = values[1];

			/* generate an integer between 1 and n in a portable manner */
			values = rng.vrandlc(tran, amult);
			tran = values[0];
			vecloc = values[1];
			i = icnvrt(vecloc, nn1) + 1;

			if (i > na) {
				continue;
			}

			/* was this integer generated already? */
			if (nzloc[mark + i] == 0) {
				nzloc[mark + i] = 1;
				nzloc[nzrow] = i;
				nzrow++;
				v[nzv] = vecelt;
				iv[nzv] = i;
				nzv++;
			}
		}
		for (int j = 0; j < nzrow; j++) {
			i = nzloc[j];
			nzloc[mark + i] = 0;
		}
	}

	/* scale a double precision number x in (0,1) by a power of 2 and chop it */
	private int icnvrt(double x, int ipwr2) {
		return (int) (x * ipwr2);
	}

	private int vecset(double v[], int iv[], int nzv, int i, double val) {

		boolean set = false;

		for (int j = 0; j < nzv; j++) {
			if (iv[j] == i) {
				v[j] = val;
				set = true;
			}
		}
		if (!set) {
			v[nzv] = val;
			iv[nzv] = i;
			nzv++;
		}

		return nzv;
	}

	private void sparse(double a[], int colidx[], int rowstr[], int acol[],
			int arow[], double x[], int mark[], int nzloc, int nnza) {

		/* how many rows of result */
		int nrows = lastRow - firstRow + 1;
		int j = 0, i;

		/* ...count the number of triples in each row */
		for (i = 0; i < na; i++) {
			rowstr[i] = 0;
			mark[i] = 0;
		}
		rowstr[na] = 0;
		for (i = 0; i < nnza; i++) {
			j = arow[i] - firstRow + 1;
			rowstr[j]++;
		}

		rowstr[0] = 1;
		for (i = 1; i < nrows + 1; i++) {
			rowstr[i] += rowstr[i - 1];
		}

		/*
		 * ... rowstr[i] now is the location of the first nonzero of row i of a
		 */

		/* ... do a bucket sort of the triples on the row index */
		for (i = 0; i < nnza; i++) {
			j = arow[i] - firstRow;
			int k = rowstr[j];
			a[k] = aelt[i];
			colidx[k] = acol[i];
			rowstr[j]++;
		}

		/* ... rowstr[j] now points to the first element of row j+1 */
		for (j = nrows - 1; j >= 0; j--) {
			rowstr[j + 1] = rowstr[j];
		}
		rowstr[0] = 1;

		/* ... generate the actual output rows by adding elements */
		int nza = 0;
		for (i = 0; i <= na; i++) {
			x[i] = 0;
			mark[i] = 0;
		}

		int jajp1 = rowstr[0];
		for (j = 0; j < nrows; j++) {
			int nzrow = 0;

			/* ...loop over the jth row of a */
			for (int k = jajp1; k <= rowstr[j + 1] - 1; k++) {
				i = colidx[k];
				x[i] += a[k];
				if (mark[i] == 0 && x[i] != 0) {
					mark[i] = 1;
					mark[nzloc + nzrow + 1] = i;
					nzrow++;
				}
			}

			/* ... extract the nonzeros of this row */
			for (int k = 0; k < nzrow; k++) {
				i = mark[nzloc + k + 1];
				mark[i] = 0;
				double xi = x[i];
				x[i] = 0;
				if (xi != 0) {
					a[nza] = xi;
					colidx[nza] = i;
					nza++;
				}
			}
			jajp1 = rowstr[j + 1];
			rowstr[j + 1] = nza + rowstr[0];
		}
	}

	private void doMainComputation(int iteration) {

		/* The call to the conjugate gradient method */
		double rnorm = conjGrad();
		double normTemp1[] = { 0, 0 };
		double normTemp2[] = { 0, 0 };

		for (int i = 0; i < lastCol - firstCol + 1; i++) {
			normTemp1[0] += x[i] * z[i];
			normTemp1[1] += z[i] * z[i];
		}

		for (int i = 0; i < l2npcols; i++) {

			if (detailedTiming) timer.start(T_COMM);

			Request request = commWork.Irecv(normTemp2, 0, 2, MPI.DOUBLE, reduceExchProc[i], i);
			commWork.Send(normTemp1, 0, 2, MPI.DOUBLE, reduceExchProc[i], i);
			request.Wait();

			if (detailedTiming) timer.stop(T_COMM);

			normTemp1[0] += normTemp2[0];
			normTemp1[1] += normTemp2[1];
		}

		normTemp1[1] = 1 / Math.sqrt(normTemp1[1]);

		if (myRank == 0) {
			zeta = shift + 1 / normTemp1[0];

			if(iteration == 1)
				System.out.println("   iteration            ||r||                 zeta");
			/* NPB 3.4.3+: the norms are printed for the first iteration, every 5 and the last one */
			if(iteration != -1 && (iteration == 1 || iteration == niter || iteration % 5 == 0))
				System.out.println("    " + iteration + "   " + "     " +rnorm+ "     " + "     " + zeta);
		}

		/* Normalize z to obtain x */
		for (int i = 0; i < lastCol - firstCol + 1; i++) {
			x[i] = normTemp1[1] * z[i];
		}
	}

	/*
	 * Works on the fields, not on parameters: with the 15 arrays and counts of the original signature as
	 * arguments C2 ran out of registers and reloaded the bases of a, colidx and p from the stack for every
	 * nonzero of the matrix-vector product (class B, 1 process: 19.2 s; like this, 15.0 s)
	 */
	private double conjGrad() {

		double sum = 0;
		double rho = 0;
		double d = 0;
		double alpha;
		double beta;
		double recvBuff[] = { 0 };
		double sendBuff[] = { 0 };

		/* Initialize the CG algorithm */
		for (int i = 0; i < na / numProcRows + 1; i++) {
			q[i] = 0;
			z[i] = 0;
			w[i] = 0;
			r[i] = x[i];
			p[i] = r[i];
		}

		/*
		 * rho = r.r Now, obtain the norm of r: first, sum squares of r elements
		 * locally...
		 */
		for (int i = 0; i < lastCol - firstCol + 1; i++) {
			sum += r[i] * r[i];
		}

		/*
		 * Exchange and sum with procs identified in reduceExchProc (This is
		 * equivalent to commWork.Allreduce) Sum the partial sums of rho,
		 * leaving rho on all processors
		 */
		for (int i = 0; i < l2npcols; i++) {

			if (detailedTiming) timer.start(T_COMM);

			Request request = commWork.Irecv(recvBuff, 0, 1, MPI.DOUBLE, reduceExchProc[i], i);
			sendBuff[0] = sum;
			commWork.Send(sendBuff, 0, 1, MPI.DOUBLE, reduceExchProc[i], i);
			request.Wait();

			if (detailedTiming) timer.stop(T_COMM);

			sum += recvBuff[0];
		}
		rho = sum;

		/* The conj grad iteration loop */
		int cgitmax = 25;
		for (int cgit = 0; cgit < cgitmax; cgit++) {

			/*
			 * q = A.p The partition submatrix-vector multiply: use workspace w
			 */
			for (int j = 0; j < lastRow - firstRow + 1; j++) {
				sum = 0;
				for (int k = rowstr[j] - 1; k < rowstr[j + 1] - 1; k++) {
					sum += a[k] * p[colidx[k] - 1];
				}
				w[j] = sum;
			}

			/*
			 * Sum the partition submatrix-vec A.p's across rows Exchange and
			 * sum piece of w with procs identified in reduceExchProc
			 */
			for (int j = l2npcols - 1; j >= 0; j--) {

				if (detailedTiming) timer.start(T_COMM);

				Request request = commWork.Irecv(q, reduceRecvStarts[j] - 1, reduceRecvLengths[j], MPI.DOUBLE, reduceExchProc[j], j);
				commWork.Send(w, reduceSendStarts[j] - 1, reduceSendLengths[j], MPI.DOUBLE, reduceExchProc[j], j);
				request.Wait();

				if (detailedTiming) timer.stop(T_COMM);

				for (int k = sendStart - 1; k < sendStart + reduceRecvLengths[j] - 1; k++) {
					w[k] += q[k];
				}
			}

			/* Exchange piece of q with transpose processor */
			if (l2npcols != 0) {

				if (detailedTiming) timer.start(T_COMM);

				Request request = commWork.Irecv(q, 0, exchRecvLength, MPI.DOUBLE, exchProc, 1);
				commWork.Send(w, sendStart - 1, sendLen, MPI.DOUBLE, exchProc, 1);
				request.Wait();

				if (detailedTiming) timer.stop(T_COMM);

			} else {
				for (int j = 0; j < exchRecvLength; j++) {
					q[j] = w[j];
				}
			}

			/* Clear w for reuse... */
			for (int j = 0; j < Math.max(lastRow - firstRow + 1, lastCol - firstCol + 1); j++) {
				w[j] = 0;
			}

			/* Obtain p.q */
			sum = 0;
			for (int j = 0; j < lastCol - firstCol + 1; j++) {
				sum += p[j] * q[j];
			}

			/* Obtain d with a sum-reduce */
			for (int j = 0; j < l2npcols; j++) {

				if (detailedTiming) timer.start(T_COMM);

				Request request = commWork.Irecv(recvBuff, 0, 1, MPI.DOUBLE, reduceExchProc[j], j);
				sendBuff[0] = sum;
				commWork.Send(sendBuff, 0, 1, MPI.DOUBLE, reduceExchProc[j], j);
				request.Wait();

				if (detailedTiming) timer.stop(T_COMM);

				sum += recvBuff[0];
			}
			d = sum;

			/* Obtain alpha = rho / (p.q) */
			alpha = rho / d;

			/* Save a temporary of rho */
			double rho0 = rho;

			/*
			 * Obtain z = z+ alpha*p and r = r - alpha*q
			 */
			for (int j = 0; j < lastCol - firstCol + 1; j++) {
				z[j] += alpha * p[j];
				r[j] -= alpha * q[j];
			}

			/*
			 * rho = r.r Now, obtain the norm of r: First, sum squares of r
			 * elements locally
			 */
			sum = 0;
			for (int j = 0; j < lastCol - firstCol + 1; j++) {
				sum += r[j] * r[j];
			}

			/* Obtain rho with a sum-reduce */
			for (int j = 0; j < l2npcols; j++) {

				if (detailedTiming) timer.start(T_COMM);

				Request request = commWork.Irecv(recvBuff, 0, 1, MPI.DOUBLE, reduceExchProc[j], j);
				sendBuff[0] = sum;
				commWork.Send(sendBuff, 0, 1, MPI.DOUBLE, reduceExchProc[j], j);
				request.Wait();

				if (detailedTiming) timer.stop(T_COMM);

				sum += recvBuff[0];
			}
			rho = sum;

			/* Obtain beta */
			beta = rho / rho0;

			/* p = r + beta*p */
			for (int j = 0; j < lastCol - firstCol + 1; j++) {
				p[j] = r[j] + beta * p[j];
			}
		}

		/*
		 * Compute residual norm explicitly: ||r|| = ||x - A.z|| First, form A.z
		 * The partition submatrix-vector multiply
		 */
		for (int i = 0; i < lastRow - firstRow + 1; i++) {
			sum = 0;
			for (int j = rowstr[i] - 1; j < rowstr[i + 1] - 1; j++) {
				sum += a[j] * z[colidx[j] - 1];
			}
			w[i] = sum;
		}

		/* Sum the partition submatrix-vec A.z's across rows */
		for (int i = l2npcols - 1; i >= 0; i--) {

			if (detailedTiming) timer.start(T_COMM);

			Request request = commWork.Irecv(r, reduceRecvStarts[i] - 1, reduceRecvLengths[i], MPI.DOUBLE, reduceExchProc[i], i);
			commWork.Send(w, reduceSendStarts[i] - 1, reduceSendLengths[i], MPI.DOUBLE, reduceExchProc[i], i);
			request.Wait();

			if (detailedTiming) timer.stop(T_COMM);

			for (int j = sendStart - 1; j < sendStart + reduceRecvLengths[i] - 1; j++) {
				w[j] += r[j];
			}
		}

		/* Exchange piece of q with transpose processor */
		if (l2npcols != 0) {

			if (detailedTiming) timer.start(T_COMM);

			Request request = commWork.Irecv(r, 0, exchRecvLength, MPI.DOUBLE, exchProc, 1);
			commWork.Send(w, sendStart - 1, sendLen, MPI.DOUBLE, exchProc, 1);
			request.Wait();

			if (detailedTiming) timer.stop(T_COMM);

		} else {
			for (int i = 0; i < exchRecvLength; i++) {
				r[i] = w[i];
			}
		}

		/* At this point, r contains A.z */
		sum = 0;
		for (int i = 0; i < lastCol - firstCol + 1; i++) {
			d = x[i] - r[i];
			sum += d * d;
		}

		/* Obtain d with a sum-reduce */
		for (int i = 0; i < l2npcols; i++) {

			if (detailedTiming) timer.start(T_COMM);

			Request request = commWork.Irecv(recvBuff, 0, 1, MPI.DOUBLE, reduceExchProc[i], i);
			sendBuff[0] = sum;
			commWork.Send(sendBuff, 0, 1, MPI.DOUBLE, reduceExchProc[i], i);
			request.Wait();

			if (detailedTiming) timer.stop(T_COMM);

			sum += recvBuff[0];
		}
		d = sum;

		return Math.sqrt(d);
	}
}
