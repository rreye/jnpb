/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB 3.4.4 OpenMP (NPB3.4-OMP/CG)
 */
package NPB.SER;

import NPB.Benchmark;
import NPB.common.io.BMArgs;
import NPB.common.io.BMResults;
import NPB.common.numerics.Random;
import NPB.common.timer.Timer;
import NPB.exceptions.BadClassException;
import NPB.exceptions.NPBException;

public class CG implements Benchmark{

	private static final String BM_NAME = "CG";
	private char CLASS;
	private boolean detailedTiming;

	private int na, nonzer, niter, nz, naz;
	private double shift, rcond, ratio;
	private double zeta, zetaVerifyValue;
	private int firstCol, firstRow, lastCol, lastRow;

	private Random rng;
	private Timer timer;
	private BMResults res;

	private double tran = 314159265;
	private final double amult = 1220703125;

	/* Marks a free slot in colidx while sparse() builds the matrix (columns are 0-based) */
	private static final int EMPTY = -1;

	private int colidx[];
	private int acol[];
	private int arow[];
	private int rowstr[];
	private	int iv[];
	private int ivc[];
	private double aelt[];
	private double a[];
	private double x[];
	private double z[];
	private double p[];
	private double q[];
	private double r[];
	private double vc[];

	/*
	 * Different timers
	 */
	private final int T_INIT = 0;
	private final int T_BENCH = 1;
	private final int T_CG = 2;

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
		default:
			throw new BadClassException("Specified class (" + CLASS + ") not valid for this benchmark\n"
					+ "Valid classes are S, W, A, B, C and D");
		}

		nz = na * (nonzer + 1) * (nonzer + 1);
		naz = na * (nonzer + 1);

		colidx = new int[nz];
		rowstr = new int[na + 1];
		iv = new int[na];
		arow = new int[na];
		acol = new int[naz];
		aelt = new double[naz];
		a = new double[nz];
		x = new double[na + 2];
		z = new double[na + 2];
		p = new double[na + 2];
		q = new double[na + 2];
		r = new double[na + 2];		
		ivc = new int[nonzer + 1];
		vc = new double[nonzer + 1];

		ratio = Math.pow(rcond, 1 / (double) na);
		rng = new Random();
	}

	private void terminate() {
	}

	public void runBenchmark(String[] args) throws NPBException {

		int i,j,k;

		this.build(args);

		timer = new Timer();
		
		timer.start(T_INIT);

		firstRow = 1;
		lastRow  = na;
		firstCol = 1;
		lastCol  = na;
		
		/*  Printout initial NPB info */
		BMArgs.bannerSER(BM_NAME, CLASS);
		System.out.println(" Size:  " + na);
		System.out.println(" Iterations:   " + niter);

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

		for (j = 0; j <= lastRow - firstRow; j++) {
			for (k = rowstr[j] - 1; k < rowstr[j + 1] - 1; k++)
				colidx[k] = colidx[k] - firstCol + 1;
		}

		/* set starting vector to (1, 1, .... 1) */
		for (i = 0; i < na + 1; i++)
			x[i] = 1.0;

		for (i = 0; i <= lastCol - firstCol; i++) {
			q[i] = 0.0;
			z[i] = 0.0;
			r[i] = 0.0;
			p[i] = 0.0;
		}

		zeta = 0;

		/*
		 * Do one iteration untimed to init all code and data page tables (then
		 * reinit, start timing, to niter its)
		 */
		doMainComputation(-1);

		/* set starting vector to (1, 1, .... 1) */
		for (i = 0; i < na + 1; i++)
			x[i] = 1.0;

		zeta = 0;

		timer.stop(T_INIT);

		System.out.format(java.util.Locale.ENGLISH, "\n Initialization time = %8.3f seconds\n\n", timer.read(T_INIT));

		timer.start(T_BENCH);

		/* Main iteration for inverse power method */
		for (i = 1; i <= niter; i++) {
			doMainComputation(i);
		}

		timer.stop(T_BENCH);

		double times[] = { timer.read(T_BENCH), 0.0 };

		System.out.printf(" Benchmark completed \n");

		double epsilon = 1.0e-10;
		double err = Math.abs(zeta - zetaVerifyValue) / zetaVerifyValue;

		int verified = 0;
		if (err <= epsilon)
			verified = 1;

		BMResults.printVerificationStatus(CLASS, verified);
		System.out.println(" Zeta is " + zeta);
		System.out.println(" Error is " + err);

		double MOPS = (2 * niter * na) * ((3 + (nonzer * (nonzer + 1)) + 25 * (5 + (nonzer * (nonzer + 1))) + 3) / times[0]) / 1000000;

		res = new BMResults(BM_NAME, CLASS, na, 0, 0, niter,
				times, MOPS, "floating point", verified, 1,
				detailedTiming);

		res.printSER();

		/*  Print additional timers  */
		if (detailedTiming)
			printAdditionalTimers();

		terminate();
	}

	private void printAdditionalTimers()
	{
		double tmax = timer.read(T_BENCH);
		double timecounter, t_percent;

		if (tmax == 0.0) tmax = 1.0;

		System.out.format("\nSECTION\tTime (secs)\n");
		timecounter = timer.read(T_INIT);
		System.out.format(java.util.Locale.ENGLISH, " init: %8.4f\n", timecounter);
		timecounter = timer.read(T_BENCH);
		t_percent = timecounter/tmax * 100.;
		System.out.format(java.util.Locale.ENGLISH, " benchmk: %8.4f (%5.2f%%)\n", timecounter, t_percent);
		timecounter = timer.read(T_CG);
		t_percent = timecounter/tmax * 100.;
		System.out.format(java.util.Locale.ENGLISH, " conjgd: %8.4f (%5.2f%%)\n", timecounter, t_percent);
	}

	/*
	 * This method organizes the memory in the same way that fortran does,
	 * to facilitate the code translation and efficiency.
	 */
	private int position2D(int x, int y, int dimx, int dimy) {
		return x - 1 + (y - 1) * dimx;
	}

	/*
	 * Generate the test problem for benchmark 6 makea generates a sparse matrix
	 * with a prescribed sparsity distribution
	 */
	private void makea() {

		int nzv, iouter, ivelt, pos, nn1 = 1;

		/* nn1 is the smallest power of two not less than na */
		do {
			nn1 *= 2;
		} while (nn1 < na);

		for (iouter = 1; iouter <= na; iouter++) {
			nzv = nonzer;
			sprnvc(na, nzv, nn1, vc, ivc);
			nzv = vecset(na, vc, ivc, nzv, iouter, 0.5);
			arow[iouter-1] = nzv;

			pos = position2D(1, iouter, nonzer+1, na);
			for (ivelt = 1; ivelt <= nzv; ivelt++) {
				acol[pos] = ivc[ivelt-1];
				aelt[pos] = vc[ivelt-1];
				pos++;
			}
		}

		/*
		 * ... make the sparse matrix from list of elements with duplicates
		 *     (iv is used as  workspace)
		 */
		sparse(a, colidx, rowstr, arow, acol, aelt, iv);
	}

	private void sprnvc(int n, int nz, int nn1, double v[], int iv[]) {
		int nzv = 0;
		double vecloc = 0;
		double vecelt = 0;
		double values[];
		int i, ii;

		while (nz > nzv) {
			values = rng.vrandlc(tran, amult);
			tran = values[0];
			vecelt = values[1];

			/* generate an integer between 1 and n in a portable manner */
			values = rng.vrandlc(tran, amult);
			tran = values[0];
			vecloc = values[1];
			i = icnvrt(vecloc, nn1) + 1;

			if (i > n)
				continue;

			/* was this integer generated already? */
			boolean repeated = false;
			for (ii = 0; ii < nzv; ii++) {
				if (iv[ii] == i) {
					repeated = true;
					break;
				}
			}
			if (repeated)
				continue;

			v[nzv] = vecelt;
			iv[nzv] = i;
			nzv++;
		}
	}

	/* scale a double precision number x in (0,1) by a power of 2 and chop it */
	private int icnvrt(double x, int ipwr2) {
		return (int) (x * ipwr2);
	}

	private int vecset(int n, double v[], int iv[], int nzv, int i, double val) {

		boolean set = false;

		for (int k = 0; k < nzv; k++) {
			if (iv[k] == i) {
				v[k] = val;
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

	private void sparseError(int nza) {
		System.err.println("Space for matrix elements exceeded in sparse");
		System.err.println("nza, nzmax = " + nza + " " + nz);
		System.exit(-1);
	}

	private void sparse(double a[], int colidx[], int rowstr[], int arow[], int acol[], double aelt[], int[] nzloc) {

		int index = nonzer+1;
		double size = 1.0;
		double scale,va;
		int i,j,j1,j2,k,kk,nzrow,nza,jcol,pos,pos2;

		/* how many rows of result */
		int nrows = lastRow - firstRow + 1;

		/* ...count the number of triples in each row */
		for (j = 0; j < nrows + 1; j++)
			rowstr[j] = 0;

		for (i = 1; i <= na ; i++) {
			pos = position2D(1, i, nonzer+1, na);
			for (nza = 1; nza <= arow[i-1] ; nza++) {
				j = acol[pos];
				rowstr[j] += arow[i-1];
				pos++;
			}
		}

		rowstr[0] = 1;
		for (j = 1; j < nrows + 1; j++) {
			rowstr[j] += rowstr[j - 1];
		}

		nza = rowstr[nrows] - 1;

		/*
		 * ... rowstr[i] now is the location of the first nonzero of row i of a
		 */
		if (nza > nz)
			sparseError(nza);

		/* ... preload data pages */
		for (j = 1; j <= nrows; j++) {
			for (k = rowstr[j-1]; k <= rowstr[j]-1; k++) {
				a[k-1] = 0.0;
				colidx[k-1] = EMPTY;
			}
			nzloc[j-1] = 0;
		}

		boolean error = true;

		/* ... generate the actual output rows by summing duplicates */
		for (i = 1; i <= na; i++) {
			pos = position2D(1, i, nonzer+1, na);
			for (nza = 1; nza <= arow[i-1] ; nza++) {
				j = acol[pos]-1;
				scale = size * aelt[pos];

				pos2 = position2D(1, i, nonzer+1, na);
				for (nzrow = 1; nzrow <= arow[i-1]; nzrow++) {
					jcol = acol[pos2]-1;
					va = aelt[pos2] * scale;

					/*
					 * ... add the identity * rcond to the generated matrix to bound
					 * 	   the smallest eigenvalue from below by rcond
					 */
					if (jcol == j && j == i - 1)
						va = va + rcond - shift;

					error = true;

					for (k = rowstr[j]; k <= rowstr[j+1]-1; k++)
					{
						if (colidx[k-1] > jcol)
						{
							/* ... insert colidx here orderly */
							for (kk = rowstr[j+1]-2; kk >= k ; kk--) {
								if (colidx[kk-1] != EMPTY)
								{
									a[kk]  = a[kk-1];
									colidx[kk] = colidx[kk-1];
								}
							}
							colidx[k-1] = jcol;
							a[k-1]  = 0.0;
							error = false;
							break;
						}
						else if (colidx[k-1] == EMPTY)
						{							
							colidx[k-1] = jcol;
							error = false;
							break;
						}
						else if (colidx[k-1] == jcol)
						{
							/* ... mark the duplicated entry */
							nzloc[j] = nzloc[j] + 1;
							error = false;
							break;
						}
					}//end for k
					if(error)
					{
						System.err.println("internal error in sparse: i="+i);
						System.exit(-1);
					}
					a[k-1] = a[k-1] + va;
					pos2++;
				}//end for nzrow
				pos++;
			}//end for nza
			size = size * ratio;
		}//end for i

		/* ... remove empty entries and generate final results */
		for (j = 1; j < nrows ; j++)
			nzloc[j] = nzloc[j] + nzloc[j-1];

		for (j = 0; j < nrows ; j++)
		{
			if (j > 0)
				j1 = rowstr[j] - nzloc[j-1] - 1;
			else
				j1 = 0;

			j2 = rowstr[j+1] - nzloc[j] - 2;
			nza = rowstr[j] - 1;

			for (k = j1; k <= j2 ; k++) {
				a[k] = a[nza];
				colidx[k] = colidx[nza];
				nza = nza + 1;	            		
			}
		}

		for (j = 2; j <= nrows + 1; j++)
			rowstr[j-1] = rowstr[j-1] - nzloc[j-2];

		nza = rowstr[nrows] - 1;
	}

	private void doMainComputation(int iteration) {

		if (detailedTiming) timer.start(T_CG);

		/* The call to the conjugate gradient method */
		double rnorm = conjGrad();

		if (detailedTiming) timer.stop(T_CG);

		double normTemp1 = 0.0;
		double normTemp2 = 0.0;

		for (int i = 0; i < lastCol - firstCol + 1; i++) {
			normTemp1 += x[i] * z[i];
			normTemp2 += z[i] * z[i];
		}

		normTemp2 = 1 / Math.sqrt(normTemp2);

		zeta = shift + 1 / normTemp1;

		if(iteration == 1)
			System.out.println("   iteration            ||r||                 zeta");
		if(iteration != -1)
			System.out.println("    " + iteration + "   " + "     " +rnorm+ "     " + "     " + zeta);

		/* Normalize z to obtain x */
		for (int i = 0; i < lastCol - firstCol + 1; i++)
			x[i] = normTemp2 * z[i];
	}

	/*
	 * Works on the fields, not on parameters: with the arrays as arguments C2 can run out of registers and
	 * reload the bases of a, colidx and p from the stack for every nonzero of the matrix-vector product
	 * (seen in the MPJ version: class B, 19.2 s instead of 15.0 s)
	 */
	private double conjGrad() {

		double sum = 0;
		double rho = 0;
		double d = 0;
		double alpha;
		double beta;
		int cgitmax = 25;

		/* Initialize the CG algorithm */
		for (int i = 0; i < na + 1; i++) {
			q[i] = 0;
			z[i] = 0;
			r[i] = x[i];
			p[i] = r[i];
		}

		/*
		 * rho = r.r Now, obtain the norm of r: first, sum squares of r elements
		 * locally...
		 */
		for (int i = 0; i < lastCol - firstCol + 1; i++) {
			rho += r[i] * r[i];
		}

		/* The conj grad iteration loop */
		for (int cgit = 0; cgit < cgitmax; cgit++) {

			/*
			 * q = A.p The partition submatrix-vector multiply: use workspace w
			 */
			for (int j = 0; j < lastRow - firstRow + 1; j++) {
				sum = 0;
				for (int k = rowstr[j] - 1; k < rowstr[j + 1] - 1; k++) {
					sum += a[k] * p[colidx[k]];
				}
				q[j] = sum;
			}

			/* Obtain p.q */
			sum = 0;
			for (int j = 0; j < lastCol - firstCol + 1; j++) {
				sum += p[j] * q[j];
			}

			/* Obtain alpha = rho / (p.q) */
			alpha = rho / sum;

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
			rho = 0;
			for (int j = 0; j < lastCol - firstCol + 1; j++) {
				rho += r[j] * r[j];
			}

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
				sum += a[j] * z[colidx[j]];
			}
			r[i] = sum;
		}

		/* At this point, r contains A.z */
		sum = 0;
		for (int i = 0; i < lastCol - firstCol + 1; i++) {
			d = x[i] - r[i];
			sum += d * d;
		}

		return Math.sqrt(sum);
	}
}
