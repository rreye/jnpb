/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB-MPJ by Damián Álvarez Mallón
 */
package NPB.MPJ;

import NPB.Benchmark;
import NPB.common.io.BMArgs;
import NPB.common.io.BMResults;
import NPB.common.numerics.Random;
import NPB.common.timer.Timer;
import NPB.exceptions.BadClassException;
import NPB.exceptions.IncorrectNumProcessException;
import NPB.exceptions.NPBException;

import mpi.MPI;

public class EP implements Benchmark{

	private static final String BM_NAME = "EP";
	private char CLASS;
	private boolean detailedTiming;

	private int m;
	private long size;
	private final int mk = 16;
	private final int nk = 65536; //2^mk);
	private final int nq = 10;
	private final double epsilon = 1.0e-8;
	private final double amult = 1220703125.0;
	private final double s = 271828183.0;
	private int nn;
	private double dum[] = { 1, 1, 1 };
	private double x[] = new double[2 * nk];
	private double q[] = new double[nq];
	/* t1 of the accepted pairs of each batch (see runBenchmark) */
	private double tt[] = new double[nk];

	private double sxVerifyValue;
	private double gcVerifyValue;
	private double syVerifyValue;

	private Random rng;
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
	private int numProcess;

	public static void main(String[] args) throws NPBException {
		EP ep = null;

		BMArgs.parseCmdLineArgs(args, BM_NAME);
		try {
			ep = new EP(BMArgs.CLASS, BMArgs.DETAILED_TIMING);
		} catch (OutOfMemoryError e) {
			BMArgs.outOfMemoryMessage();
			throw e;
		}
		ep.runBenchmark(args);
	}

	public BMResults getResults(){
		return res;
	}

	public EP(char classSize, boolean dtiming) {

		detailedTiming = dtiming;
		CLASS = classSize;
	}

	public void build(String[] args) throws NPBException {

		MPI.Init(args);

		myRank = MPI.COMM_WORLD.Rank();
		numProcess = MPI.COMM_WORLD.Size();

		switch (CLASS) {
		case 'S':
			m = 24;
			sxVerifyValue = 1.051299420395306E+07;
			syVerifyValue = 1.051517131857535E+07;
			gcVerifyValue = 13176389.0;
			break;
		case 'W':
			m = 25;
			sxVerifyValue = 2.102505525182392E+07;
			syVerifyValue = 2.103162209578822E+07;
			gcVerifyValue = 26354769.0;
			break;
		case 'A':
			m = 28;
			sxVerifyValue = 1.682235632304711E+08;
			syVerifyValue = 1.682195123368299E+08;
			gcVerifyValue = 210832767.0;
			break;
		case 'B':
			m = 30;
			sxVerifyValue = 6.728927543423024E+08;
			syVerifyValue = 6.728951822504275E+08;
			gcVerifyValue = 843345606.0;
			break;
		case 'C':
			m = 32;
			sxVerifyValue = 2.691444083862931E+09;
			syVerifyValue = 2.691519118724585E+09;
			gcVerifyValue = 3373275903.0;
			break;
		case 'D':
			m = 36;
			sxVerifyValue = 4.306350280812112E+10;
			syVerifyValue = 4.306347571859157E+10;
			gcVerifyValue = 53972171957.0;
			break;
		case 'E':
			m = 40;
			sxVerifyValue = 6.890169663167274E+11;
			syVerifyValue = 6.890164670688535E+11;
			gcVerifyValue = 863554308186.0;
			break;
		default:
			terminate();
			throw new BadClassException("Specified class (" + CLASS + ") not valid for this benchmark\n"
					+ "Valid classes are S, W, A, B, C, D and E");
		}

		int mm = m - mk;
		//nn = (int) Math.pow(2, mm);
		nn = 2 << (mm - 1);
		rng = new Random();
	}

	private void terminate() {
		MPI.Finalize();
	}

	public void runBenchmark(String[] args) throws NPBException {

		this.build(args);
		
		timer = new Timer();
		
		int npAdd;

		if (myRank == 0) {
			BMArgs.bannerMPJ(BM_NAME, CLASS, numProcess);
			size = (long)Math.pow(2.0, m + 1);
			System.out.println(" Number of random numbers generated:  "+size);
			System.out.println(" Number of active processes:  "+numProcess);
			System.out.println();
		}

		/*
		 * Compute the number of "batches" of random number pairs generated per
		 * processor. Adjust if the number of processors does not evenly divide
		 * the total number
		 */

		int np = nn / numProcess;
		int numLargeNodes = nn % numProcess;
		double t1, t2, t3, t4, x1, x2, an, gc, sx, sy;
		int kOffset, ik, kk, l, i, k;
		double recvbuff[] = new double[3];

		if (myRank < numLargeNodes)
			npAdd = 1;
		else
			npAdd = 0;

		np += npAdd;

		if (np == 0) {
			terminate();
			throw new IncorrectNumProcessException("Too many processes ("+numProcess+","+nn+")");
		}

		/*
		 * Call the random number generator functions and initialize the x-array
		 * to reduce the effects of paging on the timings
		 */
		dum[0] = rng.vranlc(0, dum[0], dum[1], dum, 2);
		double values[] = rng.vrandlc(dum[1], dum[2]);
		dum[0] = values[1];
		dum[1] = values[0];

		for (i = 0; i < 2 * nk; i++) {
			x[i] = -1.e99;
		}

		/* Synchronize before placing time stamp */
		MPI.COMM_WORLD.Barrier();	
		timer.start(T_COMP);

		t1 = amult;
		t2 = 0;
		t1 = rng.vranlc(0, t1, amult, x, 0);

		/* Compute AN = A ^ (2 * NK) (mod 2^46) */
		for (i = 0; i < mk + 1; i++) {
			values = rng.vrandlc(t1, t1);
			t1 = values[0];
			t2 = values[1];
		}

		an = t1;
		gc = sx = sy = 0;

		/*
		 * Each instance of this loop may be performed independently. We compute
		 * the k offsets separately to take into account the fact that some
		 * nodes have more numbers to generate than others
		 */
		if (npAdd == 1) {
			kOffset = myRank * np - 1;
		} else {
			kOffset = numLargeNodes * (np + 1) + (myRank - numLargeNodes) * np - 1;
		}

		for (k = 0; k < np; k++) {
			kk = kOffset + k + 1;
			t1 = s;
			t2 = an;

			/* Find starting seed t1 for this kk */
			for (i = 0; i < 100; i++) {
				ik = kk >> 1;

			if(ik << 1 != kk) {
				values = rng.vrandlc(t1, t2);
				t1 = values[0];
				t3 = values[1];
			}
			if (ik == 0) {
				break;
			}
			values = rng.vrandlc(t2, t2);
			t2 = values[0];
			t3 = values[1];
			kk = ik;
			}

			/* Compute uniform pseudorandom numbers */
			t1 = rng.vranlc(2 * nk, t1, amult, x, 0);

			/*
			 * Compute Gaussian deviates by acceptance-rejection method and
			 * tally counts in concentric square annuli. This loop is not
			 * vectorizable
			 */
			/*
			 * Optimization (deviates from the structure of the original;
			 * bit-identical results): the loop is split in two.
			 * (1) Filtering: the accepted pairs (t1 <= 1) are compacted in
			 * place in x[] (n <= i, so nothing unread is overwritten) and
			 * their t1 goes into tt[].
			 * (2) Tally of the accepted pairs only, in the same order (the
			 * sums sx and sy do not change), without the unpredictable
			 * rejection branch.
			 * (int) max(t3, t4) == max((int) t3, (int) t4) because t3, t4 >= 0
			 * (or both NaN -> 0) and truncation is monotonic; the integer max
			 * avoids the -0.0/NaN handling of Math.max(double, double). Do not
			 * use "t3 > t4 ? t3 : t4": it is an unpredictable branch (1.67x
			 * slower).
			 */
			int n = 0;
			for (i = 0; i < nk; i++) {
				x1 = 2.0 * x[2 * i] - 1.0;
				x2 = 2.0 * x[2 * i + 1] - 1.0;
				t1 = x1 * x1 + x2 * x2;
				x[2 * n] = x1;
				x[2 * n + 1] = x2;
				tt[n] = t1;
				if (t1 <= 1.0) n++;
			}
			for (i = 0; i < n; i++) {
				t1 = tt[i];
				t2 = Math.sqrt(-2.0 * Math.log(t1) / t1);
				/* NPB 3.4.2+ scheme: sums of absolute values */
				t3 = Math.abs(x[2 * i] * t2);
				t4 = Math.abs(x[2 * i + 1] * t2);
				l = Math.max((int) t3, (int) t4);
				q[l]++;
				sx += t3;
				sy += t4;
			}
		}

		double sendbuff[] = { sx, sy, 0 };

		if (detailedTiming) timer.start(T_COMM);

		MPI.COMM_WORLD.Allreduce(sendbuff, 0, recvbuff, 0, 2, MPI.DOUBLE, MPI.SUM);
		MPI.COMM_WORLD.Allreduce(q, 0, x, 0, nq, MPI.DOUBLE, MPI.SUM);

		if (detailedTiming) timer.stop(T_COMM);

		sx = recvbuff[0];
		sy = recvbuff[1];

		for (i = 0; i < nq; i++) {
			q[i] = x[i];
			gc += q[i];
		}

		timer.stop(T_COMP);

		double times[] = { timer.read(T_COMP), timer.read(T_COMM) };
		double maxtimes[] = { 0, 0 };

		MPI.COMM_WORLD.Reduce(times, 0, maxtimes, 0, 2, MPI.DOUBLE, MPI.MAX, 0);

		if (myRank == 0) {

			int verified = 0;

			if (m == 24 || m == 25 || m == 28 || m == 30 || m == 32 || m == 36 || m == 40) {
				double sxErr = Math.abs((sx - sxVerifyValue) / sxVerifyValue);
				double syErr = Math.abs((sy - syVerifyValue) / syVerifyValue);
				double gcErr = Math.abs((gc - gcVerifyValue) / gcVerifyValue);

				/* Written so that a NaN error never counts as verified */
				if ((sxErr <= epsilon) && (syErr <= epsilon) && (gcErr <= epsilon)) {
					verified = 1;
				} else
					verified = 0;
			}

			double MOPS = Math.pow(2, m + 1) / maxtimes[0] / 1000000;

			System.out.println("EP Benchmark Results:");
			System.out.println();
			System.out.format(java.util.Locale.ENGLISH,"CPU Time = %8.4f\n",maxtimes[0]);
			System.out.println("N = 2^" + m);
			System.out.format(java.util.Locale.ENGLISH,"No. Gaussian Pairs =\t%10.0f.\n",gc);
			System.out.format(java.util.Locale.ENGLISH,"Sums =\t" + sx + "\t" + sy+ "\n");
			System.out.println("Counts:");

			for (i = 0; i < nq; i++)
				System.out.format(java.util.Locale.ENGLISH,"  %d\t%10.0f.\n",i,q[i]);

			res = new BMResults(BM_NAME, CLASS, size, 0, 0, 1,
					maxtimes, MOPS, "Random numbers generated", verified,
					numProcess, detailedTiming);

			res.printMPJ();
		}
		terminate();
	}
}
