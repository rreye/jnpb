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
import NPB.common.numerics.Random;
import NPB.common.timer.Timer;
import NPB.exceptions.BadClassException;
import NPB.exceptions.IncorrectNumProcessException;
import NPB.exceptions.NPBException;

public class IS implements Benchmark{

	private static final String BM_NAME = "IS";
	private char CLASS;
	private final int MAX_ITERATIONS = 10;
	private final int TEST_ARRAY_SIZE = 5;
	/* 128 for class S, 1024 otherwise (as in NPB-MPI); set in build() */
	private int MAX_PROCS;

	private int TOTAL_KEYS_LOG_2;
	private int MAX_KEY_LOG_2;
	private int NUM_BUCKETS_LOG_2;
	/*
	 * Defining MIN_PROCS is to avoid integer overflow for large problem sizes
	 * without using a larger integer type for the keys. The actual total keys =
	 * TOTAL_KEYS * MIN_PROCS (as in NPB-MPI 3.4).
	 */
	private int MIN_PROCS;
	// common variables
	private int TOTAL_KEYS;
	private long MAX_KEY; /* 2^31 for class E: does not fit in an int */
	private int NUM_BUCKETS;
	private int NUM_KEYS;
	private int SIZE_OF_BUFFERS;

	private int MAX_KEY_DIV4;
	private int SHIFT;
	private Timer timer;
	private Random rng;
	private BMResults res;
	private static final double seed = 314159265.0;
	private static final double amult = 1220703125.0;

	private boolean detailedTiming = false;

	private static long testIndexArray[], testRankArray[];
	private static long sTestIndexArray[] = { 48427, 17148, 23627, 62548, 4431 },
	sTestRankArray[] = { 0, 18, 346, 64917, 65463 },

	wTestIndexArray[] = { 357773, 934767, 875723, 898999, 404505 },
	wTestRankArray[] = { 1249, 11698, 1039987, 1043896, 1048018 },

	aTestIndexArray[] = { 2112377, 662041, 5336171, 3642833, 4250760 },
	aTestRankArray[] = { 104, 17523, 123928, 8288932, 8388264 },

	bTestIndexArray[] = { 41869, 812306, 5102857, 18232239, 26860214 },
	bTestRankArray[] = { 33422937, 10244, 59149, 33135281, 99 },

	cTestIndexArray[] = { 44172927, 72999161, 74326391, 129606274, 21736814 },
	cTestRankArray[] = { 61147, 882988, 266290, 133997595, 133525895 },

	dTestIndexArray[] = { 1317351170, 995930646, 1157283250, 1503301535, 1453734525 },
	dTestRankArray[] = { 1, 36538729, 1978098519, 2145192618, 2147425337 },

	eTestIndexArray[] = { 21492309536L, 24606226181L, 12608530949L, 4065943607L, 3324513396L },
	eTestRankArray[] = { 3L, 27580354L, 3248475153L, 30048754302L, 31485259697L };

	private static int passedVerification;

	/*
	 * These are the three main arrays.
	 */
	private int keyArray[];
	private int keyBuff1[];
	private int keyBuff2[];

	private int recvBuff[];
	private int bucketSize[];
	private int bucketSizeTotals[];
	private int bucketPtrs[];
	private int processBucketDistribPtr1[];
	private int processBucketDistribPtr2[];
	private int sendDispl[];
	private int sendCount[];
	private int recvDispl[];
	private int recvCount[];

	/*
	 * Different timers
	 */
	private final int T_COMP = 0;
	private final int T_COMM = 1;

	/*
	 * MPJ variables
	 */
	private int myRank;
	private int numProcess; /* active processes: comm_size in NPB-MPI (a power of two) */
	private int npTotal; /* processes launched: np_total in NPB-MPI */
	private boolean active = true; /* false for the excess ranks masked out when npTotal is not a power of two */
	private Intracomm commWork; /* communicator of the active processes */

	/*
	 * Some global info
	 */
	private int keyBuffPtrGlobal[], /* used by fullVerify to get copies of rank info */
	totalLocalKeys,
	totalLesserKeys;

	private int offset;

	public BMResults getResults(){
		return res;
	}

	public IS(char classSize, boolean dtiming){

		CLASS = classSize;
		detailedTiming = dtiming;
	}

	public void build(String[] args) throws NPBException{

		MPI.Init(args);

		myRank = MPI.COMM_WORLD.Rank();
		npTotal = MPI.COMM_WORLD.Size();

		/* Initialize the verification arrays if a valid class */
		switch (CLASS) {
		case 'S':
			testIndexArray = sTestIndexArray;
			testRankArray = sTestRankArray;
			TOTAL_KEYS_LOG_2 = 16;
			MAX_KEY_LOG_2 = 11;
			NUM_BUCKETS_LOG_2 = 9;
			MIN_PROCS = 1;
			break;
		case 'W':
			testIndexArray = wTestIndexArray;
			testRankArray = wTestRankArray;
			TOTAL_KEYS_LOG_2 = 20;
			MAX_KEY_LOG_2 = 16;
			NUM_BUCKETS_LOG_2 = 10;
			MIN_PROCS = 1;
			break;
		case 'A':
			testIndexArray = aTestIndexArray;
			testRankArray = aTestRankArray;
			TOTAL_KEYS_LOG_2 = 23;
			MAX_KEY_LOG_2 = 19;
			NUM_BUCKETS_LOG_2 = 10;
			MIN_PROCS = 1;
			break;
		case 'B':
			testIndexArray = bTestIndexArray;
			testRankArray = bTestRankArray;
			TOTAL_KEYS_LOG_2 = 25;
			MAX_KEY_LOG_2 = 21;
			NUM_BUCKETS_LOG_2 = 10;
			MIN_PROCS = 1;
			break;
		case 'C':
			testIndexArray = cTestIndexArray;
			testRankArray = cTestRankArray;
			TOTAL_KEYS_LOG_2 = 27;
			MAX_KEY_LOG_2 = 23;
			NUM_BUCKETS_LOG_2 = 10;
			MIN_PROCS = 1;
			break;
		case 'D':
			testIndexArray = dTestIndexArray;
			testRankArray = dTestRankArray;
			TOTAL_KEYS_LOG_2 = 29; /* 2^31 keys in total: see MIN_PROCS */
			MAX_KEY_LOG_2 = 27;
			NUM_BUCKETS_LOG_2 = 10;
			MIN_PROCS = 4;
			break;
		case 'E':
			testIndexArray = eTestIndexArray;
			testRankArray = eTestRankArray;
			TOTAL_KEYS_LOG_2 = 29; /* 2^35 keys in total: see MIN_PROCS */
			MAX_KEY_LOG_2 = 31;
			NUM_BUCKETS_LOG_2 = 10;
			MIN_PROCS = 64;
			break;
		default:
			terminate();
			throw new BadClassException("Especified class (" + CLASS
					+ ") not valid for this benchmark\n"
					+ "Valid classes are S, W, A, B, C, D and E");
		}

		/*
		 * Check whether the total number of processes is within bounds (NPB-MPI
		 * 3.4: MAX_PROCS is 128 for class S and 1024 otherwise; MIN_PROCS depends
		 * on the class).
		 */
		MAX_PROCS = (CLASS == 'S') ? 128 : 1024;
		if (npTotal < MIN_PROCS || npTotal > MAX_PROCS) {
			terminate();
			throw new IncorrectNumProcessException("Number of processes " + npTotal
					+ " not within range " + MIN_PROCS + "-" + MAX_PROCS + " for class " + CLASS
					+ " of this benchmark");
		}

		/* The number of active processes needs to be a power of two */
		for (numProcess = 1; numProcess < npTotal; numProcess *= 2)
			;
		if (numProcess > npTotal)
			numProcess /= 2;

		/*
		 * If the actual number of processes doesn't agree with the active ones,
		 * check if the excess ranks need to be masked: the enforcement is turned
		 * off by setting the environment variable NPB_NPROCS_STRICT to (0, off,
		 * no, false), as in NPB-MPI 3.4.
		 */
		active = true;
		commWork = MPI.COMM_WORLD;
		if (numProcess != npTotal) {
			int strict[] = { 1 };
			if (myRank == 0) {
				String ep = System.getenv("NPB_NPROCS_STRICT");
				if (ep != null && ep.length() > 0) {
					if ("nNfF-".indexOf(ep.charAt(0)) >= 0 || ep.equals("0"))
						strict[0] = 0;
					else if (ep.equals("off") || ep.equals("OFF"))
						strict[0] = 0;
				}
			}
			MPI.COMM_WORLD.Bcast(strict, 0, 1, MPI.INT, 0);

			/* abort if a strict NPROCS enforcement is required */
			if (strict[0] != 0) {
				terminate();
				throw new IncorrectNumProcessException("Number of processes (" + npTotal
						+ ") is not a power of two (" + numProcess + "?)");
			}

			/* mark excess ranks as inactive */
			active = (myRank < numProcess);
			commWork = MPI.COMM_WORLD.Split(active ? 1 : 0, myRank);
		}

		if (!active)
			return; /* nothing to allocate: runBenchmark() finalizes MPI */

		TOTAL_KEYS = (1 << TOTAL_KEYS_LOG_2);
		MAX_KEY = (1L << MAX_KEY_LOG_2);
		NUM_BUCKETS = (1 << NUM_BUCKETS_LOG_2);
		/* problem size after partition */
		NUM_KEYS = (TOTAL_KEYS / numProcess) * MIN_PROCS;

		/* buffer size for communication (long arithmetic: 13 * NUM_KEYS may exceed an int before the division) */
		if (numProcess < 256) {
			SIZE_OF_BUFFERS = (int) (3L * NUM_KEYS / 2);
		} else if (numProcess < 512) {
			SIZE_OF_BUFFERS = (int) (5L * NUM_KEYS / 2);
		} else if (numProcess < 1024) {
			SIZE_OF_BUFFERS = (int) (4L * NUM_KEYS);
		} else
			SIZE_OF_BUFFERS = (int) (13L * NUM_KEYS / 2);

		MAX_KEY_DIV4 = (int) (MAX_KEY / 4);
		SHIFT = MAX_KEY_LOG_2 - NUM_BUCKETS_LOG_2;

		keyArray = new int[SIZE_OF_BUFFERS];
		keyBuff1 = new int[SIZE_OF_BUFFERS];
		keyBuff2 = new int[SIZE_OF_BUFFERS];
		bucketSize = new int[NUM_BUCKETS + TEST_ARRAY_SIZE];
		bucketSizeTotals = new int[NUM_BUCKETS + TEST_ARRAY_SIZE];
		bucketPtrs = new int[NUM_BUCKETS];
		processBucketDistribPtr1 = new int[NUM_BUCKETS + TEST_ARRAY_SIZE];
		processBucketDistribPtr2 = new int[NUM_BUCKETS + TEST_ARRAY_SIZE];
		sendDispl = new int[numProcess];
		sendCount = new int[numProcess];
		recvDispl = new int[numProcess];
		recvCount = new int[numProcess];
		recvBuff = new int[1];

		rng = new Random();
	}

	private void terminate() {
		MPI.Finalize();
	}

	public static void main(String args[]) throws NPBException {
		IS is = null;

		BMArgs.parseCmdLineArgs(args, BM_NAME);
		try {
			is = new IS(BMArgs.CLASS, BMArgs.DETAILED_TIMING);
		} catch (OutOfMemoryError e) {
			BMArgs.outOfMemoryMessage();
			throw e;
		}
		is.runBenchmark(args);
	}

	private void initKeys(double seed, double a) {
		double x;
		double r46 = Random.r46;

		for (int i = 0; i < NUM_KEYS; i++) {
			seed = rng.randlc(seed, a);
			x = seed*r46;
			seed = rng.randlc(seed, a);
			x += seed*r46;
			seed = rng.randlc(seed, a);
			x += seed*r46;
			seed = rng.randlc(seed, a);
			x += seed*r46;

			keyArray[i] = (int) (x * MAX_KEY_DIV4);
		}
	}

	private double findMySeed() {
		long i;
		double t1, t2, an;
		long mq, nq, kk, ik;
		double values[] = {0,0};

		nq = (4L * TOTAL_KEYS * MIN_PROCS) / numProcess;

		for (mq = 0; nq > 1; mq++, nq /= 2)
			;

		t1 = amult;

		for (i = 1; i <= mq; i++) {
			values = rng.vrandlc(t1, t1);
			t1 = values[0];
			t2 = values[1];
		}

		an = t1;
		kk = myRank;
		t1 = seed;
		t2 = an;

		for (i = 1; i <= 100; i++) {
			ik = kk / 2;

			if (2 * ik != kk) {
				values = rng.vrandlc(t1, t2);
				t1 = values[0];
			}
			if (ik == 0)
				break;
			values = rng.vrandlc(t2, t2);
			t2 = values[0];
			kk = ik;
		}

		return (t1);
	}

	public void runBenchmark(String[] args) throws NPBException {

		this.build(args);

		/* excess ranks masked out (NPB_NPROCS_STRICT off): nothing to do */
		if (!active) {
			terminate();
			return;
		}

		timer = new Timer();
		
		/*  Printout initial NPB info */
		if (myRank == 0) {

			BMArgs.bannerMPJ(BM_NAME, CLASS, npTotal);
			System.out.println(" Size:  " + (long) TOTAL_KEYS * MIN_PROCS + "  (class "+CLASS+")");
			System.out.println(" Iterations:   " + MAX_ITERATIONS);
			System.out.println(" Total number of processes:   " + npTotal);
			if (numProcess != npTotal)
				System.out.println(" WARNING: Number of processes is not a power of two (" + numProcess + " active)");
			System.out.println();
		}

		/* Generate random number sequence and subsequent keys on all procs */
		initKeys(findMySeed(), amult);

		rank(1);

		/* Start verification counter */
		passedVerification = 0;

		if (myRank == 0 && CLASS != 'S')
			System.out.println("   iteration");

		/* Start timer */
		timer.start(T_COMP);

		/* This is the main iteration */
		for (int iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {
			if (myRank == 0 && CLASS != 'S')
				System.out.println("        " + iteration);
			rank(iteration);
		}

		/* Stop timer, obtain time for processors */
		timer.stop(T_COMP);

		double timecounter[] = new double[2];
		timecounter[0] = timer.read(T_COMP);
		timecounter[1] = timer.read(T_COMM);
		double maxtimes[] = { 0, 0 };

		/* End of timing, obtain maximum time of all processors */
		commWork.Reduce(timecounter, 0, maxtimes, 0, 2, MPI.DOUBLE, MPI.MAX, 0);

		/*
		 * This tests that keys are in sequence: sorting of last ranked key seq
		 * occurs here, but is an untimed operation
		 */
		fullVerify();

		/* Obtain verification counter sum */
		int itemp[] = { passedVerification };
		int passedVerificationPtr[] = { passedVerification };
		commWork.Reduce(itemp, 0, passedVerificationPtr, 0, 1, MPI.INT, MPI.SUM, 0);
		passedVerification = passedVerificationPtr[0];

		/* The final printout */
		if (myRank == 0) {
			if (passedVerification == 5 * MAX_ITERATIONS + numProcess)
				passedVerification = 1;
			else
				passedVerification = 0;

			res = new BMResults(BM_NAME, CLASS, (long) TOTAL_KEYS * MIN_PROCS, 0, 0,
					MAX_ITERATIONS, maxtimes, getMOPS(maxtimes[0],
							MAX_ITERATIONS, (long) TOTAL_KEYS * MIN_PROCS), "keys ranked",
							passedVerification, npTotal, numProcess, detailedTiming);

			res.printMPJ();
		}
		terminate();
	}

	private double getMOPS(double totalTime, int niter, long numKeys) {
		double mops = 0.0;
		if (totalTime > 0) {
			/* Official NPB-MPI: MAX_ITERATIONS * TOTAL_KEYS * MIN_PROCS / maxtime / 1e6
			 * (numKeys is already TOTAL_KEYS * MIN_PROCS). */
			mops = (double) niter * numKeys;
			mops /= totalTime * 1000000.0;
		}
		return mops;
	}

	/* Hot loops of rank() as small static methods: each one gets its own compilation */
	private static void countKeys(int keys[], int n, int shift, int count[]) {
		for (int i = 0; i < n; i++)
			count[keys[i] >> shift]++;
	}

	private static void scatterKeys(int keys[], int n, int shift, int ptrs[], int out[]) {
		for (int i = 0; i < n; i++) {
			int key = keys[i];
			out[ptrs[key >> shift]++] = key;
		}
	}

	private static void keyPopulation(int keys[], int n, int minKey, int pop[]) {
		for (int i = 0; i < n; i++)
			pop[keys[i] - minKey]++;
	}

	private static void runningSum(int a[], int n) {
		int run = 0;
		for (int i = 0; i < n; i++) {
			run += a[i];
			a[i] = run;
		}
	}

	private void rank(int iteration) {

		int key;
		long bucketSumAccumulator; /* KEY_TYPE in NPB-MPI: long for class D */
		int localBucketSumAccumulator;
		int minKeyVal, maxKeyVal;
		int keyBuffPtr[];
		int i, j, k;
		long m, totalLocal; /* KEY_TYPE in NPB-MPI: long for class D */

		/* Iteration alteration of keys */
		if (myRank == 0) {
			keyArray[iteration] = iteration;
			keyArray[iteration + MAX_ITERATIONS] = (int) (MAX_KEY - iteration);
		}

		/* Initialize */
		for (i = 0; i < NUM_BUCKETS + TEST_ARRAY_SIZE; i++) {
			bucketSize[i] = 0;
			bucketSizeTotals[i] = 0;
			processBucketDistribPtr1[i] = 0;
			processBucketDistribPtr2[i] = 0;
		}

		/*
		 * Determine where the partial verify test keys are, load into top of
		 * array bucketSize
		 */
		for (i = 0; i < TEST_ARRAY_SIZE; i++)
			if ((testIndexArray[i] / NUM_KEYS) == myRank)
				bucketSize[NUM_BUCKETS + i] = keyArray[(int) (testIndexArray[i] % NUM_KEYS)];

		/* Determine the number of keys in each bucket */
		countKeys(keyArray, NUM_KEYS, SHIFT, bucketSize);

		/* Accumulative bucket sizes are the bucket pointers */
		bucketPtrs[0] = 0;
		for (i = 1; i < NUM_BUCKETS; i++)
			bucketPtrs[i] = bucketPtrs[i - 1] + bucketSize[i - 1];

		/* Sort into appropriate bucket */
		scatterKeys(keyArray, NUM_KEYS, SHIFT, bucketPtrs, keyBuff1);

		if (detailedTiming) timer.start(T_COMM);
		/*
		 * Get the bucket size totals for the entire problem. These will be used
		 * to determine the redistribution of keys
		 */
		commWork.Allreduce(bucketSize, 0, bucketSizeTotals, 0, NUM_BUCKETS + TEST_ARRAY_SIZE, MPI.INT, MPI.SUM);

		if (detailedTiming) timer.stop(T_COMM);

		/*
		 * Determine Redistibution of keys: accumulate the bucket size totals
		 * till this number surpasses NUM_KEYS (which the average number of keys
		 * per processor). Then all keys in these buckets go to processor 0.
		 * Continue accumulating again until supassing 2*NUM_KEYS. All keys in
		 * these buckets go to processor 1, etc. This algorithm guarantees that
		 * all processors have work ranking; no processors are left idle. The
		 * optimum number of buckets, however, does not result in as high a
		 * degree of load balancing (as even a distribution of keys as is
		 * possible) as is obtained from increasing the number of buckets, but
		 * more buckets results in more computation per processor so that the
		 * optimum number of buckets turns out to be 1024 for machines tested.
		 * Note that processBucketDistribPtr1 and ...Ptr2 hold the bucket number
		 * of first and last bucket which each processor will have after the
		 * redistribution is done.
		 */

		bucketSumAccumulator = 0;
		localBucketSumAccumulator = 0;
		sendDispl[0] = 0;
		processBucketDistribPtr1[0] = 0;

		for (i = 0, j = 0; i < NUM_BUCKETS; i++) {
			bucketSumAccumulator += bucketSizeTotals[i];
			localBucketSumAccumulator += bucketSize[i];

			if (bucketSumAccumulator >= ((long) j + 1) * NUM_KEYS) {
				sendCount[j] = localBucketSumAccumulator;
				if (j != 0) {
					sendDispl[j] = sendDispl[j - 1] + sendCount[j - 1];
					processBucketDistribPtr1[j] = processBucketDistribPtr2[j - 1] + 1;
				}
				processBucketDistribPtr2[j++] = i;
				localBucketSumAccumulator = 0;
			}
		}

		/*  When NUM_PROCS approaching NUM_BUCKETS, it is highly possible
		 * that the last few processors don't get any buckets.  So, we 
		 * need to set counts properly in this case to avoid any fallouts.
		 */
		while( j < numProcess )
		{
			sendCount[j] = 0;
			processBucketDistribPtr1[j] = 1;
			j++;
		}

		if (detailedTiming) timer.start(T_COMM);
		/*
		 * This is the redistribution section: first find out how many keys each
		 * processor will send to every other processor:
		 */
		commWork.Alltoall(sendCount, 0, 1, MPI.INT, recvCount, 0, 1, MPI.INT);

		/* Determine the receive array displacements for the buckets */
		recvDispl[0] = 0;
		for (i = 1; i < numProcess; i++) {
			recvDispl[i] = recvDispl[i - 1] + recvCount[i - 1];
		}

		/* Now send the keys to respective processors */
		commWork.Alltoallv(keyBuff1, 0, sendCount, sendDispl, MPI.INT, keyBuff2, 0, recvCount, recvDispl, MPI.INT);

		if (detailedTiming) timer.stop(T_COMM);

		/*
		 * The starting and ending bucket numbers on each processor are
		 * multiplied by the interval size of the buckets to obtain the smallest
		 * possible min and greatest possible max value of any key on each
		 * processor
		 */
		minKeyVal = processBucketDistribPtr1[myRank] << SHIFT;
		maxKeyVal = ((processBucketDistribPtr2[myRank] + 1) << SHIFT) - 1;

		/* Clear the work array */
		java.util.Arrays.fill(keyBuff1, 0, maxKeyVal - minKeyVal + 1, 0);

		/*
		 * Determine the total number of keys on all other processors holding
		 * keys of lesser value
		 */
		m = 0;
		for (k = 0; k < myRank; k++)
			for (i = processBucketDistribPtr1[k]; i <= processBucketDistribPtr2[k]; i++)
				m += bucketSizeTotals[i]; /* m has total # of lesser keys */

		/* Determine total number of keys on this processor */
		totalLocal = 0;
		for (i = processBucketDistribPtr1[myRank]; i <= processBucketDistribPtr2[myRank]; i++)
			totalLocal += bucketSizeTotals[i]; /* totalLocal has total # of local keys */

		/* Ranking of all keys occurs in this section */
		keyBuffPtr = keyBuff1;

		/*
		 * In this section, the keys themselves are used as their own indexes to
		 * determine how many of each there are: their individual population
		 */
		/* Now they have individual key population */
		keyPopulation(keyBuff2, (int) totalLocal, minKeyVal, keyBuffPtr);

		/* To obtain ranks of each key, successively add the individual key
		 * population, not forgetting the total of lesser keys, m.
		 * NOTE: Since the total of lesser keys would be subtracted later 
		 * in verification, it is no longer added to the first key population 
		 * here, but still needed during the partial verify test.  This is to 
		 * ensure that 32-bit key_buff can still be used for class D.
		 */
		//keyBuffPtr[minKeyVal - minKeyVal] += m;
		runningSum(keyBuffPtr, maxKeyVal - minKeyVal + 1);

		parcialVerify(minKeyVal, maxKeyVal, iteration, keyBuffPtr, m);

		/*
		 * Make copies of rank info for use by fullVerify: these variables in
		 * rank are local; making them global slows down the code, probably
		 * since they cannot be made register by compiler
		 */
		if (iteration == MAX_ITERATIONS) {
			keyBuffPtrGlobal = keyBuffPtr;
			offset = minKeyVal;
			totalLocalKeys = (int) totalLocal;
			totalLesserKeys = 0;	/* no longer set to 'm', see note above */
		}

	}

	private void parcialVerify(int minKeyVal, int maxKeyVal, int iteration, int keyBuffPtr[], long totalLesserKeys) {
		/*
		 * Observe that testRankArray vals are shifted differently for different
		 * cases
		 */
		for (int i = 0; i < TEST_ARRAY_SIZE; i++) {

			int k = bucketSizeTotals[i + NUM_BUCKETS]; /* Keys were hidden here */

			if (minKeyVal <= k && k <= maxKeyVal)
			{
				/* Add the total of lesser keys here */
				long keyRank = keyBuffPtr[k-1-minKeyVal] + totalLesserKeys;
				long testRank = testRankArray[i];
				int failed = 0;

				switch (CLASS) {
				case 'S':
					if (i <= 2)
						testRank += iteration;
					else
						testRank -= iteration;
					break;
				case 'W':
					if (i < 2)
						testRank += iteration - 2;
					else
						testRank -= iteration;
					break;
				case 'A':
					if (i <= 2)
						testRank += iteration - 1;
					else
						testRank -= iteration - 1;
					break;
				case 'B':
					if (i == 1 || i == 2 || i == 4)
						testRank += iteration;
					else
						testRank -= iteration;
					break;
				case 'C':
					if (i <= 2)
						testRank += iteration;
					else
						testRank -= iteration;
					break;
				case 'D':
					if (i < 2)
						testRank += iteration;
					else
						testRank -= iteration;
					break;
				case 'E':
					if (i < 2)
						testRank += iteration - 2;
					else if (i == 2) {
						testRank += iteration - 2;
						if (iteration > 4)
							testRank -= 2;
						else if (iteration > 2)
							testRank -= 1;
					} else
						testRank -= iteration - 2;
					break;
				}

				if (keyRank != testRank)
					failed = 1;
				else
					passedVerification++;

				if(failed == 1)
				{
					System.out.println("Failed partial verification: "
							+ "iteration " + iteration + ", processor "
							+ myRank + ", test key " + i + ", key rank " + keyRank);
				}
			}
		}
	}

	private void fullVerify() {

		Request request = null;
		int lastLocalKey;
		int i, j = 0;

		/* Now, finally, sort the keys: */
		for (i = 0; i < totalLocalKeys; i++) {
			keyArray[--keyBuffPtrGlobal[keyBuff2[i] - offset] - totalLesserKeys] = keyBuff2[i];
		}

		lastLocalKey = (totalLocalKeys<1)? 0 : (totalLocalKeys-1);

		/* Send largest key value to next processor */
		if (myRank > 0) {
			request = commWork.Irecv(recvBuff, 0, 1, MPI.INT, myRank - 1, 1000);
		}
		if (myRank < numProcess - 1) {
			commWork.Send(keyArray, lastLocalKey, 1, MPI.INT, myRank + 1, 1000);
		}
		if (myRank > 0) {
			request.Wait();
		}

		/*
		 * Confirm that neighbor's greatest key value is not greater than my
		 * least key value
		 */
		if (myRank > 0 && totalLocalKeys > 0)
			if (recvBuff[0] > keyArray[0])
				j++;

		/* Confirm keys correctly sorted: count incorrectly sorted keys, if any */
		for (i = 1; i < totalLocalKeys; i++)
			if (keyArray[i-1] > keyArray[i])
				j++;

		/*
		 * Not in NPB-MPI: digest of the globally sorted key array (number of keys and sums of k, k^2 and k^3
		 * in 64-bit arithmetic, exact and independent of the order of the reduction). The partial
		 * verification only looks at 5 ranks per iteration, so this lets the result be compared with the
		 * official one (scripts/gen_golden_mpi.py). Untimed, like the rest of fullVerify.
		 */
		long digestLocal[] = new long[4], digest[] = new long[4];
		for (i = 0; i < totalLocalKeys; i++) {
			long key = keyArray[i];
			digestLocal[0]++;
			digestLocal[1] += key;
			digestLocal[2] += key * key;
			digestLocal[3] += key * key * key;
		}
		commWork.Reduce(digestLocal, 0, digest, 0, 4, MPI.LONG, MPI.SUM, 0);
		if (myRank == 0)
			System.out.println(" Sorted keys digest = " + Long.toUnsignedString(digest[0]) + " "
					+ Long.toUnsignedString(digest[1]) + " " + Long.toUnsignedString(digest[2]) + " "
					+ Long.toUnsignedString(digest[3]));

		if (j != 0) {
			System.out.println("Processor " + myRank + ": fullVerify: number of keys out of sort: " + j);
		} else
			passedVerification++;
	}
}
