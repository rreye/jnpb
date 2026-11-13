/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB 3.4.4 OpenMP (NPB3.4-OMP/IS)
 */
package NPB.SER;

import java.util.Arrays;

import NPB.Benchmark;
import NPB.common.io.BMArgs;
import NPB.common.io.BMResults;
import NPB.common.numerics.Random;
import NPB.common.timer.Timer;
import NPB.exceptions.BadClassException;
import NPB.exceptions.NPBException;

public class IS implements Benchmark{

	private static final String BM_NAME = "IS";
	private char CLASS;
	private final int MAX_ITERATIONS = 10;
	private final int TEST_ARRAY_SIZE = 5;

	private int TOTAL_KEYS_LOG_2;
	private int MAX_KEY_LOG_2;
	private int NUM_BUCKETS_LOG_2;
	// common variables
	// TOTAL_KEYS and NUM_KEYS are 2^31 in class D: they do not fit in an int (INT_TYPE is long there in NPB)
	private long TOTAL_KEYS;
	private long MAX_KEY;
	private int NUM_BUCKETS;
	private long NUM_KEYS;
	private long SIZE_OF_BUFFERS;

	private long MAX_KEY_DIV4;
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

	dTestIndexArray[] = { 1317351170L, 995930646L, 1157283250L, 1503301535L, 1453734525L },
	dTestRankArray[] = { 1L, 36538729L, 1978098519L, 2145192618L, 2147425337L };

	private static int passedVerification;

	/*
	 * Chunked layout of the three main arrays. Class D has 2^31 keys (and 2^27 possible key values): keyArray and
	 * keyBuff2 are more than a Java array can hold (and their index does not fit in an int), so keyArray, keyBuff1
	 * and keyBuff2 are arrays of chunks of 2^chunkLog ints: element i is in chunk (i >> chunkLog) at offset (i & chunkMask). All the sizes are
	 * powers of two, so every chunk is full, and when the array is smaller than a chunk (classes S to C with the
	 * default chunk size) there is a single chunk of exactly the size of the array. Sequential loops run chunk by
	 * chunk over plain int[]; only the scattered accesses compute (chunk, offset) for each element.
	 *
	 * The keys are < 2^31 and fit in an int, but the ranks do not always: keyBuff1 holds counts and their running
	 * sums, and in class D the last running sum (of the last key value) is TOTAL_KEYS = 2^31. It is stored in an int as 2^31 mod 2^32 (the
	 * additions wrap around exactly) and ranks are read as unsigned (unsignedRank). It is the only such element; the
	 * decrement in fullVerify, which always yields a value <= 2^31-1, is also exact.
	 */
	/** Default chunk size (log2 of the number of ints): 1 GB chunks, so classes S to C use a single chunk. */
	static final int DEFAULT_CHUNK_LOG = 28;
	/** System property that overrides the default chunk size log2 of the default constructor. */
	static final String CHUNK_LOG_PROPERTY = "npb.is.chunklog";
	private final int chunkLog;
	private final int chunkMask;
	private int keyArray[][];
	private int keyBuff1[][];
	private int keyBuff2[][];

	private int partialVerifyVals[];
	private long bucketSize[];
	private long bucketPtrs[];
	private int bucketSizeInt[], bucketPtrsInt[];

	/*
	 * Timers
	 */
	private final int T_0 = 0;
	private final int T_1 = 1;
	private final int T_2 = 2;
	private final int T_3 = 3;

	/*
	 * Some global info
	 */
	private int keyBuffPtrGlobal[][]; /* used by fullVerify to get copies of rank info */

	public BMResults getResults(){
		return res;
	}

	public IS(char classSize, boolean dtiming){
		this(classSize, dtiming, Integer.getInteger(CHUNK_LOG_PROPERTY, DEFAULT_CHUNK_LOG));
	}

	/**
	 * @param chunkLog log2 of the number of ints of a chunk of the three main arrays (see the chunked layout above),
	 *                 between 0 and 30.
	 */
	public IS(char classSize, boolean dtiming, int chunkLog){
		if (chunkLog < 0 || chunkLog > 30)
			throw new IllegalArgumentException("chunkLog must be between 0 and 30: " + chunkLog);
		CLASS = classSize;
		detailedTiming = dtiming;
		this.chunkLog = chunkLog;
		this.chunkMask = (int) ((1L << chunkLog) - 1);
	}

	/** Number of chunks of an array of total ints (a power of two): see the chunked layout above. */
	int chunkCountOf(long total) {
		return (int) Math.max(1, total >> chunkLog);
	}

	/** Length of every chunk of an array of total ints (a power of two). */
	int chunkLengthOf(long total) {
		return (int) Math.min(total, 1L << chunkLog);
	}

	/** NUM_KEYS (valid after build): 2^31 in class D. */
	long getNumKeys() {
		return NUM_KEYS;
	}

	/** MAX_KEY (valid after build): 2^27 in class D. */
	long getMaxKey() {
		return MAX_KEY;
	}

	/** New chunked array of total ints (a power of two): see the chunked layout above. */
	private int[][] newChunked(long total) {
		int n = chunkCountOf(total);
		int len = chunkLengthOf(total);
		int a[][] = new int[n][];
		for (int c = 0; c < n; c++)
			a[c] = new int[len];
		return a;
	}

	private int get(int a[][], long i) {
		return a[(int) (i >> chunkLog)][(int) (i & chunkMask)];
	}

	private void set(int a[][], long i, int v) {
		a[(int) (i >> chunkLog)][(int) (i & chunkMask)] = v;
	}

	/**
	 * In-place running sum of the whole chunked array: a[i] += a[i-1] for every i (the original loop), done chunk by
	 * chunk with a carry. int arithmetic wraps around, which is exact mod 2^32 (see the layout above).
	 */
	static void runningSum(int a[][]) {
		int run = 0;
		for (int c = 0; c < a.length; c++) {
			int chunk[] = a[c];
			for (int j = 0; j < chunk.length; j++) {
				run += chunk[j];
				chunk[j] = run;
			}
		}
	}

	/** A rank stored in an int as an unsigned 32-bit value (class D: the last running sum is 2^31). */
	static long unsignedRank(int stored) {
		return Integer.toUnsignedLong(stored);
	}

	public void build(String[] args) throws NPBException{

		/* Initialize the verification arrays if a valid class */
		switch (CLASS) {
		case 'S':
			testIndexArray = sTestIndexArray;
			testRankArray = sTestRankArray;
			TOTAL_KEYS_LOG_2 = 16;
			MAX_KEY_LOG_2 = 11;
			NUM_BUCKETS_LOG_2 = 9;
			break;
		case 'W':
			testIndexArray = wTestIndexArray;
			testRankArray = wTestRankArray;
			TOTAL_KEYS_LOG_2 = 20;
			MAX_KEY_LOG_2 = 16;
			NUM_BUCKETS_LOG_2 = 10;
			break;
		case 'A':
			testIndexArray = aTestIndexArray;
			testRankArray = aTestRankArray;
			TOTAL_KEYS_LOG_2 = 23;
			MAX_KEY_LOG_2 = 19;
			NUM_BUCKETS_LOG_2 = 10;
			break;
		case 'B':
			testIndexArray = bTestIndexArray;
			testRankArray = bTestRankArray;
			TOTAL_KEYS_LOG_2 = 25;
			MAX_KEY_LOG_2 = 21;
			NUM_BUCKETS_LOG_2 = 10;
			break;
		case 'C':
			testIndexArray = cTestIndexArray;
			testRankArray = cTestRankArray;
			TOTAL_KEYS_LOG_2 = 27;
			MAX_KEY_LOG_2 = 23;
			NUM_BUCKETS_LOG_2 = 10;
			break;
		case 'D':
			testIndexArray = dTestIndexArray;
			testRankArray = dTestRankArray;
			TOTAL_KEYS_LOG_2 = 31;
			MAX_KEY_LOG_2 = 27;
			NUM_BUCKETS_LOG_2 = 10;
			break;
		default:
			terminate();
			throw new BadClassException("Especified class (" + CLASS
					+ ") not valid for this benchmark\n"
					+ "Valid classes are S, W, A, B, C and D");
		}

		TOTAL_KEYS = (1L << TOTAL_KEYS_LOG_2);
		MAX_KEY = (1L << MAX_KEY_LOG_2);
		NUM_BUCKETS = (1 << NUM_BUCKETS_LOG_2);
		NUM_KEYS = TOTAL_KEYS;
		SIZE_OF_BUFFERS = NUM_KEYS;
		MAX_KEY_DIV4 = MAX_KEY / 4;
		SHIFT = MAX_KEY_LOG_2 - NUM_BUCKETS_LOG_2;

		if ((NUM_KEYS >> chunkLog) > Integer.MAX_VALUE - 8)
			throw new IllegalArgumentException("chunkLog " + chunkLog + " is too small for class " + CLASS + ": "
					+ (NUM_KEYS >> chunkLog) + " chunks do not fit in an array");
		rng = new Random();
	}

	/* The big arrays are not allocated in build() so the tests can look at the geometry of class D (24 GB). */
	private void allocate() {
		keyArray = newChunked(SIZE_OF_BUFFERS);
		keyBuff1 = newChunked(MAX_KEY);
		keyBuff2 = newChunked(SIZE_OF_BUFFERS);
		partialVerifyVals = new int[TEST_ARRAY_SIZE];
		bucketSize = new long[NUM_BUCKETS];
		bucketPtrs = new long[NUM_BUCKETS];
		bucketSizeInt = new int[NUM_BUCKETS];
		bucketPtrsInt = new int[NUM_BUCKETS];
	}

	private void terminate() {
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

		for (int c = 0; c < keyArray.length; c++) {
			int chunk[] = keyArray[c];
			for (int i = 0; i < chunk.length; i++) {
				seed = rng.randlc(seed, a);
				x = seed*r46;
				seed = rng.randlc(seed, a);
				x += seed*r46;
				seed = rng.randlc(seed, a);
				x += seed*r46;
				seed = rng.randlc(seed, a);
				x += seed*r46;

				chunk[i] = (int) (x * MAX_KEY_DIV4);
			}
		}
	}

	public void runBenchmark(String[] args) throws NPBException {

		this.build(args);
		allocate();

		timer = new Timer();
		
		if (detailedTiming) timer.start(T_3);

		/*  Printout initial NPB info */
		BMArgs.bannerSER(BM_NAME, CLASS);
		System.out.println(" Size:  " + TOTAL_KEYS+ "  (class "+CLASS+")");
		System.out.println(" Iterations:   " + MAX_ITERATIONS);
		System.out.println();

		if (detailedTiming) timer.start(T_1);
		/* Generate random number sequence and subsequent keys on all procs */
		initKeys(seed, amult);
		if (detailedTiming) timer.stop(T_1);

		rank(1);

		/* Start verification counter */
		passedVerification = 0;

		if (CLASS != 'S')
			System.out.println("   iteration");

		/* Start timers */
		timer.start(T_0);

		/* This is the main iteration */
		for (int iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {
			if (CLASS != 'S')
				System.out.println("        " + iteration);
			rank(iteration);
		}

		/*  End of timing, obtain maximum time of all processors */
		timer.stop(T_0);

		double timecounter[] = { timer.read(T_0), 0.0 };

		/*
		 * This tests that keys are in sequence: sorting of last ranked key seq
		 * occurs here, but is an untimed operation
		 */
		if (detailedTiming) timer.start(T_2);
		fullVerify();
		if (detailedTiming) timer.stop(T_2);

		if (detailedTiming) timer.stop(T_3);

		/* The final printout */
		if (passedVerification == 5 * MAX_ITERATIONS + 1)
			passedVerification = 1;
		else
			passedVerification = 0;

		res = new BMResults(BM_NAME, CLASS, TOTAL_KEYS, 0, 0,
				MAX_ITERATIONS, timecounter, getMOPS(timecounter[0],
						MAX_ITERATIONS, TOTAL_KEYS), "keys ranked",
						passedVerification, 1, detailedTiming);

		res.printSER();

		/*  Print additional timers  */
		if (detailedTiming)
			printAdditionalTimers();

		terminate();
	}

	private void printAdditionalTimers()
	{
		double t_total, t_percent;

		t_total = timer.read(T_3);
		System.out.format("\nAdditional timers -\n");
		System.out.format(java.util.Locale.ENGLISH, " Total execution: %8.4f\n", t_total);

		if (t_total == 0.0) t_total = 1.0;

		double timecounter = timer.read(T_1);
		t_percent = timecounter/t_total * 100.;
		System.out.format(java.util.Locale.ENGLISH, " Initialization : %8.4f (%5.2f%%)\n", timecounter, t_percent);
		timecounter = timer.read(T_0);
		t_percent = timecounter/t_total * 100.;
		System.out.format(java.util.Locale.ENGLISH, " Benchmarking   : %8.4f (%5.2f%%)\n", timecounter, t_percent);
		timecounter = timer.read(T_2);
		t_percent = timecounter/t_total * 100.;
		System.out.format(java.util.Locale.ENGLISH, " Sorting        : %8.4f (%5.2f%%)\n", timecounter, t_percent);
	}

	private double getMOPS(double totalTime, int niter, long numKeys) {
		double mops = 0.0;
		if (totalTime > 0) {
			/* Official NPB: 1e-6 * TOTAL_KEYS * MAX_ITERATIONS / time */
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

	private void rank(int iteration) {

		int key;
		int keyBuffPtr[][];
		int i;

		set(keyArray, iteration, iteration);
		set(keyArray, iteration + MAX_ITERATIONS, (int) (MAX_KEY - iteration));

		/*
		 * Determine where the partial verify test keys are, load into top of
		 * array bucketSize
		 */
		for (i = 0; i < TEST_ARRAY_SIZE; i++)
			partialVerifyVals[i] = get(keyArray, testIndexArray[i]);

		/* Initialize */
		for (i = 0; i < NUM_BUCKETS; i++)
			bucketSize[i] = 0;

		/* Determine the number of keys in each bucket */
		final int bp[] = bucketPtrsInt;
		if (keyBuff2.length == 1) {
			/* Fast path: fewer than 2^31 keys, int counters */
			final int bs[] = bucketSizeInt;
			Arrays.fill(bs, 0);
			for (int c = 0; c < keyArray.length; c++)
				countKeys(keyArray[c], keyArray[c].length, SHIFT, bs);
			bp[0] = 0;
			for (i = 1; i < NUM_BUCKETS; i++)
				bp[i] = bp[i - 1] + bs[i - 1];
			for (i = 0; i < NUM_BUCKETS; i++)
				bucketSize[i] = bs[i];
		} else {
		for (int c = 0; c < keyArray.length; c++) {
			int chunk[] = keyArray[c];
			for (i = 0; i < chunk.length; i++)
				bucketSize[chunk[i] >> SHIFT]++;
		}
		}

		/* Accumulative bucket sizes are the bucket pointers */
		bucketPtrs[0] = 0;
		for (i = 1; i < NUM_BUCKETS; i++)
			bucketPtrs[i] = bucketPtrs[i - 1] + bucketSize[i - 1];

		/* Sort into appropriate bucket */
		if (keyBuff2.length == 1) {
			/* Fast path (classes S to C with the default chunk size): the destination is a plain int[] */
			int out[] = keyBuff2[0];
			for (int c = 0; c < keyArray.length; c++)
				scatterKeys(keyArray[c], keyArray[c].length, SHIFT, bp, out);
		} else {
			for (int c = 0; c < keyArray.length; c++) {
				int chunk[] = keyArray[c];
				for (i = 0; i < chunk.length; i++) {
					key = chunk[i];
					set(keyBuff2, bucketPtrs[key >> SHIFT]++, key);
				}
			}
		}

		/* Clear the work array */
		for (int c = 0; c < keyBuff1.length; c++)
			Arrays.fill(keyBuff1[c], 0);

		/* Ranking of all keys occurs in this section */
		keyBuffPtr = keyBuff1;

		/*
		 * In this section, the keys themselves are used as their own indexes to
		 * determine how many of each there are: their individual population
		 */
		if (keyBuffPtr.length == 1) {
			/* Fast path: the population array is a single plain int[] (also for class D) */
			int pop[] = keyBuffPtr[0];
			/* Now they have individual key population */
			for (int c = 0; c < keyBuff2.length; c++)
				keyPopulation(keyBuff2[c], keyBuff2[c].length, 0, pop);
		} else {
			for (int c = 0; c < keyBuff2.length; c++) {
				int chunk[] = keyBuff2[c];
				for (i = 0; i < chunk.length; i++) {
					/* Now they have individual key population */
					key = chunk[i];
					keyBuffPtr[key >> chunkLog][key & chunkMask]++;
				}
			}
		}

		/*
		 * To obtain ranks of each key, successively add the individual key
		 * population
		 */
		runningSum(keyBuffPtr);

		parcialVerify(0, NUM_KEYS-1, iteration, keyBuffPtr);

		/*
		 * Make copies of rank info for use by fullVerify: these variables in
		 * rank are local; making them global slows down the code, probably
		 * since they cannot be made register by compiler
		 */
		if (iteration == MAX_ITERATIONS)
			keyBuffPtrGlobal = keyBuffPtr;
	}

	private void parcialVerify(long minKeyVal, long maxKeyVal, int iteration, int keyBuffPtr[][]) {
		/*
		 * This is the partial verify test section
		 * Observe that testRankArray vals are shifted differently for different
		 * cases
		 */

		for (int i = 0; i < TEST_ARRAY_SIZE; i++) {

			int k = partialVerifyVals[i];	/* test vals were put here */

			if (minKeyVal < k && k <= maxKeyVal)
			{
				long keyRank = unsignedRank(get(keyBuffPtr, k-1));
				int failed = 0;

				switch (CLASS) {

				case 'S':
					if (i <= 2) {
						if (keyRank != testRankArray[i] + iteration)
							failed = 1;
						else
							passedVerification++;
					} else {
						if (keyRank != testRankArray[i] - iteration)
							failed = 1;
						else
							passedVerification++;
					}
					break;
				case 'W':
					if (i < 2) {
						if (keyRank != testRankArray[i] + (iteration-2))
							failed = 1;
						else
							passedVerification++;
					} else {
						if (keyRank != testRankArray[i] - iteration)
							failed = 1;
						else
							passedVerification++;
					}
					break;
				case 'A':
					if (i <= 2) {
						if (keyRank != testRankArray[i] + (iteration-1))
							failed = 1;
						else
							passedVerification++;
					} else {
						if (keyRank != testRankArray[i] - (iteration-1))
							failed = 1;
						else
							passedVerification++;
					}
					break;
				case 'B':
					if (i == 1 || i == 2 || i == 4) {
						if (keyRank != testRankArray[i] + iteration)
							failed = 1;
						else
							passedVerification++;
					} else {
						if (keyRank != testRankArray[i] - iteration)
							failed = 1;
						else
							passedVerification++;
					}
					break;
				case 'C':
					if (i <= 2) {
						if (keyRank != testRankArray[i] + iteration)
							failed = 1;
						else
							passedVerification++;
					} else {
						if (keyRank != testRankArray[i] - iteration)
							failed = 1;
						else
							passedVerification++;
					}
					break;
				case 'D':
					if (i < 2) {
						if (keyRank != testRankArray[i] + iteration)
							failed = 1;
						else
							passedVerification++;
					} else {
						if (keyRank != testRankArray[i] - iteration)
							failed = 1;
						else
							passedVerification++;
					}
					break;
				}

				if(failed == 1)
				{
					System.out.println("Failed partial verification: "
							+ "iteration " + iteration
							+ ", test key " + i);
				}
			}
		}
	}

	private void fullVerify() {
		int i;
		long j = 0;

		/*
		 * Now, finally, sort the keys. The decremented rank is always a valid index (<= 2^31-1, see the chunked
		 * layout above), so it is a non-negative int even when the stored value was 2^31 (as a negative int).
		 */
		for (int c = 0; c < keyBuff2.length; c++) {
			int chunk[] = keyBuff2[c];
			for (i = 0; i < chunk.length; i++) {
				int key = chunk[i];
				int rank = --keyBuffPtrGlobal[key >> chunkLog][key & chunkMask];
				keyArray[rank >> chunkLog][rank & chunkMask] = key;
			}
		}

		/* Confirm keys correctly sorted: count incorrectly sorted keys, if any */
		int prev = keyArray[0][0];
		for (int c = 0; c < keyArray.length; c++) {
			int chunk[] = keyArray[c];
			for (i = (c == 0 ? 1 : 0); i < chunk.length; i++) {
				if (prev > chunk[i])
					j++;
				prev = chunk[i];
			}
		}

		/*
		 * Not in NPB-OMP: digest of the sorted key array (number of keys and sums of k, k^2 and k^3 in 64-bit
		 * arithmetic, exact). The verification only looks at 5 ranks per iteration plus the order of the array, so
		 * this lets the result be compared with the official one (scripts/gen_golden.py). Untimed, like fullVerify.
		 */
		long count = 0, sum1 = 0, sum2 = 0, sum3 = 0;
		for (int c = 0; c < keyArray.length; c++) {
			int chunk[] = keyArray[c];
			for (i = 0; i < chunk.length; i++) {
				long key = chunk[i];
				count++;
				sum1 += key;
				sum2 += key * key;
				sum3 += key * key * key;
			}
		}
		System.out.println(" Sorted keys digest = " + Long.toUnsignedString(count) + " " + Long.toUnsignedString(sum1)
				+ " " + Long.toUnsignedString(sum2) + " " + Long.toUnsignedString(sum3));

		if (j != 0) {
			System.out.println("fullVerify: number of keys out of sort: " + j);
		} else
			passedVerification++;
	}
}
