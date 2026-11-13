package NPB.SER;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import NPB.Benchmark;

/**
 * Chunking of keyArray, keyBuff1 and keyBuff2 of SER IS, needed for class D (2^31 keys, more than fits in a
 * Java array). Pure refactor: with any chunk size (2^chunkLog ints) the digest of the sorted array
 * ({@code Sorted keys digest}, which is exact) and the verification result must be identical to those of a
 * single chunk, which in turn are those of the original unchunked code (checked by hand when the refactor was done
 * and against the {@link GoldenCheck} golden values in the matrix).
 * <ul>
 * <li>{@link #chunkedRunsAreIdenticalToSingleChunk}: S, W and A with chunks of 1, 8 and 32 ints and some larger ones.</li>
 * <li>{@link #classDGeometry}: class D needs ~17 GB and is not run in the tests, but {@code build()} does not
 * allocate the arrays: their geometry is checked (sizes, number of chunks, coverage of the last index).</li>
 * <li>{@link #runningSumWrapsExactlyAt2To31}: the last running sum of keyBuff1 in D is 2^31, which an int stores
 * as 2^31 mod 2^32 and is read as unsigned; checked with hand-made arrays, across several chunks.</li>
 * <li>{@link #seededChunkingBugsAreDetected}: chunking mutations that must make the run fail.</li>
 * </ul>
 */
class IsChunkingTest {

	private static final Pattern DIGEST = Pattern.compile("Sorted keys digest =\\s*(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)");

	private static final class Run {
		String digest;
		int verified;
		int chunks;
	}

	private static Run run(char cls, int chunkLog) throws Exception {
		IS is = new IS(cls, false, chunkLog);
		PrintStream original = System.out;
		ByteArrayOutputStream buf = new ByteArrayOutputStream();
		try {
			System.setOut(new PrintStream(buf, true, "UTF-8"));
			is.runBenchmark(new String[0]);
		} finally {
			System.setOut(original);
		}
		Matcher m = DIGEST.matcher(buf.toString("UTF-8"));
		assertTrue(m.find(), "sorted array digest not found");
		Run r = new Run();
		r.digest = m.group(1) + " " + m.group(2) + " " + m.group(3) + " " + m.group(4);
		r.verified = is.getResults().getVerified();
		r.chunks = is.chunkCountOf(is.getNumKeys());
		return r;
	}

	static Stream<Arguments> chunkedCases() {
		return Stream.of(
				Arguments.of('S', new int[] { 0, 3, 5, 11 }),
				Arguments.of('W', new int[] { 3, 5, 12, 19 }),
				Arguments.of('A', new int[] { 10, 16 }));
	}

	@ParameterizedTest(name = "class {0}")
	@MethodSource("chunkedCases")
	void chunkedRunsAreIdenticalToSingleChunk(char cls, int[] chunkLogs) throws Exception {
		Run single = run(cls, IS.DEFAULT_CHUNK_LOG);
		assertEquals(1, single.chunks, "with the default size classes S to C use a single chunk");
		assertEquals(1, single.verified);
		for (int log : chunkLogs) {
			Run r = run(cls, log);
			assertTrue(r.chunks > 1, cls + " chunkLog " + log + ": there must be several chunks");
			assertEquals(1, r.verified, cls + " chunkLog " + log);
			assertEquals(single.digest, r.digest, cls + " chunkLog " + log + ": different digest");
		}
	}

	@Test
	void classDGeometry() throws Exception {
		// With single-int chunks D would have 2^31 chunks: not even the chunk array would fit.
		assertThrows(IllegalArgumentException.class, () -> new IS('D', false, 0).build(new String[0]));
		for (int log : new int[] { 1, 10, 20, 27, 28, 29, 30 }) {
			IS is = new IS('D', false, log);
			is.build(new String[0]); // does not allocate the arrays
			assertEquals(1L << 31, is.getNumKeys(), "NUM_KEYS of D is 2^31 (does not fit in an int)");
			assertEquals(1L << 27, is.getMaxKey());
			for (long total : new long[] { is.getNumKeys(), is.getMaxKey() }) {
				long chunks = is.chunkCountOf(total), len = is.chunkLengthOf(total);
				assertEquals(total, chunks * len, "log " + log + ": the chunks do not add up to the array");
				assertTrue(len <= (1L << Math.min(log, 30)), "log " + log + ": chunk too large");
				// Last index: its chunk and offset fit.
				long last = total - 1;
				assertTrue((last >> log) < chunks || chunks == 1, "log " + log + ": the last index falls outside");
				assertTrue((last & ((1L << log) - 1)) < len || chunks == 1);
			}
		}
		IS def = new IS('D', false);
		def.build(new String[0]);
		assertEquals(8, def.chunkCountOf(def.getNumKeys()), "D with the default size: 8 chunks of 1 GB");
		assertEquals(1, def.chunkCountOf(def.getMaxKey()));
	}

	@Test
	void classesBelowDUseASingleChunkByDefault() throws Exception {
		for (char cls : new char[] { 'S', 'W', 'A', 'B', 'C' }) {
			IS is = new IS(cls, false);
			is.build(new String[0]);
			assertEquals(1, is.chunkCountOf(is.getNumKeys()), "class " + cls);
			assertEquals(1, is.chunkCountOf(is.getMaxKey()), "class " + cls);
		}
	}

	@Test
	void invalidChunkLogIsRejected() {
		assertThrows(IllegalArgumentException.class, () -> new IS('S', false, -1));
		assertThrows(IllegalArgumentException.class, () -> new IS('S', false, 31));
	}

	@Test
	void chunkLogPropertyOverridesTheDefault() throws Exception {
		String old = System.getProperty(IS.CHUNK_LOG_PROPERTY);
		try {
			System.setProperty(IS.CHUNK_LOG_PROPERTY, "10");
			IS is = new IS('S', false);
			is.build(new String[0]);
			assertEquals(64, is.chunkCountOf(is.getNumKeys()));
		} finally {
			if (old == null)
				System.clearProperty(IS.CHUNK_LOG_PROPERTY);
			else
				System.setProperty(IS.CHUNK_LOG_PROPERTY, old);
		}
	}

	@Test
	void runningSumWrapsExactlyAt2To31() {
		// Four chunks of 2: counts 2^30, 0 | 0, 2^30 | 1, 0 | 0, 0. The running sum reaches 2^31 at the 4th element.
		int[][] a = { { 1 << 30, 0 }, { 0, 1 << 30 }, { 0, 0 }, { 0, 0 } };
		IS.runningSum(a);
		assertArrayEquals(new int[] { 1 << 30, 1 << 30 }, a[0]);
		// 2^31 is stored as Integer.MIN_VALUE and is read as unsigned.
		assertArrayEquals(new int[] { 1 << 30, Integer.MIN_VALUE }, a[1]);
		assertArrayEquals(new int[] { Integer.MIN_VALUE, Integer.MIN_VALUE }, a[2]);
		assertEquals(1L << 31, IS.unsignedRank(a[1][1]));
		assertEquals((1L << 31) - 1, IS.unsignedRank(Integer.MAX_VALUE));
		// The carry passes from one chunk to the next (a chunk starting at 0 would give something else).
		int[][] b = { { 3, 0 }, { 4, 0 }, { 5, 6 } };
		IS.runningSum(b);
		assertArrayEquals(new int[] { 3, 3 }, b[0]);
		assertArrayEquals(new int[] { 7, 7 }, b[1]);
		assertArrayEquals(new int[] { 12, 18 }, b[2]);
		// Decrementing the count 2^31 (as int, MIN_VALUE) gives 2^31-1: a valid non-negative index, as fullVerify requires.
		int stored = Integer.MIN_VALUE;
		assertEquals(Integer.MAX_VALUE, --stored);
	}

	/** name, original text (must appear exactly once), new text. */
	private static Arguments m(String name, String old, String nu) {
		return Arguments.of(name, old, nu);
	}

	static Stream<Arguments> chunkingMutations() {
		return Stream.of(
			m("the histogram always counts in chunk 0", "keyBuffPtr[key >> chunkLog][key & chunkMask]++;",
					"keyBuffPtr[0][key & chunkMask]++;"),
			m("the running-sum carry is reset in each chunk", "int chunk[] = a[c];\n",
					"int chunk[] = a[c];\n\t\t\trun = 0;\n"),
			m("get always reads from chunk 0", "private int get(int a[][], long i) {\n\t\treturn a[(int) (i >> chunkLog)]",
					"private int get(int a[][], long i) {\n\t\treturn a[0]"),
			m("set always writes to chunk 0", "private void set(int a[][], long i, int v) {\n\t\ta[(int) (i >> chunkLog)]",
					"private void set(int a[][], long i, int v) {\n\t\ta[0]"),
			m("fullVerify always sorts into chunk 0", "keyArray[rank >> chunkLog][rank & chunkMask] = key;",
					"keyArray[0][rank & chunkMask] = key;"),
			m("fullVerify reads the counters from chunk 0", "--keyBuffPtrGlobal[key >> chunkLog][key & chunkMask];",
					"--keyBuffPtrGlobal[0][key & chunkMask];"),
			m("a chunk is missing", "return (int) Math.max(1, total >> chunkLog);", "return (int) Math.max(1, (total >> chunkLog) - 1);"),
			m("a chunk that is too short", "return (int) Math.min(total, 1L << chunkLog);",
					"return (int) Math.min(total, (1L << chunkLog) - 1);"),
			m("the bucket of each key uses the chunk", "set(keyBuff2, bucketPtrs[key >> SHIFT]++, key);",
					"set(keyBuff2, bucketPtrs[key >> SHIFT]++ & chunkMask, key);"),
			m("the last filler key at another index", "set(keyArray, iteration + MAX_ITERATIONS, (int) (MAX_KEY - iteration));",
					"set(keyArray, iteration + MAX_ITERATIONS + chunkMask + 1, (int) (MAX_KEY - iteration));"),
			m("the partial verification test keys at another index",
					"partialVerifyVals[i] = get(keyArray, testIndexArray[i]);",
					"partialVerifyVals[i] = get(keyArray, testIndexArray[i] & chunkMask);"),
			m("the digest does not go past the first chunk", "for (int c = 0; c < keyArray.length; c++) {\n\t\t\tint chunk[] = keyArray[c];\n\t\t\tfor (i = 0; i < chunk.length; i++) {\n\t\t\t\tlong key",
					"for (int c = 0; c < 1; c++) {\n\t\t\tint chunk[] = keyArray[c];\n\t\t\tfor (i = 0; i < chunk.length; i++) {\n\t\t\t\tlong key"));
	}

	@ParameterizedTest(name = "{0}")
	@MethodSource("chunkingMutations")
	void seededChunkingBugsAreDetected(String name, String old, String nu, @TempDir Path tmp) throws Exception {
		Path src = Paths.get("src/main/java/NPB/SER/IS.java");
		String code = new String(Files.readAllBytes(src), StandardCharsets.ISO_8859_1);
		int first = code.indexOf(old);
		assertTrue(first >= 0, "the text to mutate is no longer in IS.java: " + old);
		assertEquals(-1, code.indexOf(old, first + 1), "the text to mutate is not unique in IS.java: " + old);
		code = code.substring(0, first) + nu + code.substring(first + old.length());
		Path srcDir = Files.createDirectories(tmp.resolve("src").resolve("NPB").resolve("SER"));
		Path out = Files.createDirectories(tmp.resolve("out"));
		Path mutated = srcDir.resolve("IS.java");
		Files.write(mutated, code.getBytes(StandardCharsets.ISO_8859_1));
		VerificationSanityTest.compile(mutated, out);

		Run reference = run('S', IS.DEFAULT_CHUNK_LOG);
		PrintStream original = System.out;
		ByteArrayOutputStream buf = new ByteArrayOutputStream();
		Throwable failure = null;
		int verified = -1;
		try (VerificationSanityTest.ChildFirst cl = new VerificationSanityTest.ChildFirst(out,
				IsChunkingTest.class.getClassLoader())) {
			Class<?> k = cl.loadClass("NPB.SER.IS");
			assertEquals(cl, k.getClassLoader(), "the mutated class was not loaded");
			Benchmark bm = (Benchmark) k.getConstructor(char.class, boolean.class, int.class).newInstance('S', false, 3);
			System.setOut(new PrintStream(buf, true, "UTF-8"));
			try {
				bm.runBenchmark(new String[0]);
				verified = bm.getResults().getVerified();
			} catch (RuntimeException e) { // e.g. ArrayIndexOutOfBounds: the bug is detected all the same
				failure = e;
			}
		} finally {
			System.setOut(original);
		}
		// Detected if the own verification fails, an exception occurs, or the exact digest changes.
		Matcher d = DIGEST.matcher(buf.toString("UTF-8"));
		boolean digestDiffers = !d.find()
				|| !(d.group(1) + " " + d.group(2) + " " + d.group(3) + " " + d.group(4)).equals(reference.digest);
		assertTrue(failure != null || verified == 0 || digestDiffers,
				"[" + name + "] THE MUTATION IS NOT DETECTED: verification " + verified + " and same digest");
	}
}
