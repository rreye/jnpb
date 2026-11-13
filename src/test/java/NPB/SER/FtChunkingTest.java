package NPB.SER;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import NPB.Benchmark;
import NPB.common.io.BMResults;

/**
 * Chunking (by z planes) of the 3D arrays of SER FT, needed for class D ((nx+1)*ny*nz = 2148532224 > 2^31-1).
 * It is a pure refactor: with any chunk size the checksums must be BIT-FOR-BIT identical to those of a
 * single-chunk run (which in turn match the original unchunked code, checked by hand when the refactor was done).
 * They are not compared with the official golden values because those carry a tolerance (Math.exp versus glibc) and
 * would not see a 1 ulp error.
 * <ul>
 * <li>{@link #chunkedRunsAreBitIdenticalToSingleChunk}: S, W and A with 1 plane per chunk, with 3 (does not divide nz: the
 * last chunk is partial) and with the default value (one chunk).</li>
 * <li>{@link #classDGeometry*}: class D does not fit in this machine (~86 GB) but {@code build()} does not allocate the grid:
 * it is checked, without allocating memory, that every plane has a valid chunk and offset, that no chunk
 * exceeds the limit or a Java array, and that ntotal (2^31) does not overflow.</li>
 * <li>{@link #seededChunkingBugsAreDetected}: chunking mutations that must make the run fail
 * (FT's own verification or an exception).</li>
 * </ul>
 */
class FtChunkingTest {

	private static final Pattern CHK = Pattern.compile("T =\\s*(\\d+)\\s+Checksum =\\s*\\(([^,]+),\\s*([^)]+)\\)");

	private static int planeSize(char cls) throws Exception {
		FT probe = new FT(cls, false, 1);
		probe.build(new String[0]);
		return probe.getPlaneSize();
	}

	/** Runs FT and returns the bits of all checksums (re, im of each iteration) and the number of chunks. */
	private static final class Run {
		final List<Long> bits = new ArrayList<>();
		int chunks;
		int verified;
	}

	private static Run run(char cls, long maxChunkElems) throws Exception {
		FT ft = new FT(cls, false, maxChunkElems);
		PrintStream original = System.out;
		ByteArrayOutputStream buf = new ByteArrayOutputStream();
		try {
			System.setOut(new PrintStream(buf, true, "UTF-8"));
			ft.runBenchmark(new String[0]);
		} finally {
			System.setOut(original);
		}
		Run r = new Run();
		Matcher m = CHK.matcher(buf.toString("UTF-8"));
		while (m.find()) {
			r.bits.add(Double.doubleToRawLongBits(Double.parseDouble(m.group(2))));
			r.bits.add(Double.doubleToRawLongBits(Double.parseDouble(m.group(3))));
		}
		r.chunks = ft.getChunkCount();
		BMResults res = ft.getResults();
		r.verified = res.getVerified();
		return r;
	}

	static Stream<Arguments> chunkedCases() {
		return Stream.of(Arguments.of('S'), Arguments.of('W'), Arguments.of('A'));
	}

	@ParameterizedTest(name = "class {0}")
	@MethodSource("chunkedCases")
	void chunkedRunsAreBitIdenticalToSingleChunk(char cls) throws Exception {
		int plane = planeSize(cls);
		Run single = run(cls, FT.DEFAULT_MAX_CHUNK_ELEMS);
		assertEquals(1, single.chunks, "with the default limit classes S to C use a single chunk");
		assertEquals(1, single.verified);
		assertTrue(single.bits.size() >= 12, "the checksums were not read");
		for (int planes : new int[] { 1, 3 }) {
			Run r = run(cls, (long) planes * plane);
			assertTrue(r.chunks > 1, cls + ": with " + planes + " planes per chunk there must be several chunks");
			assertEquals(1, r.verified, cls + " with " + planes + " planes per chunk");
			assertEquals(single.bits, r.bits, cls + " with " + planes + " planes per chunk: checksums differ bit for bit");
		}
		// A limit that is not a multiple of a plane is rounded down (here 2.5 planes -> 2).
		Run r = run(cls, 5L * plane / 2);
		assertEquals(single.bits, r.bits, cls + " with a limit that is not a multiple of the plane");
	}

	private static void checkGeometry(char cls, long maxElems) throws Exception {
		FT ft = new FT(cls, false, maxElems);
		ft.build(new String[0]);
		int plane = ft.getPlaneSize();
		int nz = ft.getNz();
		boolean[] seen = new boolean[ft.getChunkCount()];
		int expectedPlanes = 0;
		int c = 0;
		for (int p = 0; p < nz; p++) {
			int chunk = ft.chunkOfPlane(p);
			int off = ft.offsetOfPlane(p);
			assertTrue(chunk >= 0 && chunk < ft.getChunkCount(), "plane " + p + ": chunk " + chunk);
			assertTrue(chunk == c || chunk == c + 1, "planes must traverse the chunks in order");
			if (chunk != c) {
				c = chunk;
				expectedPlanes = 0;
			}
			seen[chunk] = true;
			assertEquals((long) expectedPlanes * plane, off, "plane " + p + ": offset");
			expectedPlanes++;
			long length = (long) ft.planesInChunk(chunk) * plane;
			assertTrue(off + (long) plane <= length, "plane " + p + " falls outside its chunk");
			assertTrue(length <= Math.max(maxElems, plane), "chunk " + chunk + " exceeds the limit");
			assertTrue(length <= Integer.MAX_VALUE - 8, "chunk " + chunk + " does not fit in a Java array");
		}
		for (boolean b : seen)
			assertTrue(b, "there is a chunk without planes");
		int total = 0;
		for (int k = 0; k < ft.getChunkCount(); k++)
			total += ft.planesInChunk(k);
		assertEquals(nz, total, "the chunks do not add up to nz planes");
	}

	@Test
	void classDGeometryWithDefaultLimit() throws Exception {
		FT ft = new FT('D', false, FT.DEFAULT_MAX_CHUNK_ELEMS);
		ft.build(new String[0]);
		assertEquals(2049 * 1024, ft.getPlaneSize());
		assertEquals(1L << 31, ft.getNtotal(), "ntotal of D is 2^31 (overflows an int)");
		// The whole grid does not fit in an array: it must be chunked.
		assertTrue((long) ft.getPlaneSize() * 1024 > Integer.MAX_VALUE);
		assertTrue(ft.getChunkCount() > 1);
		checkGeometry('D', FT.DEFAULT_MAX_CHUNK_ELEMS);
	}

	@Test
	void classDGeometryWithOtherLimits() throws Exception {
		for (long limit : new long[] { 1, 2049L * 1024, 1000L * 2049 * 1024, 1L << 30, 1L << 31, Long.MAX_VALUE })
			checkGeometry('D', limit);
	}

	@Test
	void classesBelowDUseASingleChunkByDefault() throws Exception {
		for (char cls : new char[] { 'S', 'W', 'A', 'B', 'C' }) {
			FT ft = new FT(cls, false);
			ft.build(new String[0]);
			assertEquals(1, ft.getChunkCount(), "class " + cls);
		}
	}

	@Test
	void chunkSizePropertyOverridesTheDefault() throws Exception {
		String old = System.getProperty(FT.CHUNK_ELEMS_PROPERTY);
		try {
			System.setProperty(FT.CHUNK_ELEMS_PROPERTY, Integer.toString(planeSize('S') * 4));
			FT ft = new FT('S', false);
			ft.build(new String[0]);
			assertEquals(16, ft.getChunkCount());
		} finally {
			if (old == null)
				System.clearProperty(FT.CHUNK_ELEMS_PROPERTY);
			else
				System.setProperty(FT.CHUNK_ELEMS_PROPERTY, old);
		}
	}

	/** name, original text (must appear exactly once), new text. */
	private static Arguments m(String name, String old, String nu) {
		return Arguments.of(name, old, nu);
	}

	static Stream<Arguments> chunkingMutations() {
		return Stream.of(
			m("evolve always uses the twiddle of chunk 0", "tw = twiddle[zChunk[i-1]];\n\t\t\tfor (k", "tw = twiddle[0];\n\t\t\tfor (k"),
			m("evolve writes y in chunk 0", "yr = y[zChunk[i-1]][REAL];", "yr = y[0][REAL];"),
			m("fftX reads from chunk 0", "xr = x[zChunk[k]][REAL]; xi = x[zChunk[k]][IMAG];\n\t\t\tdr = dst",
					"xr = x[0][REAL]; xi = x[zChunk[k]][IMAG];\n\t\t\tdr = dst"),
			m("fftX writes to chunk 0", "dr = dst[zChunk[k]][REAL];", "dr = dst[0][REAL];"),
			m("fftY always works in chunk 0", "xr = x[zChunk[k]][REAL]; xi = x[zChunk[k]][IMAG];\n\t\t\tdr = xr;",
					"xr = x[0][REAL]; xi = x[zChunk[k]][IMAG];\n\t\t\tdr = xr;"),
			m("fftZ reads the next plane of the same chunk (offset)",
					"xi = x[zChunk[j]][IMAG];\n\t\t\t\t\tpos = zOff[j] + k*(n1+1) + bls;",
					"xi = x[zChunk[j]][IMAG];\n\t\t\t\t\tpos = zOff[j + 1] + k*(n1+1) + bls;"),
			m("fftZ gather always from chunk 0", "xr = x[zChunk[j]][REAL];", "xr = x[0][REAL];"),
			m("fftZ scatter to chunk 0", "dr = dst[zChunk[j]][REAL];", "dr = dst[0][REAL];"),
			m("the checksum reads the wrong plane", "pos = zOff[ki-1] + (ji-1)*(d1+1) + (ii-1);",
					"pos = zOff[ki-1] + (ii-1)*(d1+1) + (ji-1);"),
			m("the checksum reads from chunk 0", "csum[REAL] + u[zChunk[ki-1]][REAL][pos]", "csum[REAL] + u[0][REAL][pos]"),
			m("initial conditions in chunk 0", "u0r = u0[zChunk[k-1]][REAL];", "u0r = u0[0][REAL];"),
			m("twiddle in chunk 0", "tw = twiddle[zChunk[i-1]];\n\t\t\t\tpos = zOff[i-1]",
					"tw = twiddle[0];\n\t\t\t\tpos = zOff[i-1]"),
			m("plane offset not reset in each chunk", "zOff[p] = (p % planesPerChunk) * planeSize;",
					"zOff[p] = (p % (planesPerChunk + 1)) * planeSize;"),
			m("planes of the last chunk miscounted", "return Math.min(planesPerChunk, nz - c * planesPerChunk);",
					"return Math.min(planesPerChunk, nz - c * planesPerChunk - 1);"));
	}

	@ParameterizedTest(name = "{0}")
	@MethodSource("chunkingMutations")
	void seededChunkingBugsAreDetected(String name, String old, String nu, @TempDir Path tmp) throws Exception {
		Path src = Paths.get("src/main/java/NPB/SER/FT.java");
		String code = new String(Files.readAllBytes(src), StandardCharsets.ISO_8859_1);
		int first = code.indexOf(old);
		assertTrue(first >= 0, "the text to mutate is no longer in FT.java: " + old);
		assertEquals(-1, code.indexOf(old, first + 1), "the text to mutate is not unique in FT.java: " + old);
		code = code.substring(0, first) + nu + code.substring(first + old.length());
		Path srcDir = Files.createDirectories(tmp.resolve("src").resolve("NPB").resolve("SER"));
		Path out = Files.createDirectories(tmp.resolve("out"));
		Path mutated = srcDir.resolve("FT.java");
		Files.write(mutated, code.getBytes(StandardCharsets.ISO_8859_1));
		VerificationSanityTest.compile(mutated, out);

		long limit = 3L * planeSize('S'); // S with 3 planes per chunk: 22 chunks, the last with a single plane
		PrintStream original = System.out;
		Throwable failure = null;
		int verified = -1;
		try (VerificationSanityTest.ChildFirst cl = new VerificationSanityTest.ChildFirst(out,
				FtChunkingTest.class.getClassLoader())) {
			Class<?> k = cl.loadClass("NPB.SER.FT");
			assertEquals(cl, k.getClassLoader(), "the mutated class was not loaded");
			Benchmark bm = (Benchmark) k.getConstructor(char.class, boolean.class, long.class).newInstance('S', false,
					limit);
			System.setOut(new PrintStream(new ByteArrayOutputStream(), true, "UTF-8"));
			try {
				bm.runBenchmark(new String[0]);
				verified = bm.getResults().getVerified();
			} catch (RuntimeException e) { // e.g. ArrayIndexOutOfBounds: the bug is detected all the same
				failure = e;
			}
		} finally {
			System.setOut(original);
		}
		assertTrue(failure != null || verified == 0,
				"[" + name + "] THE MUTATION IS NOT DETECTED: the verification gave " + verified);
	}

	@Test
	void unmutatedControlPassesWithTheSameLimit() throws Exception {
		Run r = run('S', 3L * planeSize('S'));
		assertEquals(1, r.verified);
	}
}
