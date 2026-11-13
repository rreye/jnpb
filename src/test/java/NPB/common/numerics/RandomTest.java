package NPB.common.numerics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * {@link Random} against the official NPB 3.4.4 generator (randdp.f90 == randi8.f90;
 * power of MG OMP/MPI and ipow46 of FT), compared bit for bit. The reference
 * data is in golden/random.properties (scripts/gen_random_golden.py).
 *
 * Correspondence between Java and Fortran methods:
 * <ul>
 * <li>{@code randlc(x, a)} (Java) returns the NEW SEED; Fortran returns
 * {@code 2^-46 * x} and updates x.</li>
 * <li>{@code vrandlc(x, a)} (Java) is a single step: {@code {seed, 2^-46 * seed}}.</li>
 * <li>{@code randlc(a)} (Java) uses the internal seed 314159265 and returns {@code 2^-46 * x}.</li>
 * <li>{@code vranlc} / {@code complexvranlc} == Fortran's {@code vranlc} (1 or 2 outputs per i).</li>
 * </ul>
 */
class RandomTest {

	private static Properties g;

	@BeforeAll
	static void load() throws IOException {
		g = new Properties();
		try (InputStream in = RandomTest.class.getResourceAsStream("/golden/random.properties")) {
			assertNotNull(in, "missing golden/random.properties (python3 -I scripts/gen_random_golden.py)");
			g.load(in);
		}
	}

	private static long[] longs(String csv) {
		String[] t = csv.split(",");
		long[] r = new long[t.length];
		for (int i = 0; i < t.length; i++)
			r[i] = Long.parseLong(t[i]);
		return r;
	}

	private static long[] hexBits(String csv) {
		String[] t = csv.split(",");
		long[] r = new long[t.length];
		for (int i = 0; i < t.length; i++)
			r[i] = Long.parseUnsignedLong(t[i], 16);
		return r;
	}

	private static double d(String key) {
		return Double.parseDouble(g.getProperty(key));
	}

	private static long bits(double v) {
		return Double.doubleToRawLongBits(v);
	}

	@ParameterizedTest(name = "case {0}")
	@ValueSource(ints = { 1, 2, 3, 4, 5 })
	void randlcExternalSeedMatchesReference(int c) {
		Random rng = new Random();
		double a = d("case." + c + ".a");
		double x = d("case." + c + ".x0");
		long[] ref = longs(g.getProperty("case." + c + ".x"));
		for (int i = 0; i < ref.length; i++) {
			x = rng.randlc(x, a);
			assertEquals(bits((double) ref[i]), bits(x), "case " + c + " step " + (i + 1));
		}
	}

	@ParameterizedTest(name = "case {0}")
	@ValueSource(ints = { 1, 2, 3, 4, 5 })
	void vrandlcStepMatchesReference(int c) {
		Random rng = new Random();
		double a = d("case." + c + ".a");
		double x = d("case." + c + ".x0");
		long[] refX = longs(g.getProperty("case." + c + ".x"));
		long[] refY = hexBits(g.getProperty("case." + c + ".y"));
		for (int i = 0; i < refX.length; i++) {
			double[] v = rng.vrandlc(x, a);
			x = v[0];
			assertEquals(bits((double) refX[i]), bits(v[0]), "caso " + c + " seed step " + (i + 1));
			assertEquals(refY[i], bits(v[1]), "caso " + c + " value step " + (i + 1));
		}
	}

	@Test
	void internalSeedRandlcMatchesReference() {
		// The internal seed is 314159265 (case 1 with a = 1220703125).
		Random rng = new Random();
		double a = d("case.1.a");
		assertEquals(d("case.1.x0"), 314159265.0);
		long[] refY = hexBits(g.getProperty("case.1.y"));
		for (int i = 0; i < refY.length; i++)
			assertEquals(refY[i], bits(rng.randlc(a)), "step " + (i + 1));
	}

	@ParameterizedTest(name = "case {0}")
	@ValueSource(ints = { 1, 2, 3, 4, 5 })
	void vranlcMatchesReference(int c) {
		Random rng = new Random();
		double a = d("case." + c + ".a");
		double x0 = d("case." + c + ".x0");
		long[] ref = hexBits(g.getProperty("case." + c + ".vranlc.y"));
		int off = 3;
		double[] y = new double[off + ref.length + 2];
		java.util.Arrays.fill(y, -1.0);
		double seed = rng.vranlc(ref.length, x0, a, y, off);
		assertEquals(Long.parseLong(g.getProperty("case." + c + ".vranlc.x")), (long) seed, "final seed");
		for (int i = 0; i < ref.length; i++)
			assertEquals(ref[i], bits(y[off + i]), "case " + c + " value " + (i + 1));
		for (int i = 0; i < off; i++)
			assertEquals(-1.0, y[i], "write before the offset");
		assertEquals(-1.0, y[y.length - 1], "write after the end");
		assertEquals(-1.0, y[y.length - 2], "write after the end");
	}

	@ParameterizedTest(name = "case {0}")
	@ValueSource(ints = { 1, 2, 3, 4, 5 })
	void complexvranlcMatchesReference(int c) {
		Random rng = new Random();
		double a = d("case." + c + ".a");
		double x0 = d("case." + c + ".x0");
		long[] ref = hexBits(g.getProperty("case." + c + ".vranlc.y"));
		long refSeed = Long.parseLong(g.getProperty("case." + c + ".vranlc.x"));
		int pairs = ref.length / 2, off = 2;

		double[][] y2 = new double[2][off + pairs];
		double s2 = rng.complexvranlc(pairs, x0, a, y2, off);
		assertEquals(refSeed, (long) s2, "seed ([2][n] format)");
		double[] y1 = new double[2 * (off + pairs)];
		double s1 = rng.complexvranlc(pairs, x0, a, y1, off);
		assertEquals(refSeed, (long) s1, "seed (interleaved format)");
		for (int i = 0; i < pairs; i++) {
			assertEquals(ref[2 * i], bits(y2[0][off + i]), "re [2][n] " + i);
			assertEquals(ref[2 * i + 1], bits(y2[1][off + i]), "im [2][n] " + i);
			assertEquals(ref[2 * i], bits(y1[2 * (off + i)]), "interleaved re " + i);
			assertEquals(ref[2 * i + 1], bits(y1[2 * (off + i) + 1]), "interleaved im " + i);
		}
	}

	@Test
	void longSequenceMatchesReference() {
		// 10^6 steps of randlc with the default seed, sampling every 10^4.
		Random rng = new Random();
		double a = d("case.1.a");
		double x = d("case.1.x0");
		int step = Integer.parseInt(g.getProperty("long.step"));
		long[] ref = longs(g.getProperty("long.x"));
		assertEquals(100, ref.length);
		for (int i = 1; i <= step * ref.length; i++) {
			x = rng.randlc(x, a);
			if (i % step == 0)
				assertEquals(ref[i / step - 1], (long) x, "step " + i);
		}
	}

	@Test
	void powerMatchesReference() {
		Random rng = new Random();
		int n = 0;
		for (int i = 0; g.getProperty("pow." + i) != null; i++, n++) {
			String[] t = g.getProperty("pow." + i).split(",");
			double a = Double.parseDouble(t[0]);
			int e = Integer.parseInt(t[1]);
			assertEquals(Long.parseLong(t[2]), (long) rng.power(a, e), "power(" + t[0] + ", " + e + ")");
		}
		assertEquals(69, n, "number of power cases");
	}

	@Test
	void power4MatchesReference() {
		Random rng = new Random();
		int n = 0;
		for (int i = 0; g.getProperty("pow4." + i) != null; i++, n++) {
			String[] t = g.getProperty("pow4." + i).split(",");
			double a = Double.parseDouble(t[0]);
			int n1 = Integer.parseInt(t[1]), n2 = Integer.parseInt(t[2]), n3 = Integer.parseInt(t[3]);
			assertEquals(Long.parseLong(t[4]), (long) rng.power(a, n1, n2, n3),
					"power(" + a + ", " + n1 + ", " + n2 + ", " + n3 + ")");
		}
		assertEquals(14, n, "number of power4 cases");
	}
}
