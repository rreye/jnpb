package NPB.SER;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Timeout;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import NPB.Benchmark;
import NPB.common.io.BMResults;

/**
 * Kernel x class matrix: every SER kernel must finish with its own verification
 * at {@code SUCCESSFUL} ({@code BMResults.getVerified() == 1}; 0 is a
 * failure and -1 is "not performed", both are errors).
 *
 * <ul>
 * <li>S, W, A, B: normal run ({@code mvn test}).</li>
 * <li>C: tagged {@code classC}, excluded by default (see {@code pom.xml});
 * run separately, with more memory:
 * {@code mvn test -DexcludedGroups= -Dgroups=classC -Dtest.xmx=24g}.</li>
 * <li>D/E/F: out of scope.</li>
 * </ul>
 *
 * In addition, CG, EP, MG, FT, IS, SP, BT and LU are compared with the golden values of the official
 * reference ({@link GoldenCheck}) and the reported Mop/s with the official formula
 * ({@link OpCountCheck}), in the same run.
 */
class VerificationMatrixTest {

	private static Benchmark create(String kernel, char cls) {
		switch (kernel) {
		case "CG": return new CG(cls, false);
		case "EP": return new EP(cls, false);
		case "FT": return new FT(cls, false);
		case "IS": return new IS(cls, false);
		case "MG": return new MG(cls, false);
		case "SP": return new SP(cls, false);
		case "BT": return new BT(cls, false);
		case "LU": return new LU(cls, false);
		default: throw new IllegalArgumentException(kernel);
		}
	}

	private static Stream<Object[]> matrix(String classes) {
		Stream.Builder<Object[]> b = Stream.builder();
		for (char c : classes.toCharArray())
			for (String k : new String[] { "CG", "EP", "FT", "IS", "MG", "SP", "BT", "LU" })
				b.add(new Object[] { k, c });
		return b.build();
	}

	static Stream<Object[]> normalClasses() {
		return matrix("SWAB");
	}

	static Stream<Object[]> bigClasses() {
		return matrix("C");
	}

	private static void run(String kernel, char cls) throws Exception {
		Benchmark bm = create(kernel, cls);
		// stdout is captured (and forwarded) to compare with the golden values.
		PrintStream original = System.out;
		ByteArrayOutputStream buf = new ByteArrayOutputStream();
		try {
			System.setOut(new PrintStream(buf, true, "UTF-8"));
			bm.runBenchmark(new String[0]);
		} finally {
			System.setOut(original);
			original.print(buf.toString("UTF-8"));
		}
		BMResults res = bm.getResults();
		assertNotNull(res, kernel + " class " + cls + ": no results");
		assertEquals(cls, res.getClassSize());
		assertEquals(1, res.getVerified(),
				kernel + " class " + cls + ": verification not SUCCESSFUL");
		GoldenCheck.check(kernel, cls, buf.toString("UTF-8"));
		OpCountCheck.check(kernel, cls, res);
	}

	@ParameterizedTest(name = "{0} class {1}")
	@MethodSource("normalClasses")
	@Timeout(value = 30, unit = TimeUnit.MINUTES)
	void verifiesSmallClasses(String kernel, char cls) throws Exception {
		run(kernel, cls);
	}

	@Tag("classC")
	@ParameterizedTest(name = "{0} class {1}")
	@MethodSource("bigClasses")
	@Timeout(value = 120, unit = TimeUnit.MINUTES)
	void verifiesClassC(String kernel, char cls) throws Exception {
		run(kernel, cls);
	}
}
