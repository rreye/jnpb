package NPB.SER;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.stream.Stream;

import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import NPB.Benchmark;
import NPB.common.io.BMResults;

/**
 * Sanity check of the verification of each SER kernel: the verification must
 * FAIL when it should. For each case the real kernel source is taken and ONE
 * mutation is applied (the text to replace must appear exactly once, so that
 * a refactor does not silently disable the test), compiled in memory
 * and run for class S:
 * <ul>
 * <li>perturbed expected value (far above epsilon) -> 0;</li>
 * <li>epsilon edge: slightly below -> 1, slightly above -> 0;</li>
 * <li>NaN in the computed or the expected value -> 0 (since {@code NaN > eps == false},
 * a badly written comparison would let it pass);</li>
 * <li>controls (no mutation, or a mutation within the tolerance) -> 1, to
 * show that the setup does not break everything by itself.</li>
 * </ul>
 * Requires a JDK (javax.tools) and the sources in {@code src/main/java}.
 */
class VerificationSanityTest {

	private static final int FAIL = 0, PASS = 1;

	/** kernel, name, original text (null = unmutated), new text, expected verified. */
	private static Arguments m(String kernel, String name, String old, String nu, int expected) {
		return Arguments.of(kernel, name, old, nu, expected);
	}

	static Stream<Arguments> mutations() {
		final String NAN = "Double.NaN";
		return Stream.of(
			// ---- CG: epsilon 1e-10, final zeta
			m("CG", "control: unmutated", null, null, PASS),
			m("CG", "expected zeta +1e-6", "zetaVerifyValue = 8.5971775078648;", "zetaVerifyValue = 8.5971775078648 * (1 + 1e-6);", FAIL),
			m("CG", "expected zeta +5e-11 (within eps)", "zetaVerifyValue = 8.5971775078648;", "zetaVerifyValue = 8.5971775078648 * (1 + 5e-11);", PASS),
			m("CG", "expected zeta +2e-10 (outside eps)", "zetaVerifyValue = 8.5971775078648;", "zetaVerifyValue = 8.5971775078648 * (1 + 2e-10);", FAIL),
			m("CG", "computed zeta NaN", "Math.abs(zeta - zetaVerifyValue)", "Math.abs(" + NAN + " - zetaVerifyValue)", FAIL),
			m("CG", "expected zeta NaN", "zetaVerifyValue = 8.5971775078648;", "zetaVerifyValue = " + NAN + ";", FAIL),

			// ---- EP: epsilon 1e-8; sx, sy and gc
			m("EP", "control: unmutated", null, null, PASS),
			m("EP", "expected sx +1e-6", "sxVerifyValue = 1.051299420395306E+07;", "sxVerifyValue = 1.051299420395306E+07 * (1 + 1e-6);", FAIL),
			m("EP", "expected sx +5e-9 (within eps)", "sxVerifyValue = 1.051299420395306E+07;", "sxVerifyValue = 1.051299420395306E+07 * (1 + 5e-9);", PASS),
			m("EP", "expected sx +2e-8 (outside eps)", "sxVerifyValue = 1.051299420395306E+07;", "sxVerifyValue = 1.051299420395306E+07 * (1 + 2e-8);", FAIL),
			m("EP", "expected sy +1e-6", "syVerifyValue = 1.051517131857535E+07;", "syVerifyValue = 1.051517131857535E+07 * (1 + 1e-6);", FAIL),
			m("EP", "expected gc +1 (the count is verified)", "gcVerifyValue = 13176389.0;", "gcVerifyValue = 13176390.0;", FAIL),
			m("EP", "computed sx NaN", "Math.abs((sx - sxVerifyValue) / sxVerifyValue)", "Math.abs((" + NAN + " - sxVerifyValue) / sxVerifyValue)", FAIL),
			m("EP", "computed sy NaN", "Math.abs((sy - syVerifyValue) / syVerifyValue)", "Math.abs((" + NAN + " - syVerifyValue) / syVerifyValue)", FAIL),
			m("EP", "computed gc NaN", "Math.abs((gc - gcVerifyValue) / gcVerifyValue)", "Math.abs((" + NAN + " - gcVerifyValue) / gcVerifyValue)", FAIL),
			m("EP", "expected sx NaN", "sxVerifyValue = 1.051299420395306E+07;", "sxVerifyValue = " + NAN + ";", FAIL),

			// ---- FT: epsilon 1e-12; checksum of each iteration
			m("FT", "control: unmutated", null, null, PASS),
			m("FT", "expected checksum +2e-8", "5.546385409189E+02", "5.546385509189E+02", FAIL),
			m("FT", "expected checksum +5e-13 (within eps)", "5.546385409189E+02", "5.546385409189E+02 * 1.0000000000005", PASS),
			m("FT", "expected checksum +2e-12 (outside eps)", "5.546385409189E+02", "5.546385409189E+02 * 1.000000000002", FAIL),
			m("FT", "computed real checksum NaN", "sums[REAL][iterN - 1] = csum[REAL];", "sums[REAL][iterN - 1] = " + NAN + ";", FAIL),
			m("FT", "computed imaginary checksum NaN", "sums[IMAG][iterN - 1] = csum[IMAG];", "sums[IMAG][iterN - 1] = " + NAN + ";", FAIL),
			m("FT", "expected checksum NaN", "{ 5.546087004964E+02,", "{ " + NAN + ",", FAIL),

			// ---- IS: integer verification (5 ranks x 10 iterations + full sort)
			m("IS", "control: unmutated", null, null, PASS),
			m("IS", "expected rango +1", "sTestRankArray[] = { 0, 18, 346, 64917, 65463 }", "sTestRankArray[] = { 0, 19, 346, 64917, 65463 }", FAIL),
			m("IS", "test key index changed", "sTestIndexArray[] = { 48427, 17148, 23627, 62548, 4431 }", "sTestIndexArray[] = { 48427, 17149, 23627, 62548, 4431 }", FAIL),
			m("IS", "invalid full sort", "if (prev > chunk[i])", "if (prev >= chunk[i])", FAIL),

			// ---- MG: epsilon 1e-8; final L2 norm
			m("MG", "control: unmutated", null, null, PASS),
			m("MG", "expected norma +1e-6", "verifyValue = 0.530770700573E-4;", "verifyValue = 0.530770700573E-4 * (1 + 1e-6);", FAIL),
			m("MG", "expected norma +5e-9 (within eps)", "verifyValue = 0.530770700573E-4;", "verifyValue = 0.530770700573E-4 * (1 + 5e-9);", PASS),
			m("MG", "expected norma +2e-8 (outside eps)", "verifyValue = 0.530770700573E-4;", "verifyValue = 0.530770700573E-4 * (1 + 2e-8);", FAIL),
			m("MG", "computed norma NaN", "Math.abs(rnm[0]-verifyValue)/verifyValue", "Math.abs(" + NAN + "-verifyValue)/verifyValue", FAIL),
			m("MG", "expected norma NaN", "verifyValue = 0.530770700573E-4;", "verifyValue = " + NAN + ";", FAIL),

			// ---- SP: epsilon 1e-8; RMS norms of the residual (xcr) and of the error (xce), and dt
			m("SP", "control: unmutated", null, null, PASS),
			m("SP", "expected xcr +1e-6", "xcrref[0] = 2.7470315451339479e-02;", "xcrref[0] = 2.7470315451339479e-02 * (1 + 1e-6);", FAIL),
			m("SP", "expected xcr +5e-9 (within eps)", "xcrref[0] = 2.7470315451339479e-02;", "xcrref[0] = 2.7470315451339479e-02 * (1 + 5e-9);", PASS),
			m("SP", "expected xcr +2e-8 (outside eps)", "xcrref[0] = 2.7470315451339479e-02;", "xcrref[0] = 2.7470315451339479e-02 * (1 + 2e-8);", FAIL),
			m("SP", "expected xce +1e-6", "xceref[0] = 2.7289258557377227e-05;", "xceref[0] = 2.7289258557377227e-05 * (1 + 1e-6);", FAIL),
			m("SP", "expected xce +5e-9 (within eps)", "xceref[0] = 2.7289258557377227e-05;", "xceref[0] = 2.7289258557377227e-05 * (1 + 5e-9);", PASS),
			m("SP", "expected xce +2e-8 (outside eps)", "xceref[0] = 2.7289258557377227e-05;", "xceref[0] = 2.7289258557377227e-05 * (1 + 2e-8);", FAIL),
			m("SP", "computed xcr NaN", "xcrdif[m] = Math.abs((xcr[m]-xcrref[m])/xcrref[m]);", "xcrdif[m] = Math.abs((" + NAN + "-xcrref[m])/xcrref[m]);", FAIL),
			m("SP", "computed xce NaN", "xcedif[m] = Math.abs((xce[m]-xceref[m])/xceref[m]);", "xcedif[m] = Math.abs((" + NAN + "-xceref[m])/xceref[m]);", FAIL),
			m("SP", "expected xcr NaN", "xcrref[0] = 2.7470315451339479e-02;", "xcrref[0] = " + NAN + ";", FAIL),
			m("SP", "expected xce NaN", "xceref[0] = 2.7289258557377227e-05;", "xceref[0] = " + NAN + ";", FAIL),
			m("SP", "reference dt +1e-3", "dtref = 1.5e-2;", "dtref = 1.6e-2;", FAIL),
			m("SP", "reference dt +5e-9 (within eps)", "dtref = 1.5e-2;", "dtref = 1.5e-2 + 5e-9;", PASS),
			m("SP", "reference dt +2e-8 (outside eps)", "dtref = 1.5e-2;", "dtref = 1.5e-2 + 2e-8;", FAIL),
			m("SP", "computed dt NaN", "Math.abs(dt-dtref) <= epsilon", "Math.abs(" + NAN + "-dtref) <= epsilon", FAIL),

			// ---- BT: epsilon 1e-8; RMS norms of the residual (xcr) and of the error (xce), and dt
			m("BT", "control: unmutated", null, null, PASS),
			m("BT", "expected xcr +1e-6", "xcrref[0] = 1.7034283709541311e-01;", "xcrref[0] = 1.7034283709541311e-01 * (1 + 1e-6);", FAIL),
			m("BT", "expected xcr +5e-9 (within eps)", "xcrref[0] = 1.7034283709541311e-01;", "xcrref[0] = 1.7034283709541311e-01 * (1 + 5e-9);", PASS),
			m("BT", "expected xcr +2e-8 (outside eps)", "xcrref[0] = 1.7034283709541311e-01;", "xcrref[0] = 1.7034283709541311e-01 * (1 + 2e-8);", FAIL),
			m("BT", "expected xce +1e-6", "xceref[0] = 4.9976913345811579e-04;", "xceref[0] = 4.9976913345811579e-04 * (1 + 1e-6);", FAIL),
			m("BT", "expected xce +5e-9 (within eps)", "xceref[0] = 4.9976913345811579e-04;", "xceref[0] = 4.9976913345811579e-04 * (1 + 5e-9);", PASS),
			m("BT", "expected xce +2e-8 (outside eps)", "xceref[0] = 4.9976913345811579e-04;", "xceref[0] = 4.9976913345811579e-04 * (1 + 2e-8);", FAIL),
			m("BT", "computed xcr NaN", "xcrdif[m] = Math.abs((xcr[m]-xcrref[m])/xcrref[m]);", "xcrdif[m] = Math.abs((" + NAN + "-xcrref[m])/xcrref[m]);", FAIL),
			m("BT", "computed xce NaN", "xcedif[m] = Math.abs((xce[m]-xceref[m])/xceref[m]);", "xcedif[m] = Math.abs((" + NAN + "-xceref[m])/xceref[m]);", FAIL),
			m("BT", "expected xcr NaN", "xcrref[0] = 1.7034283709541311e-01;", "xcrref[0] = " + NAN + ";", FAIL),
			m("BT", "expected xce NaN", "xceref[0] = 4.9976913345811579e-04;", "xceref[0] = " + NAN + ";", FAIL),
			m("BT", "reference dt +1e-3", "dtref = 1.0e-2;", "dtref = 1.1e-2;", FAIL),
			m("BT", "reference dt +5e-9 (within eps)", "dtref = 1.0e-2;", "dtref = 1.0e-2 + 5e-9;", PASS),
			m("BT", "reference dt +2e-8 (outside eps)", "dtref = 1.0e-2;", "dtref = 1.0e-2 + 2e-8;", FAIL),
			m("BT", "computed dt NaN", "Math.abs(dt-dtref) <= epsilon", "Math.abs(" + NAN + "-dtref) <= epsilon", FAIL),

			// ---- LU: epsilon 1e-8; RMS norms of the residual (xcr) and of the error (xce), surface integral (xci) and dt
			m("LU", "control: unmutated", null, null, PASS),
			m("LU", "expected xcr +1e-6", "xcrref[0] = 1.6196343210976702e-02;", "xcrref[0] = 1.6196343210976702e-02 * (1 + 1e-6);", FAIL),
			m("LU", "expected xcr +5e-9 (within eps)", "xcrref[0] = 1.6196343210976702e-02;", "xcrref[0] = 1.6196343210976702e-02 * (1 + 5e-9);", PASS),
			m("LU", "expected xcr +2e-8 (outside eps)", "xcrref[0] = 1.6196343210976702e-02;", "xcrref[0] = 1.6196343210976702e-02 * (1 + 2e-8);", FAIL),
			m("LU", "expected xce +1e-6", "xceref[0] = 6.4223319957960924e-04;", "xceref[0] = 6.4223319957960924e-04 * (1 + 1e-6);", FAIL),
			m("LU", "expected xce +5e-9 (within eps)", "xceref[0] = 6.4223319957960924e-04;", "xceref[0] = 6.4223319957960924e-04 * (1 + 5e-9);", PASS),
			m("LU", "expected xce +2e-8 (outside eps)", "xceref[0] = 6.4223319957960924e-04;", "xceref[0] = 6.4223319957960924e-04 * (1 + 2e-8);", FAIL),
			m("LU", "expected xci +1e-6", "xciref = 7.8418928865937083e+00;", "xciref = 7.8418928865937083e+00 * (1 + 1e-6);", FAIL),
			m("LU", "expected xci +5e-9 (within eps)", "xciref = 7.8418928865937083e+00;", "xciref = 7.8418928865937083e+00 * (1 + 5e-9);", PASS),
			m("LU", "expected xci +2e-8 (outside eps)", "xciref = 7.8418928865937083e+00;", "xciref = 7.8418928865937083e+00 * (1 + 2e-8);", FAIL),
			m("LU", "computed xcr NaN", "xcrdif[m] = Math.abs((rsdnm[m]-xcrref[m])/xcrref[m]);", "xcrdif[m] = Math.abs((" + NAN + "-xcrref[m])/xcrref[m]);", FAIL),
			m("LU", "computed xce NaN", "xcedif[m] = Math.abs((errnm[m]-xceref[m])/xceref[m]);", "xcedif[m] = Math.abs((" + NAN + "-xceref[m])/xceref[m]);", FAIL),
			m("LU", "computed xci NaN", "xcidif = Math.abs((frc - xciref)/xciref);", "xcidif = Math.abs((" + NAN + " - xciref)/xciref);", FAIL),
			m("LU", "expected xcr NaN", "xcrref[0] = 1.6196343210976702e-02;", "xcrref[0] = " + NAN + ";", FAIL),
			m("LU", "expected xce NaN", "xceref[0] = 6.4223319957960924e-04;", "xceref[0] = " + NAN + ";", FAIL),
			m("LU", "expected xci NaN", "xciref = 7.8418928865937083e+00;", "xciref = " + NAN + ";", FAIL),
			m("LU", "reference dt +1e-3", "dtref = 5.0e-1;", "dtref = 5.1e-1;", FAIL),
			m("LU", "reference dt +5e-9 (within eps)", "dtref = 5.0e-1;", "dtref = 5.0e-1 + 5e-9;", PASS),
			m("LU", "reference dt +2e-8 (outside eps)", "dtref = 5.0e-1;", "dtref = 5.0e-1 + 2e-8;", FAIL),
			m("LU", "computed dt NaN", "Math.abs(dt-dtref) <= epsilon", "Math.abs(" + NAN + "-dtref) <= epsilon", FAIL));
	}

	@TempDir
	Path tmp;

	@ParameterizedTest(name = "{0}: {1}")
	@MethodSource("mutations")
	void verificationBehavesUnderMutation(String kernel, String name, String old, String nu, int expected)
			throws Exception {
		check(kernel, 'S', name, old, nu, expected);
	}

	/**
	 * IS: the partial verification has one branch per class (add or subtract the iteration depending on the position of
	 * the test key) and the other tests only exercise the one for S. Here, in each class, one expected value on
	 * each side of the branch and one test key index are mutated, plus an unmutated control.
	 */
	@ParameterizedTest(name = "IS class {0}: {1}")
	@MethodSource("isClassMutations")
	void isVerificationPerClass(char cls, String name, String old, String nu, int expected) throws Exception {
		check("IS", cls, name, old, nu, expected);
	}

	/** Same for class C (~13 s per case): tagged classC, excluded by default (see pom.xml). */
	@Tag("classC")
	@ParameterizedTest(name = "IS class {0}: {1}")
	@MethodSource("isClassCMutations")
	void isVerificationClassC(char cls, String name, String old, String nu, int expected) throws Exception {
		check("IS", cls, name, old, nu, expected);
	}

	private static Arguments c(char cls, String name, String old, String nu, int expected) {
		return Arguments.of(cls, name, old, nu, expected);
	}

	static Stream<Arguments> isClassMutations() {
		return Stream.of(
			c('W', "control: unmutated", null, null, PASS),
			c('W', "expected rango +1 (addition branch, clave 1)", "wTestRankArray[] = { 1249, 11698, 1039987, 1043896, 1048018 }", "wTestRankArray[] = { 1249, 11699, 1039987, 1043896, 1048018 }", FAIL),
			c('W', "expected rango +1 (subtraction branch, clave 3)", "wTestRankArray[] = { 1249, 11698, 1039987, 1043896, 1048018 }", "wTestRankArray[] = { 1249, 11698, 1039987, 1043897, 1048018 }", FAIL),
			c('W', "test key index changed", "wTestIndexArray[] = { 357773, 934767, 875723, 898999, 404505 }", "wTestIndexArray[] = { 357773, 934768, 875723, 898999, 404505 }", FAIL),

			c('A', "control: unmutated", null, null, PASS),
			c('A', "expected rango +1 (addition branch, clave 1)", "aTestRankArray[] = { 104, 17523, 123928, 8288932, 8388264 }", "aTestRankArray[] = { 104, 17524, 123928, 8288932, 8388264 }", FAIL),
			c('A', "expected rango +1 (subtraction branch, clave 3)", "aTestRankArray[] = { 104, 17523, 123928, 8288932, 8388264 }", "aTestRankArray[] = { 104, 17523, 123928, 8288933, 8388264 }", FAIL),
			c('A', "test key index changed", "aTestIndexArray[] = { 2112377, 662041, 5336171, 3642833, 4250760 }", "aTestIndexArray[] = { 2112377, 662042, 5336171, 3642833, 4250760 }", FAIL),

			c('B', "control: unmutated", null, null, PASS),
			c('B', "expected rango +1 (addition branch, clave 1)", "bTestRankArray[] = { 33422937, 10244, 59149, 33135281, 99 }", "bTestRankArray[] = { 33422937, 10245, 59149, 33135281, 99 }", FAIL),
			c('B', "expected rango +1 (addition branch, clave 4)", "bTestRankArray[] = { 33422937, 10244, 59149, 33135281, 99 }", "bTestRankArray[] = { 33422937, 10244, 59149, 33135281, 100 }", FAIL),
			c('B', "expected rango +1 (subtraction branch, clave 3)", "bTestRankArray[] = { 33422937, 10244, 59149, 33135281, 99 }", "bTestRankArray[] = { 33422937, 10244, 59149, 33135282, 99 }", FAIL),
			c('B', "expected rango +1 (subtraction branch, clave 0)", "bTestRankArray[] = { 33422937, 10244, 59149, 33135281, 99 }", "bTestRankArray[] = { 33422938, 10244, 59149, 33135281, 99 }", FAIL),
			c('B', "test key index changed", "bTestIndexArray[] = { 41869, 812306, 5102857, 18232239, 26860214 }", "bTestIndexArray[] = { 41869, 812307, 5102857, 18232239, 26860214 }", FAIL));
	}

	static Stream<Arguments> isClassCMutations() {
		return Stream.of(
			c('C', "control: unmutated", null, null, PASS),
			c('C', "expected rango +1 (addition branch, clave 1)", "cTestRankArray[] = { 61147, 882988, 266290, 133997595, 133525895 }", "cTestRankArray[] = { 61147, 882989, 266290, 133997595, 133525895 }", FAIL),
			c('C', "expected rango +1 (subtraction branch, clave 3)", "cTestRankArray[] = { 61147, 882988, 266290, 133997595, 133525895 }", "cTestRankArray[] = { 61147, 882988, 266290, 133997596, 133525895 }", FAIL),
			c('C', "test key index changed", "cTestIndexArray[] = { 44172927, 72999161, 74326391, 129606274, 21736814 }", "cTestIndexArray[] = { 44172927, 72999162, 74326391, 129606274, 21736814 }", FAIL));
	}

	private void check(String kernel, char cls, String name, String old, String nu, int expected) throws Exception {
		Path src = Paths.get(System.getProperty("basedir", "."), "src", "main", "java", "NPB", "SER", kernel + ".java");
		String code = new String(Files.readAllBytes(src), StandardCharsets.ISO_8859_1);
		if (old != null) {
			int first = code.indexOf(old);
			assertTrue(first >= 0, "the text to mutate is no longer in " + kernel + ".java: " + old);
			assertEquals(-1, code.indexOf(old, first + 1), "the text to mutate is not unique in " + kernel + ".java: " + old);
			code = code.substring(0, first) + nu + code.substring(first + old.length());
		}
		Path srcDir = Files.createDirectories(tmp.resolve("src").resolve("NPB").resolve("SER"));
		Path out = Files.createDirectories(tmp.resolve("out"));
		Path mutated = srcDir.resolve(kernel + ".java");
		Files.write(mutated, code.getBytes(StandardCharsets.ISO_8859_1));

		compile(mutated, out);
		int verified = run(kernel, cls, out);
		assertEquals(expected, verified, kernel + " " + cls + " [" + name + "]: expected verified " + expected
				+ " but was " + verified + (expected == FAIL && verified == PASS
						? " -> THE VERIFICATION DOES NOT DETECT THIS ERROR" : ""));
	}

	static void compile(Path file, Path outDir) throws IOException {
		JavaCompiler jc = ToolProvider.getSystemJavaCompiler();
		assertNotNull(jc, "a JDK (javax.tools) is required for this test");
		String cp = System.getProperty("surefire.test.class.path", System.getProperty("java.class.path"));
		ByteArrayOutputStream err = new ByteArrayOutputStream();
		int rc = jc.run(null, null, err, "-nowarn", "-proc:none", "--release", "8", "-encoding", "ISO-8859-1",
				"-classpath", cp, "-d", outDir.toString(), file.toString());
		assertEquals(0, rc, "the mutation does not compile:\n" + err.toString("UTF-8"));
	}

	/** Loads NPB.SER.* from outDir with priority over the classpath; everything else from the parent. */
	static final class ChildFirst extends URLClassLoader {
		ChildFirst(Path dir, ClassLoader parent) throws IOException {
			super(new URL[] { dir.toUri().toURL() }, parent);
		}

		@Override
		protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
			if (name.startsWith("NPB.SER.")) {
				synchronized (getClassLoadingLock(name)) {
					Class<?> c = findLoadedClass(name);
					if (c == null)
						c = findClass(name);
					if (resolve)
						resolveClass(c);
					return c;
				}
			}
			return super.loadClass(name, resolve);
		}
	}

	private static int run(String kernel, char cls, Path outDir) throws Exception {
		PrintStream original = System.out;
		ByteArrayOutputStream buf = new ByteArrayOutputStream();
		try (ChildFirst cl = new ChildFirst(outDir, VerificationSanityTest.class.getClassLoader())) {
			Class<?> k = cl.loadClass("NPB.SER." + kernel);
			assertEquals(cl, k.getClassLoader(), "the mutated class was not loaded");
			Benchmark bm = (Benchmark) k.getConstructor(char.class, boolean.class).newInstance(cls, false);
			System.setOut(new PrintStream(buf, true, "UTF-8"));
			bm.runBenchmark(new String[0]);
			BMResults res = bm.getResults();
			assertNotNull(res);
			return res.getVerified();
		} finally {
			System.setOut(original);
		}
	}

	@Test
	void everyMutatedKernelIsCovered() {
		// Every SER kernel has at least one FAIL case and one PASS control.
		for (String k : Arrays.asList("CG", "EP", "FT", "IS", "MG", "SP", "BT", "LU")) {
			long fails = mutations().filter(a -> a.get()[0].equals(k) && (int) a.get()[4] == FAIL).count();
			long passes = mutations().filter(a -> a.get()[0].equals(k) && (int) a.get()[4] == PASS).count();
			assertTrue(fails >= 1 && passes >= 1, k + ": missing FAIL/PASS cases");
		}
	}
}
