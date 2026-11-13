package NPB.MPJ;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Sanity check of the verification of each MPJ kernel: verification must FAIL when it should (counterpart of
 * {@code NPB.SER.VerificationSanityTest}). For each case the real source of {@code NPB/MPJ/<kernel>.java} is taken and
 * ONE mutation is applied (the text to replace must appear exactly once, so a refactor cannot silently disable the
 * test); the result is compiled to a temporary directory and launched with {@code fmpjrun} with that directory first
 * in the classpath (the mutated class shadows the one in {@code bin}), class S:
 * <ul>
 * <li>perturbed expected value (far above epsilon) -> {@code UNSUCCESSFUL};</li>
 * <li>epsilon edge: slightly below -> {@code SUCCESSFUL}, slightly above -> {@code UNSUCCESSFUL};</li>
 * <li>NaN in the computed or the expected value -> {@code UNSUCCESSFUL} (since {@code NaN > eps == false}, a badly
 * written comparison would accept it);</li>
 * <li>controls (unmutated, or within tolerance) -> {@code SUCCESSFUL}, to show that the setup does not break
 * everything by itself.</li>
 * </ul>
 * Requires FastMPJ ({@code FMPJ_HOME}, with {@code lib/mpj.jar}), a JDK and the compiled MPJ classes (profile
 * {@code mpj}); if anything is missing the test is skipped. An explicit {@code UNSUCCESSFUL} is required; "not
 * SUCCESSFUL" is not enough (an exception does not count as a failed verification).
 */
class MpjVerificationSanityTest {
	/** "--release=8" where javac supports it (JDK 9 or later); JDK 8 compiles for 8 anyway. */
	private static String release() {
		return System.getProperty("java.specification.version").startsWith("1.") ? "-Xlint:none" : "--release=8";
	}

	private static final boolean FAIL = false, PASS = true;
	private static final Pattern VERIFICATION = Pattern.compile("Verification\\s*=\\s*(\\w+)");

	/** kernel, name, processes, original text (null = unmutated), new text, expected. */
	private static Arguments m(String kernel, String name, int np, String old, String nu, boolean expected) {
		return Arguments.of(kernel, name, np, old, nu, expected);
	}

	static Stream<Arguments> mutations() {
		final String NAN = "Double.NaN";
		return Stream.of(
			// ---- CG: epsilon 1e-10, final zeta
			m("CG", "control: unmutated", 1, null, null, PASS),
			m("CG", "control: unmutated, 4 processes", 4, null, null, PASS),
			m("CG", "expected zeta +1e-6", 1, "zetaVerifyValue = 8.5971775078648;", "zetaVerifyValue = 8.5971775078648 * (1 + 1e-6);", FAIL),
			m("CG", "expected zeta +5e-11 (within eps)", 1, "zetaVerifyValue = 8.5971775078648;", "zetaVerifyValue = 8.5971775078648 * (1 + 5e-11);", PASS),
			m("CG", "expected zeta +2e-10 (outside eps)", 1, "zetaVerifyValue = 8.5971775078648;", "zetaVerifyValue = 8.5971775078648 * (1 + 2e-10);", FAIL),
			m("CG", "computed zeta NaN", 1, "Math.abs(zeta - zetaVerifyValue) / zetaVerifyValue", "Math.abs(" + NAN + " - zetaVerifyValue) / zetaVerifyValue", FAIL),
			m("CG", "expected zeta NaN", 1, "zetaVerifyValue = 8.5971775078648;", "zetaVerifyValue = " + NAN + ";", FAIL),

			// ---- EP: epsilon 1e-8; sx, sy and gc
			m("EP", "control: unmutated", 1, null, null, PASS),
			m("EP", "control: unmutated, 3 processes", 3, null, null, PASS),
			m("EP", "expected sx +1e-6", 1, "sxVerifyValue = 1.051299420395306E+07;", "sxVerifyValue = 1.051299420395306E+07 * (1 + 1e-6);", FAIL),
			m("EP", "expected sx +5e-9 (within eps)", 1, "sxVerifyValue = 1.051299420395306E+07;", "sxVerifyValue = 1.051299420395306E+07 * (1 + 5e-9);", PASS),
			m("EP", "expected sx +2e-8 (outside eps)", 1, "sxVerifyValue = 1.051299420395306E+07;", "sxVerifyValue = 1.051299420395306E+07 * (1 + 2e-8);", FAIL),
			m("EP", "expected sy +1e-6", 1, "syVerifyValue = 1.051517131857535E+07;", "syVerifyValue = 1.051517131857535E+07 * (1 + 1e-6);", FAIL),
			m("EP", "expected gc +1 (the count is verified)", 1, "gcVerifyValue = 13176389.0;", "gcVerifyValue = 13176390.0;", FAIL),
			m("EP", "computed sx NaN", 1, "Math.abs((sx - sxVerifyValue) / sxVerifyValue)", "Math.abs((" + NAN + " - sxVerifyValue) / sxVerifyValue)", FAIL),
			m("EP", "computed sy NaN", 1, "Math.abs((sy - syVerifyValue) / syVerifyValue)", "Math.abs((" + NAN + " - syVerifyValue) / syVerifyValue)", FAIL),
			m("EP", "computed gc NaN", 1, "Math.abs((gc - gcVerifyValue) / gcVerifyValue)", "Math.abs((" + NAN + " - gcVerifyValue) / gcVerifyValue)", FAIL),
			m("EP", "expected sx NaN", 1, "sxVerifyValue = 1.051299420395306E+07;", "sxVerifyValue = " + NAN + ";", FAIL),

			// ---- FT: epsilon 1e-12; checksum of each iteration (interleaved in a single array)
			m("FT", "control: unmutated", 1, null, null, PASS),
			m("FT", "control: unmutated, 4 processes", 4, null, null, PASS),
			m("FT", "expected checksum +2e-8", 1, "5.546385409189E+02", "5.546385509189E+02", FAIL),
			m("FT", "expected checksum +5e-13 (within eps)", 1, "5.546385409189E+02", "5.546385409189E+02 * 1.0000000000005", PASS),
			m("FT", "expected checksum +2e-12 (outside eps)", 1, "5.546385409189E+02", "5.546385409189E+02 * 1.000000000002", FAIL),
			m("FT", "computed real checksum NaN", 1, "sums[(i - 1) << 1] = allchk[REAL];", "sums[(i - 1) << 1] = " + NAN + ";", FAIL),
			m("FT", "computed imaginary checksum NaN", 1, "sums[IMAG + ((i - 1) << 1)] = allchk[IMAG];", "sums[IMAG + ((i - 1) << 1)] = " + NAN + ";", FAIL),

			// ---- IS: integer verification (5 ranks x 10 iterations + full sort); 2 processes sharing the work
			m("IS", "control: unmutated", 1, null, null, PASS),
			m("IS", "control: unmutated, 2 processes", 2, null, null, PASS),
			m("IS", "expected rank +1", 1, "sTestRankArray[] = { 0, 18, 346, 64917, 65463 }", "sTestRankArray[] = { 0, 19, 346, 64917, 65463 }", FAIL),
			m("IS", "expected rank +1, 2 processes", 2, "sTestRankArray[] = { 0, 18, 346, 64917, 65463 }", "sTestRankArray[] = { 0, 19, 346, 64917, 65463 }", FAIL),
			m("IS", "test key index changed", 1, "sTestIndexArray[] = { 48427, 17148, 23627, 62548, 4431 }", "sTestIndexArray[] = { 48427, 17149, 23627, 62548, 4431 }", FAIL),
			m("IS", "invalid full sort", 1, "if (keyArray[i-1] > keyArray[i])", "if (keyArray[i-1] >= keyArray[i])", FAIL),
			m("IS", "invalid full sort, 2 processes", 2, "if (keyArray[i-1] > keyArray[i])", "if (keyArray[i-1] >= keyArray[i])", FAIL),
			m("IS", "control: unmutated, 4 processes", 4, null, null, PASS),
			m("IS", "control: unmutated, 8 processes", 8, null, null, PASS),
			m("IS", "expected rank +1, 4 processes", 4, "sTestRankArray[] = { 0, 18, 346, 64917, 65463 }", "sTestRankArray[] = { 0, 19, 346, 64917, 65463 }", FAIL),
			m("IS", "expected rank +1 (last key), 4 processes", 4, "sTestRankArray[] = { 0, 18, 346, 64917, 65463 }", "sTestRankArray[] = { 0, 18, 346, 64917, 65464 }", FAIL),
			m("IS", "expected rank +1, 8 processes", 8, "sTestRankArray[] = { 0, 18, 346, 64917, 65463 }", "sTestRankArray[] = { 0, 19, 346, 64917, 65463 }", FAIL),
			m("IS", "expected rank +1 (last key), 8 processes", 8, "sTestRankArray[] = { 0, 18, 346, 64917, 65463 }", "sTestRankArray[] = { 0, 18, 346, 64917, 65464 }", FAIL),
			m("IS", "invalid full sort, 8 processes", 8, "if (keyArray[i-1] > keyArray[i])", "if (keyArray[i-1] >= keyArray[i])", FAIL),
			m("IS", "required verification count +1, 4 processes", 4, "if (passedVerification == 5 * MAX_ITERATIONS + numProcess)", "if (passedVerification == 5 * MAX_ITERATIONS + numProcess + 1)", FAIL),
			m("IS", "required verification count +1", 2, "if (passedVerification == 5 * MAX_ITERATIONS + numProcess)", "if (passedVerification == 5 * MAX_ITERATIONS + numProcess + 1)", FAIL),

			// ---- MG: epsilon 1e-8; final L2 norm
			m("MG", "control: unmutated", 1, null, null, PASS),
			m("MG", "control: unmutated, 4 processes", 4, null, null, PASS),
			m("MG", "expected norm +1e-6", 1, "verifyValue = 0.530770700573E-4;", "verifyValue = 0.530770700573E-4 * (1 + 1e-6);", FAIL),
			m("MG", "expected norm +5e-9 (within eps)", 1, "verifyValue = 0.530770700573E-4;", "verifyValue = 0.530770700573E-4 * (1 + 5e-9);", PASS),
			m("MG", "expected norm +2e-8 (outside eps)", 1, "verifyValue = 0.530770700573E-4;", "verifyValue = 0.530770700573E-4 * (1 + 2e-8);", FAIL),
			m("MG", "computed norm NaN", 1, "Math.abs(rnm[0]-verifyValue)/verifyValue", "Math.abs(" + NAN + "-verifyValue)/verifyValue", FAIL),
			m("MG", "expected norm NaN", 1, "verifyValue = 0.530770700573E-4;", "verifyValue = " + NAN + ";", FAIL),

			// ---- SP: epsilon 1e-8; 5 residual norms, 5 error norms and dt
			m("SP", "control: unmutated", 1, null, null, PASS),
			m("SP", "control: unmutated, 4 processes", 4, null, null, PASS),
			m("SP", "expected xcr +1e-6", 1, "xcrref[0] = 2.7470315451339479E-2;", "xcrref[0] = 2.7470315451339479E-2 * (1 + 1e-6);", FAIL),
			m("SP", "expected xcr +5e-9 (within eps)", 1, "xcrref[0] = 2.7470315451339479E-2;", "xcrref[0] = 2.7470315451339479E-2 * (1 + 5e-9);", PASS),
			m("SP", "expected xcr +2e-8 (outside eps)", 1, "xcrref[0] = 2.7470315451339479E-2;", "xcrref[0] = 2.7470315451339479E-2 * (1 + 2e-8);", FAIL),
			m("SP", "expected xce +1e-6", 1, "xceref[0] = 2.7289258557377227E-5;", "xceref[0] = 2.7289258557377227E-5 * (1 + 1e-6);", FAIL),
			m("SP", "dt different from the reference", 1, "dtref = 1.5E-2;", "dtref = 1.6E-2;", FAIL),
			m("SP", "computed xcr NaN", 1, "xcrdif[m] = Math.abs((xcr[m]-xcrref[m])/xcrref[m]);", "xcrdif[m] = Math.abs((" + NAN + "-xcrref[m])/xcrref[m]);", FAIL),
			m("SP", "computed xce NaN", 1, "xcedif[m] = Math.abs((xce[m]-xceref[m])/xceref[m]);", "xcedif[m] = Math.abs((" + NAN + "-xceref[m])/xceref[m]);", FAIL),
			m("SP", "expected xcr NaN", 1, "xcrref[3] = 1.5840557224455615E-2;", "xcrref[3] = " + NAN + ";", FAIL),

			// ---- BT: epsilon 1e-8; 5 residual norms, 5 error norms and dt (only process 0 prints and verifies)
			m("BT", "control: unmutated", 1, null, null, PASS),
			m("BT", "control: unmutated, 4 processes", 4, null, null, PASS),
			m("BT", "expected xcr +1e-6", 1, "xcrref[0] = 1.7034283709541311e-01;", "xcrref[0] = 1.7034283709541311e-01 * (1 + 1e-6);", FAIL),
			m("BT", "expected xcr +5e-9 (within eps)", 1, "xcrref[0] = 1.7034283709541311e-01;", "xcrref[0] = 1.7034283709541311e-01 * (1 + 5e-9);", PASS),
			m("BT", "expected xcr +2e-8 (outside eps)", 1, "xcrref[0] = 1.7034283709541311e-01;", "xcrref[0] = 1.7034283709541311e-01 * (1 + 2e-8);", FAIL),
			m("BT", "expected xcr +1e-6, 4 processes", 4, "xcrref[0] = 1.7034283709541311e-01;", "xcrref[0] = 1.7034283709541311e-01 * (1 + 1e-6);", FAIL),
			m("BT", "expected xce +1e-6", 1, "xceref[0] = 4.9976913345811579e-04;", "xceref[0] = 4.9976913345811579e-04 * (1 + 1e-6);", FAIL),
			m("BT", "expected xce +5e-9 (within eps)", 1, "xceref[0] = 4.9976913345811579e-04;", "xceref[0] = 4.9976913345811579e-04 * (1 + 5e-9);", PASS),
			m("BT", "expected xce +2e-8 (outside eps)", 1, "xceref[0] = 4.9976913345811579e-04;", "xceref[0] = 4.9976913345811579e-04 * (1 + 2e-8);", FAIL),
			m("BT", "dt different from the reference", 1, "dtref = 1.0e-2;", "dtref = 1.1e-2;", FAIL),
			m("BT", "reference dt +5e-9 (within eps)", 1, "dtref = 1.0e-2;", "dtref = 1.0e-2 + 5e-9;", PASS),
			m("BT", "reference dt +2e-8 (outside eps)", 1, "dtref = 1.0e-2;", "dtref = 1.0e-2 + 2e-8;", FAIL),
			m("BT", "computed dt NaN", 1, "Math.abs(dt-dtref) <= epsilon", "Math.abs(" + NAN + "-dtref) <= epsilon", FAIL),
			m("BT", "computed xcr NaN", 1, "xcrdif[m] = Math.abs((xcr[m]-xcrref[m])/xcrref[m]);", "xcrdif[m] = Math.abs((" + NAN + "-xcrref[m])/xcrref[m]);", FAIL),
			m("BT", "computed xce NaN", 1, "xcedif[m] = Math.abs((xce[m]-xceref[m])/xceref[m]);", "xcedif[m] = Math.abs((" + NAN + "-xceref[m])/xceref[m]);", FAIL),
			m("BT", "expected xcr NaN", 1, "xcrref[3] = 2.6436421275166801e-02;", "xcrref[3] = " + NAN + ";", FAIL),
			m("BT", "expected xce NaN", 1, "xceref[3] = 7.3821238632439731e-05;", "xceref[3] = " + NAN + ";", FAIL),

			// ---- LU: epsilon 1e-8; 5 residual norms, 5 error norms, the surface integral and dt (only process 0 verifies)
			m("LU", "control: unmutated", 1, null, null, PASS),
			m("LU", "control: unmutated, 4 processes", 4, null, null, PASS),
			m("LU", "control: unmutated, 6 processes (3x2 grid)", 6, null, null, PASS),
			m("LU", "expected xcr +1e-6", 1, "xcrref[0] = 1.6196343210976702e-02;", "xcrref[0] = 1.6196343210976702e-02 * (1 + 1e-6);", FAIL),
			m("LU", "expected xcr +5e-9 (within eps)", 1, "xcrref[0] = 1.6196343210976702e-02;", "xcrref[0] = 1.6196343210976702e-02 * (1 + 5e-9);", PASS),
			m("LU", "expected xcr +2e-8 (outside eps)", 1, "xcrref[0] = 1.6196343210976702e-02;", "xcrref[0] = 1.6196343210976702e-02 * (1 + 2e-8);", FAIL),
			m("LU", "expected xcr +1e-6, 4 processes", 4, "xcrref[0] = 1.6196343210976702e-02;", "xcrref[0] = 1.6196343210976702e-02 * (1 + 1e-6);", FAIL),
			m("LU", "expected xce +1e-6", 1, "xceref[0] = 6.4223319957960924e-04;", "xceref[0] = 6.4223319957960924e-04 * (1 + 1e-6);", FAIL),
			m("LU", "expected xce +5e-9 (within eps)", 1, "xceref[0] = 6.4223319957960924e-04;", "xceref[0] = 6.4223319957960924e-04 * (1 + 5e-9);", PASS),
			m("LU", "expected xce +2e-8 (outside eps)", 1, "xceref[0] = 6.4223319957960924e-04;", "xceref[0] = 6.4223319957960924e-04 * (1 + 2e-8);", FAIL),
			m("LU", "expected xci +1e-6", 1, "xciref = 7.8418928865937083e+00;", "xciref = 7.8418928865937083e+00 * (1 + 1e-6);", FAIL),
			m("LU", "expected xci +5e-9 (within eps)", 1, "xciref = 7.8418928865937083e+00;", "xciref = 7.8418928865937083e+00 * (1 + 5e-9);", PASS),
			m("LU", "expected xci +2e-8 (outside eps)", 1, "xciref = 7.8418928865937083e+00;", "xciref = 7.8418928865937083e+00 * (1 + 2e-8);", FAIL),
			m("LU", "computed xcr NaN", 1, "xcrdif[m] = Math.abs((rsdnm[m]-xcrref[m])/xcrref[m]);", "xcrdif[m] = Math.abs((" + NAN + "-xcrref[m])/xcrref[m]);", FAIL),
			m("LU", "computed xce NaN", 1, "xcedif[m] = Math.abs((errnm[m]-xceref[m])/xceref[m]);", "xcedif[m] = Math.abs((" + NAN + "-xceref[m])/xceref[m]);", FAIL),
			m("LU", "computed xci NaN", 1, "xcidif = Math.abs((frc - xciref)/xciref);", "xcidif = Math.abs((" + NAN + " - xciref)/xciref);", FAIL),
			m("LU", "expected xcr NaN", 1, "xcrref[0] = 1.6196343210976702e-02;", "xcrref[0] = " + NAN + ";", FAIL),
			m("LU", "expected xce NaN", 1, "xceref[0] = 6.4223319957960924e-04;", "xceref[0] = " + NAN + ";", FAIL),
			m("LU", "expected xci NaN", 1, "xciref = 7.8418928865937083e+00;", "xciref = " + NAN + ";", FAIL),
			m("LU", "reference dt +1e-3", 1, "dtref = 5.0e-1;", "dtref = 5.1e-1;", FAIL),
			m("LU", "reference dt +5e-9 (within eps)", 1, "dtref = 5.0e-1;", "dtref = 5.0e-1 + 5e-9;", PASS),
			m("LU", "reference dt +2e-8 (outside eps)", 1, "dtref = 5.0e-1;", "dtref = 5.0e-1 + 2e-8;", FAIL),
			m("LU", "computed dt NaN", 1, "Math.abs(dt-dtref) <= epsilon", "Math.abs(" + NAN + "-dtref) <= epsilon", FAIL));
	}

	@TempDir
	Path tmp;

	@ParameterizedTest(name = "{0}: {1}")
	@MethodSource("mutations")
	@Timeout(value = 10, unit = TimeUnit.MINUTES)
	void verificationBehavesUnderMutation(String kernel, String name, int np, String old, String nu, boolean expected)
			throws Exception {
		check(kernel, "S", name, np, old, nu, expected);
	}

	/**
	 * IS: the partial verification has one branch per class (add or subtract the iteration depending on the position
	 * of the test key) and the other tests only exercise the one for S. Here, for each class, an expected value on
	 * each side of the branch and a test key index are mutated, plus an unmutated control. The D and E branches
	 * cannot be run on a typical machine.
	 */
	@ParameterizedTest(name = "IS class {0}: {1} np={2}")
	@MethodSource("isClassMutations")
	@Timeout(value = 15, unit = TimeUnit.MINUTES)
	void isVerificationPerClass(String cls, String name, int np, String old, String nu, boolean expected)
			throws Exception {
		check("IS", cls, name, np, old, nu, expected);
	}

	/** Same for class C (tens of seconds per case): tag classC, excluded by default (see pom.xml). */
	@Tag("classC")
	@ParameterizedTest(name = "IS class {0}: {1} np={2}")
	@MethodSource("isClassCMutations")
	@Timeout(value = 30, unit = TimeUnit.MINUTES)
	void isVerificationClassC(String cls, String name, int np, String old, String nu, boolean expected)
			throws Exception {
		check("IS", cls, name, np, old, nu, expected);
	}

	private static Arguments c(String cls, String name, int np, String old, String nu, boolean expected) {
		return Arguments.of(cls, name, np, old, nu, expected);
	}

	static Stream<Arguments> isClassMutations() {
		final String W = "wTestRankArray[] = { 1249, 11698, 1039987, 1043896, 1048018 }";
		final String A = "aTestRankArray[] = { 104, 17523, 123928, 8288932, 8388264 }";
		final String B = "bTestRankArray[] = { 33422937, 10244, 59149, 33135281, 99 }";
		return Stream.of(
			c("W", "control: unmutated", 1, null, null, PASS),
			c("W", "expected rank +1 (add branch, key 1)", 1, W, "wTestRankArray[] = { 1249, 11699, 1039987, 1043896, 1048018 }", FAIL),
			c("W", "expected rank +1 (subtract branch, key 3)", 1, W, "wTestRankArray[] = { 1249, 11698, 1039987, 1043897, 1048018 }", FAIL),
			c("W", "test key index changed", 1, "wTestIndexArray[] = { 357773, 934767, 875723, 898999, 404505 }", "wTestIndexArray[] = { 357773, 934768, 875723, 898999, 404505 }", FAIL),

			c("A", "control: unmutated", 1, null, null, PASS),
			c("A", "control: unmutated", 2, null, null, PASS),
			c("A", "expected rank +1 (add branch, key 1)", 1, A, "aTestRankArray[] = { 104, 17524, 123928, 8288932, 8388264 }", FAIL),
			c("A", "expected rank +1 (subtract branch, key 3)", 1, A, "aTestRankArray[] = { 104, 17523, 123928, 8288933, 8388264 }", FAIL),
			c("A", "expected rank +1 (add branch, key 1)", 2, A, "aTestRankArray[] = { 104, 17524, 123928, 8288932, 8388264 }", FAIL),
			c("A", "expected rank +1 (subtract branch, key 3)", 2, A, "aTestRankArray[] = { 104, 17523, 123928, 8288933, 8388264 }", FAIL),
			c("A", "test key index changed", 1, "aTestIndexArray[] = { 2112377, 662041, 5336171, 3642833, 4250760 }", "aTestIndexArray[] = { 2112377, 662042, 5336171, 3642833, 4250760 }", FAIL),

			c("B", "control: unmutated", 1, null, null, PASS),
			c("B", "expected rank +1 (add branch, key 1)", 1, B, "bTestRankArray[] = { 33422937, 10245, 59149, 33135281, 99 }", FAIL),
			c("B", "expected rank +1 (add branch, key 4)", 1, B, "bTestRankArray[] = { 33422937, 10244, 59149, 33135281, 100 }", FAIL),
			c("B", "expected rank +1 (subtract branch, key 3)", 1, B, "bTestRankArray[] = { 33422937, 10244, 59149, 33135282, 99 }", FAIL),
			c("B", "expected rank +1 (subtract branch, key 0)", 1, B, "bTestRankArray[] = { 33422938, 10244, 59149, 33135281, 99 }", FAIL),
			c("B", "test key index changed", 1, "bTestIndexArray[] = { 41869, 812306, 5102857, 18232239, 26860214 }", "bTestIndexArray[] = { 41869, 812307, 5102857, 18232239, 26860214 }", FAIL));
	}

	static Stream<Arguments> isClassCMutations() {
		final String C = "cTestRankArray[] = { 61147, 882988, 266290, 133997595, 133525895 }";
		return Stream.of(
			c("C", "control: unmutated", 1, null, null, PASS),
			c("C", "expected rank +1 (add branch, key 1)", 1, C, "cTestRankArray[] = { 61147, 882989, 266290, 133997595, 133525895 }", FAIL),
			c("C", "expected rank +1 (subtract branch, key 3)", 1, C, "cTestRankArray[] = { 61147, 882988, 266290, 133997596, 133525895 }", FAIL),
			c("C", "test key index changed", 1, "cTestIndexArray[] = { 44172927, 72999161, 74326391, 129606274, 21736814 }", "cTestIndexArray[] = { 44172927, 72999162, 74326391, 129606274, 21736814 }", FAIL));
	}

	private void check(String kernel, String cls, String name, int np, String old, String nu, boolean expected)
			throws Exception {
		File fmpjrun = MpjLaunchTest.fastMpj();
		File bin = MpjLaunchTest.classes();
		File mpjJar = new File(fmpjrun.getParentFile().getParentFile(), "lib/mpj.jar");
		assumeTrue(mpjJar.exists(), "does not exist: " + mpjJar);

		Path src = Paths.get(System.getProperty("basedir", "."), "src", "main", "java", "NPB", "MPJ", kernel + ".java");
		String code = new String(Files.readAllBytes(src), StandardCharsets.ISO_8859_1);
		if (old != null) {
			int first = code.indexOf(old);
			assertTrue(first >= 0, "the text to mutate is no longer in " + kernel + ".java: " + old);
			assertEquals(-1, code.indexOf(old, first + 1), "the text to mutate is not unique in " + kernel + ".java: " + old);
			code = code.substring(0, first) + nu + code.substring(first + old.length());
		}
		Path srcDir = Files.createDirectories(tmp.resolve("src").resolve("NPB").resolve("MPJ"));
		Path out = Files.createDirectories(tmp.resolve("out"));
		Path mutated = srcDir.resolve(kernel + ".java");
		Files.write(mutated, code.getBytes(StandardCharsets.ISO_8859_1));

		JavaCompiler jc = ToolProvider.getSystemJavaCompiler();
		assertNotNull(jc, "a JDK (javax.tools) is required for this test");
		ByteArrayOutputStream err = new ByteArrayOutputStream();
		int rc = jc.run(null, null, err, "-nowarn", "-proc:none", release(), "-encoding", "ISO-8859-1",
				"-classpath", bin.getAbsolutePath() + File.pathSeparator + mpjJar.getAbsolutePath(), "-d",
				out.toString(), mutated.toString());
		assertEquals(0, rc, "the mutation does not compile:\n" + err.toString("UTF-8"));

		String output = MpjLaunchTest.run(kernel, cls, np, Collections.<String, String>emptyMap(), out.toString());
		Matcher v = VERIFICATION.matcher(output);
		String status = v.find() ? v.group(1) : "(no verification line)";
		String want = expected ? "SUCCESSFUL" : "UNSUCCESSFUL";
		assertEquals(want, status, kernel + " " + cls + " [" + name + "] np=" + np + ": expected " + want + " but was " + status
				+ (!expected && "SUCCESSFUL".equals(status) ? " -> VERIFICATION DOES NOT DETECT THIS ERROR" : "") + "\n"
				+ MpjLaunchTest.tail(output));
	}
}
