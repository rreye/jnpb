package NPB.MPJ;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Tests of the MPJ versions: launch each benchmark with {@code fmpjrun} (smdev device,
 * single node) and check the output. They do not depend on FastMPJ at compile time, but
 * skip themselves if there is no installation: {@code FMPJ_HOME} (or {@code -Dfmpj.home})
 * and the compiled MPJ classes (profile {@code mpj}) are required.
 *
 * <ul>
 * <li>Kernel x class x number of processes matrix: {@code Verification = SUCCESSFUL}.</li>
 * <li>Invalid number of processes: rejected by default; with NPB_NPROCS_STRICT=off the
 * surplus processes are masked and the run verifies.</li>
 * <li>Unimplemented classes or too few processes: rejected with a clear message.</li>
 * </ul>
 * Class C and the E classes are separate ({@code -Dgroups=classC}); see the pom.
 */
class MpjLaunchTest {

	private static final Pattern VERIFICATION = Pattern.compile("Verification\\s*=\\s*(\\w+)");

	static File fastMpj() {
		String home = System.getProperty("fmpj.home");
		if (home == null || home.isEmpty())
			home = System.getenv("FMPJ_HOME");
		assumeTrue(home != null && !home.isEmpty(), "FMPJ_HOME not defined: MPJ tests skipped");
		File run = new File(home, "bin/fmpjrun");
		assumeTrue(run.canExecute(), "does not exist: " + run);
		return run;
	}

	static File classes() {
		File bin = new File(System.getProperty("basedir", "."), "bin");
		assumeTrue(new File(bin, "NPB/MPJ/IS.class").exists(), "MPJ classes not compiled (profile mpj)");
		return bin;
	}

	/** Heap of each run: 2g for S, W and A; {@code -Dtest.mpj.xmx.big} (8g) for B and C (FT B needs several GB). */
	private static String heap(String cls) {
		return "B".equals(cls) || "C".equals(cls) ? System.getProperty("test.mpj.xmx.big", "8g") : "2g";
	}

	/** Runs a kernel and returns stdout+stderr. */
	static String run(String kernel, String cls, int np, Map<String, String> env) throws Exception {
		return run(kernel, cls, np, env, null);
	}

	/**
	 * Like {@link #run(String, String, int, Map)}, with {@code extraClasspath} (if not null) placed before the
	 * compiled classes: used to launch a mutated class (MpjVerificationSanityTest).
	 */
	static String run(String kernel, String cls, int np, Map<String, String> env, String extraClasspath) throws Exception {
		return run(kernel, cls, np, env, extraClasspath, Collections.<String>emptyList());
	}

	/** Same, with extra arguments for the kernel (e.g. {@code GRAPH=SH} for DT). */
	static String run(String kernel, String cls, int np, Map<String, String> env, String extraClasspath,
			List<String> appArgs) throws Exception {
		File run = fastMpj();
		File bin = classes();
		String classpath = (extraClasspath == null ? "" : extraClasspath + File.pathSeparator) + bin.getAbsolutePath();
		List<String> cmd = new ArrayList<>();
		cmd.add(run.getPath());
		cmd.addAll(java.util.Arrays.asList("-np", String.valueOf(np), "-dev", "smdev", "-Xmx" + heap(cls), "-cp",
				classpath, "-class", "NPB.MPJ." + kernel, "CLASS=" + cls));
		cmd.addAll(appArgs);
		/*
		 * fmpjrun takes classes from the working directory BEFORE those in -cp: with cwd = bin a mutated class placed
		 * first in the classpath would never be loaded. With an extra classpath it is launched from that directory.
		 */
		File cwd = extraClasspath == null ? bin : new File(extraClasspath);
		ProcessBuilder pb = new ProcessBuilder(cmd).directory(cwd).redirectErrorStream(true);
		pb.environment().remove("NPB_NPROCS_STRICT");
		pb.environment().putAll(env);
		Process p = pb.start();
		java.io.ByteArrayOutputStream buf = new java.io.ByteArrayOutputStream();
		byte[] chunk = new byte[8192];
		for (int n; (n = p.getInputStream().read(chunk)) > 0;)
			buf.write(chunk, 0, n);
		String out = buf.toString("UTF-8");
		assertTrue(p.waitFor(10, TimeUnit.MINUTES), kernel + " " + cls + " np=" + np + ": timeout");
		return out;
	}

	static String verification(String out) {
		Matcher m = VERIFICATION.matcher(out);
		return m.find() ? m.group(1) : null;
	}

	private static Arguments a(String kernel, String cls, int np) {
		return Arguments.of(kernel, cls, np);
	}

	static Stream<Arguments> matrix() {
		Stream.Builder<Arguments> b = Stream.builder();
		for (String cls : new String[] { "S", "W" }) {
			for (String k : new String[] { "CG", "EP", "FT", "IS", "MG" })
				for (int np : new int[] { 1, 2, 4, 8 })
					b.add(a(k, cls, np));
			for (int np : new int[] { 1, 4, 9, 16 })
				b.add(a("SP", cls, np));
			for (int np : new int[] { 1, 4, 9, 16 })
				b.add(a("BT", cls, np));
			// LU: np1 x np2 grid con np2 <= np1 <= 2*np2 (not only powers of 2: 6 = 3x2, 9 = 3x3)
			for (int np : new int[] { 1, 2, 4, 6, 9, 16 })
				b.add(a("LU", cls, np));
		}
		// Regression: MG failed with >= 64 processes (give3Ex sent the whole row instead of 2 values)
		b.add(a("MG", "S", 64));
		b.add(a("MG", "W", 64));
		// IS: sizes and process counts larger than the matrix above. With 1..64 processes the global
		// sorted array is identical to that of NPB-MPI (digest compared against it);
		// 128 is MAX_PROCS for class S.
		for (int np : new int[] { 16, 32, 64, 128 })
			b.add(a("IS", "S", np));
		for (int np : new int[] { 16, 64 })
			b.add(a("IS", "W", np));
		for (int np : new int[] { 1, 4, 16, 64 })
			b.add(a("IS", "A", np));
		return b.build();
	}

	@ParameterizedTest(name = "{0} class {1} np={2}")
	@MethodSource("matrix")
	@Timeout(value = 15, unit = TimeUnit.MINUTES)
	void verifies(String kernel, String cls, int np) throws Exception {
		String out = run(kernel, cls, np, Collections.<String, String>emptyMap());
		assertTrue("SUCCESSFUL".equals(verification(out)),
				kernel + " " + cls + " np=" + np + ": verification not SUCCESSFUL\n" + tail(out));
	}

	/** Kernels with a power-of-2 topology (CG, FT, MG, IS), square (SP, BT) or grid (LU). */
	static Stream<Arguments> badCounts() {
		return Stream.of(a("CG", "S", 3), a("FT", "S", 6), a("MG", "S", 7), a("IS", "S", 3),
				a("SP", "S", 2), a("SP", "S", 5), a("BT", "S", 2), a("BT", "S", 5),
				a("LU", "S", 3), a("LU", "S", 5), a("LU", "S", 7));
	}

	@ParameterizedTest(name = "{0} class {1} np={2}")
	@MethodSource("badCounts")
	@Timeout(value = 5, unit = TimeUnit.MINUTES)
	void invalidProcessCountIsRejectedByDefault(String kernel, String cls, int np) throws Exception {
		String out = run(kernel, cls, np, Collections.<String, String>emptyMap());
		assertTrue(verification(out) == null || !"SUCCESSFUL".equals(verification(out)),
				kernel + " np=" + np + " should not verify without NPB_NPROCS_STRICT=off");
		assertTrue(out.contains("IncorrectNumProcessException"), kernel + " np=" + np + ": rejection missing\n" + tail(out));
	}

	@ParameterizedTest(name = "{0} class {1} np={2}")
	@MethodSource("badCounts")
	@Timeout(value = 5, unit = TimeUnit.MINUTES)
	void excessProcessesAreMaskedWhenStrictIsOff(String kernel, String cls, int np) throws Exception {
		String out = run(kernel, cls, np, Collections.singletonMap("NPB_NPROCS_STRICT", "off"));
		assertTrue("SUCCESSFUL".equals(verification(out)),
				kernel + " np=" + np + " with STRICT=off: does not verify\n" + tail(out));
		assertTrue(out.contains("Total processes =\t" + np), kernel + ": does not report " + np + " total processes");
		assertTrue(out.contains("Active processes") || kernel.equals("SP") || kernel.equals("BT") || kernel.equals("LU") || kernel.equals("CG") || kernel.startsWith("FT")
				|| kernel.equals("MG"), kernel + ": missing 'Active processes'");
	}

	/** Classes that a kernel does not implement, or that require more processes than given. */
	static Stream<Arguments> unavailable() {
		return Stream.of(a("IS", "E", 8), a("FT", "E", 4), a("MG", "E", 4), a("IS", "D", 2));
	}

	@ParameterizedTest(name = "{0} class {1} np={2}")
	@MethodSource("unavailable")
	@Timeout(value = 5, unit = TimeUnit.MINUTES)
	void classNeedingMoreProcessesIsRejectedClearly(String kernel, String cls, int np) throws Exception {
		String out = run(kernel, cls, np, Collections.<String, String>emptyMap());
		assertTrue(out.contains("IncorrectNumProcessException"), kernel + " " + cls + " np=" + np + ":\n" + tail(out));
	}

	/** Class C kernels with several processes: actually run, but slow (EP takes ~8 s at 16 processes). */
	@Tag("classC")
	@ParameterizedTest(name = "{0} class C np={1}")
	@MethodSource("classCCases")
	@Timeout(value = 30, unit = TimeUnit.MINUTES)
	void verifiesClassC(String kernel, int np) throws Exception {
		String out = run(kernel, "C", np, Collections.<String, String>emptyMap());
		assertTrue("SUCCESSFUL".equals(verification(out)), kernel + " C np=" + np + ":\n" + tail(out));
	}

	static Stream<Arguments> classCCases() {
		return Stream.of(Arguments.of("EP", 8), Arguments.of("IS", 4), Arguments.of("CG", 4), Arguments.of("MG", 8));
	}

	/**
	 * SP reads the optional file inputsp.data from the working directory. The official sample writes dt in Fortran
	 * form (0.0015d0), which Double.valueOf rejects: regression test for that failure.
	 */
	@Test
	@Timeout(value = 5, unit = TimeUnit.MINUTES)
	void spReadsInputFileWithFortranDt(@TempDir Path dir) throws Exception {
		Files.write(dir.resolve("inputsp.data"),
				java.util.Arrays.asList("100       number of time steps", "0.015d0   dt", "12 12 12"));
		String out = run("SP", "S", 1, Collections.<String, String>emptyMap(), dir.toString());
		assertTrue(out.contains("Reading from input file inputsp.data"), "does not read inputsp.data:\n" + tail(out));
		assertTrue("SUCCESSFUL".equals(verification(out)), "SP with inputsp.data:\n" + tail(out));
	}

	/**
	 * BT reads the optional file inputbt.data from the working directory, with dt in Fortran form (0.01d0): regression
	 * test for the same failure as SP (Double.valueOf rejects the d).
	 */
	@Test
	@Timeout(value = 5, unit = TimeUnit.MINUTES)
	void btReadsInputFileWithFortranDt(@TempDir Path dir) throws Exception {
		Files.write(dir.resolve("inputbt.data"),
				java.util.Arrays.asList("60        number of time steps", "0.01d0    dt", "12 12 12"));
		String out = run("BT", "S", 1, Collections.<String, String>emptyMap(), dir.toString());
		assertTrue(out.contains("Reading from input file inputbt.data"), "does not read inputbt.data:\n" + tail(out));
		assertTrue("SUCCESSFUL".equals(verification(out)), "BT with inputbt.data:\n" + tail(out));
	}

	/**
	 * LU reads the optional file inputlu.data from the working directory (two comment lines and one line of values,
	 * with dt and omega in Fortran form: 0.5d0 and 1.2d0): regression test for the SP and BT failure (Double.valueOf
	 * rejects the d).
	 */
	@Test
	@Timeout(value = 5, unit = TimeUnit.MINUTES)
	void luReadsInputFileWithFortranDt(@TempDir Path dir) throws Exception {
		Files.write(dir.resolve("inputlu.data"), java.util.Arrays.asList("c", "c", "1 50", "c", "c", "50", "c", "c",
				"0.5d0", "c", "c", "1.2d0", "c", "c", "1.0e-08 1.0e-08 1.0e-08 1.0e-08 1.0e-08", "c", "c", "12 12 12"));
		String out = run("LU", "S", 1, Collections.<String, String>emptyMap(), dir.toString());
		assertTrue(out.contains("Reading from input file inputlu.data"), "does not read inputlu.data:\n" + tail(out));
		assertTrue("SUCCESSFUL".equals(verification(out)), "LU with inputlu.data:\n" + tail(out));
	}

	/**
	 * LU requires subdomains of >= 3 points per dimension (S: 12^3 with 32 processes gives an 8 x 4 grid and subdomains
	 * of 1 point in x): the official code aborts with that message; it must neither verify nor hang.
	 */
	@Test
	@Timeout(value = 5, unit = TimeUnit.MINUTES)
	void luSubdomainTooSmallIsRejected() throws Exception {
		String out = run("LU", "S", 32, Collections.<String, String>emptyMap());
		assertTrue(out.contains("SUBDOMAIN SIZE IS TOO SMALL"), "subdomain rejection missing:\n" + tail(out));
		assertTrue(!"SUCCESSFUL".equals(verification(out)), "LU S np=32 should not verify:\n" + tail(out));
	}

	static String tail(String out) {
		String[] l = out.split("\n");
		StringBuilder sb = new StringBuilder();
		for (int i = Math.max(0, l.length - 15); i < l.length; i++)
			sb.append(l[i]).append('\n');
		return sb.toString();
	}
}
