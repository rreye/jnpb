package NPB.SER;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

/**
 * LU reads the optional file {@code inputlu.data} from the working directory. A file in the official format
 * ({@code inputlu.data.sample}: two comment lines per line of values) with dt in Fortran form ({@code 0.5d0}), which
 * {@code Double.valueOf} rejects: regression test for that failure. It runs in a separate process with that
 * directory as cwd.
 */
class LuInputFileTest {

	private static String run(Path dir) throws Exception {
		String java = new File(System.getProperty("java.home"), "bin/java").getPath();
		ProcessBuilder pb = new ProcessBuilder(java, "-cp", System.getProperty("java.class.path"), "NPB.SER.LU", "CLASS=S")
				.directory(dir.toFile()).redirectErrorStream(true);
		Process p = pb.start();
		java.io.ByteArrayOutputStream buf = new java.io.ByteArrayOutputStream();
		byte[] chunk = new byte[8192];
		for (int n; (n = p.getInputStream().read(chunk)) > 0;)
			buf.write(chunk, 0, n);
		assertTrue(p.waitFor(5, TimeUnit.MINUTES), "timeout");
		return buf.toString("UTF-8");
	}

	@Test
	@Timeout(value = 6, unit = TimeUnit.MINUTES)
	void readsInputFileWithFortranDt(@TempDir Path dir) throws Exception {
		Files.write(dir.resolve("inputlu.data"), Arrays.asList(
				"c", "c***controls printing of the progress of iterations: ipr    inorm", "1 50",
				"c", "c***the maximum no. of pseudo-time steps to be performed: nitmax", "50",
				"c", "c***magnitude of the time step: dt", "0.5d0",
				"c", "c***relaxation factor for SSOR iterations: omega", "1.2",
				"c", "c***tolerance levels for steady-state residuals", "1.0e-08 1.0e-08 1.0e-08 1.0e-08 1.0e-08",
				"c", "c***number of grid points in xi and eta and zeta directions", "12 12 12"));
		String out = run(dir);
		assertTrue(out.contains("Reading from input file inputlu.data"), "does not read inputlu.data:\n" + out);
		assertTrue(out.matches("(?s).*Verification\\s*=\\s*SUCCESSFUL.*"), "does not verify with inputlu.data:\n" + out);
	}

	@Test
	@Timeout(value = 6, unit = TimeUnit.MINUTES)
	void usesCompiledDefaultsWithoutInputFile(@TempDir Path dir) throws Exception {
		String out = run(dir);
		assertTrue(out.contains("No input file inputlu.data. Using compiled defaults"), out);
		assertTrue(out.matches("(?s).*Verification\\s*=\\s*SUCCESSFUL.*"), out);
	}
}
