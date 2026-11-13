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
 * BT reads the optional file {@code inputbt.data} from the working directory. The official sample
 * ({@code inputbt.data.sample}) writes dt in Fortran form ({@code 0.01d0}), which {@code Double.valueOf} rejects:
 * regression test for that failure. It runs in a separate process with that directory as cwd.
 */
class BtInputFileTest {

	private static String run(Path dir) throws Exception {
		String java = new File(System.getProperty("java.home"), "bin/java").getPath();
		ProcessBuilder pb = new ProcessBuilder(java, "-cp", System.getProperty("java.class.path"), "NPB.SER.BT", "CLASS=S")
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
		Files.write(dir.resolve("inputbt.data"), Arrays.asList("60        number of time steps", "0.01d0    dt", "12 12 12"));
		String out = run(dir);
		assertTrue(out.contains("Reading from input file inputbt.data"), "does not read inputbt.data:\n" + out);
		assertTrue(out.matches("(?s).*Verification\\s*=\\s*SUCCESSFUL.*"), "does not verify with inputbt.data:\n" + out);
	}

	@Test
	@Timeout(value = 6, unit = TimeUnit.MINUTES)
	void usesCompiledDefaultsWithoutInputFile(@TempDir Path dir) throws Exception {
		String out = run(dir);
		assertTrue(out.contains("No input file inputbt.data. Using compiled defaults"), out);
		assertTrue(out.matches("(?s).*Verification\\s*=\\s*SUCCESSFUL.*"), out);
	}
}
