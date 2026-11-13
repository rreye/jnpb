package NPB.MPJ;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * {@link ActiveProcs} without MPI: topology (get_active_nprocs of NPB3.4-MPI) and parsing of
 * NPB_NPROCS_STRICT. Only compiled with the {@code mpj} profile (ActiveProcs uses mpi.*).
 */
class ActiveProcsTest {

	@Test
	void powerOfTwoTopologyIsTheLargestPowerOfTwoNotAboveNprocs() {
		int[][] expected = { { 1, 1, 1 }, { 2, 1, 2 }, { 2, 1, 2 }, { 2, 2, 4 }, { 2, 2, 4 }, { 2, 2, 4 }, { 2, 2, 4 },
				{ 4, 2, 8 }, { 4, 2, 8 }, { 4, 2, 8 } };
		for (int n = 1; n <= 10; n++)
			assertArrayEquals(expected[n - 1], ActiveProcs.topology(ActiveProcs.POW2, n), "POW2 np=" + n);
		assertArrayEquals(new int[] { 8, 4, 32 }, ActiveProcs.topology(ActiveProcs.POW2, 63));
		assertArrayEquals(new int[] { 8, 8, 64 }, ActiveProcs.topology(ActiveProcs.POW2, 64));
		assertArrayEquals(new int[] { 32, 16, 512 }, ActiveProcs.topology(ActiveProcs.POW2, 1000));
		// np1 >= np2 and np1 * np2 is always a power of 2 <= np
		for (int n = 1; n <= 4096; n++) {
			int[] t = ActiveProcs.topology(ActiveProcs.POW2, n);
			assertTrue(t[0] >= t[1] && t[2] == t[0] * t[1] && t[2] <= n && 2 * t[2] > n, "np=" + n);
			assertEquals(1, Integer.bitCount(t[2]), "np=" + n);
		}
	}

	/**
	 * LU grid topology (pkind 2): {np1, np2, npa} from get_active_nprocs of NPB 3.4.4 MPI for 1..40 processes,
	 * obtained by running the official source (common/get_active_nprocs.f90 with mpif90 and mpirun --oversubscribe,
	 * NPB_NPROCS_STRICT=off), not with this class.
	 */
	@Test
	void gridTopologyMatchesTheOfficialGetActiveNprocs() {
		int[][] official = {
				{ 1, 1, 1 }, { 2, 1, 2 }, { 2, 1, 2 }, { 2, 2, 4 }, { 2, 2, 4 },
				{ 3, 2, 6 }, { 3, 2, 6 }, { 4, 2, 8 }, { 3, 3, 9 }, { 3, 3, 9 },
				{ 3, 3, 9 }, { 4, 3, 12 }, { 4, 3, 12 }, { 4, 3, 12 }, { 5, 3, 15 },
				{ 4, 4, 16 }, { 4, 4, 16 }, { 6, 3, 18 }, { 6, 3, 18 }, { 5, 4, 20 },
				{ 5, 4, 20 }, { 5, 4, 20 }, { 5, 4, 20 }, { 6, 4, 24 }, { 5, 5, 25 },
				{ 5, 5, 25 }, { 5, 5, 25 }, { 7, 4, 28 }, { 7, 4, 28 }, { 6, 5, 30 },
				{ 6, 5, 30 }, { 8, 4, 32 }, { 8, 4, 32 }, { 8, 4, 32 }, { 7, 5, 35 },
				{ 6, 6, 36 }, { 6, 6, 36 }, { 6, 6, 36 }, { 6, 6, 36 }, { 8, 5, 40 }
		};
		for (int n = 1; n <= official.length; n++)
			assertArrayEquals(official[n - 1], ActiveProcs.topology(ActiveProcs.GRID, n), "GRID np=" + n);
		// np2 <= np1 <= 2*np2, product = npa <= np and more than half of the processes active
		for (int n = 1; n <= 4096; n++) {
			int[] t = ActiveProcs.topology(ActiveProcs.GRID, n);
			assertTrue(t[1] <= t[0] && t[0] <= 2 * t[1] && t[2] == t[0] * t[1] && t[2] <= n && 2 * t[2] > n, "np=" + n);
		}
	}

	@Test
	void squareTopologyIsTheLargestSquareNotAboveNprocs() {
		for (int n = 1; n <= 4096; n++) {
			int[] t = ActiveProcs.topology(ActiveProcs.SQUARE, n);
			int r = (int) Math.floor(Math.sqrt(n));
			assertEquals(r, t[0], "np=" + n);
			assertEquals(r, t[1], "np=" + n);
			assertEquals(r * r, t[2], "np=" + n);
		}
	}

	@Test
	void strictEnvironmentValuesFollowTheOfficialParsing() {
		for (String off : new String[] { "0", "-", "-1", "off", "OFF", "n", "N", "no", "No", "NO", "f", "F", "false", "False" })
			assertFalse(ActiveProcs.isStrict(off), "'" + off + "' must disable strict checking");
		for (String on : new String[] { null, "", "1", "yes", "Yes", "on", "true", "abc", "Off", "oFF", "00" })
			assertTrue(ActiveProcs.isStrict(on), "'" + on + "' must keep strict checking");
	}

	@Test
	void errorMessageNamesTheExpectedKind() {
		assertTrue(ActiveProcs.errorMessage(ActiveProcs.POW2, 6, 4).contains("power-of-two"));
		assertTrue(ActiveProcs.errorMessage(ActiveProcs.SQUARE, 5, 4).contains("square"));
		assertTrue(ActiveProcs.errorMessage(ActiveProcs.POW2, 6, 4).contains("(such as 4)"));
		assertTrue(ActiveProcs.errorMessage(ActiveProcs.GRID, 5, 4).contains("grid"));
	}
}
