package NPB.common.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * Command line parsing and the timer flag, which follows check_timer_flag of
 * the official NPB.
 */
class BMArgsTest {

	@Test
	void timerFlagFollowsOfficialRules() {
		// variable not set: the timer.flag file decides
		assertTrue(BMArgs.timerFlag(null, true));
		assertFalse(BMArgs.timerFlag(null, false));
		// variable set: it decides, the file is ignored
		assertTrue(BMArgs.timerFlag("", false));
		for (String on : new String[] { "1", "9", "5abc", "on", "ON", "yes", "YES", "true", "TRUE" })
			assertTrue(BMArgs.timerFlag(on, false), on);
		for (String off : new String[] { "0", "off", "no", "false", "On", "x" })
			assertFalse(BMArgs.timerFlag(off, true), off);
	}

	@Test
	void parsesClassTimingAndGraph() {
		BMArgs.parseCmdLineArgs(new String[] { "CLASS=b", "DTIMING=true", "GRAPH=sh" }, "DT");
		assertEquals('B', BMArgs.CLASS);
		assertTrue(BMArgs.DETAILED_TIMING);
		assertEquals("SH", BMArgs.GRAPH);
		BMArgs.parseCmdLineArgs(new String[] { "class=S", "dtiming=false" }, "CG");
		assertEquals('S', BMArgs.CLASS);
		assertFalse(BMArgs.DETAILED_TIMING);
	}

	@Test
	void bareGraphNameLikeOfficialDt() {
		BMArgs.parseCmdLineArgs(new String[] { "WH", "CLASS=A" }, "DT");
		assertEquals("WH", BMArgs.GRAPH);
		assertEquals('A', BMArgs.CLASS);
		BMArgs.parseCmdLineArgs(new String[] { "bh" }, "DT");
		assertEquals("BH", BMArgs.GRAPH);
	}
}
