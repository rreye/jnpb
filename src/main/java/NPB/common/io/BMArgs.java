/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB-MPJ by Damián Álvarez Mallón
 */
package NPB.common.io;

public class BMArgs {
	public static char CLASS = 'U';
	/**
	 * Detailed timing. Like the official NPB it can be switched on with the
	 * environment variable NPB_TIMER_FLAG or, if it is not set, with an
	 * existing file "timer.flag" in the working directory; DTIMING=true|false on
	 * the command line overrides both.
	 */
	public static boolean DETAILED_TIMING = timerFlagDefault();
	public static String GRAPH = "BH";

	private static boolean timerFlagDefault() {
		boolean file = false;
		try {
			file = new java.io.File("timer.flag").exists();
		} catch (SecurityException e) {
			// no access to the working directory: treated as no file
		}
		return timerFlag(System.getenv("NPB_TIMER_FLAG"), file);
	}

	/**
	 * Same rules as check_timer_flag of the official NPB: a set variable decides
	 * (empty, starting with 1-9, or on/ON/yes/YES/true/TRUE switch timing on);
	 * if it is not set, the timer.flag file decides.
	 */
	static boolean timerFlag(String env, boolean fileExists) {
		if (env == null)
			return fileExists;
		if (env.isEmpty())
			return true;
		char c = env.charAt(0);
		if (c >= '1' && c <= '9')
			return true;
		return env.equals("on") || env.equals("ON") || env.equals("yes")
				|| env.equals("YES") || env.equals("true") || env.equals("TRUE");
	}

	static boolean isGraphType(String s) {
		return s.equalsIgnoreCase("BH") || s.equalsIgnoreCase("WH")
				|| s.equalsIgnoreCase("SH");
	}

	public static void parseCmdLineArgs(String[] argv, String BMName) {
		for (int i = 0; i < argv.length; i++) {
			if (argv[i].startsWith("class=") || argv[i].startsWith("CLASS=")
					|| argv[i].startsWith("-class")
					|| argv[i].startsWith("-CLASS")) {
				if (argv[i].length() > 6)
					CLASS = Character.toUpperCase(argv[i].charAt(6));
				if (CLASS != 'A' && CLASS != 'B' && CLASS != 'C'
					&& CLASS != 'S' && CLASS != 'W' && CLASS != 'D'
					&& CLASS != 'E') {
					System.err
					.println("Classes allowed are A, B, C, D, E, W and S.");
					commandLineError(BMName);
				}
			} else if (argv[i].startsWith("dtiming=")
					|| argv[i].startsWith("DTIMING=")
					|| argv[i].startsWith("-DTIMING")
					|| argv[i].startsWith("-dtiming")) {
				if (argv[i].length() > 3) {
					DETAILED_TIMING = Boolean.valueOf(argv[i].substring(8));
				}
			} else if (argv[i].startsWith("graph=")
					|| argv[i].startsWith("GRAPH=")
					|| argv[i].startsWith("-graph")
					|| argv[i].startsWith("-GRAPH")) {
				if (argv[i].length() > 6)
					GRAPH = Character.toString(Character.toUpperCase(argv[i]
					                                                      .charAt(6)))
					                                                      + Character.toString(Character.toUpperCase(argv[i]
					                                                                                                      .charAt(7)));
				if (GRAPH.compareTo("BH") != 0 && GRAPH.compareTo("WH") != 0
						&& GRAPH.compareTo("SH") != 0) {
					System.err
					.println("Graphs types are BH, WH and SH. Entered: "
							+ GRAPH);
					commandLineError(BMName);
				}
				} else if (isGraphType(argv[i])) {
				// bare graph name, as in the official DT (dt.B.x BH)
				GRAPH = argv[i].toUpperCase();
			}
		}
	}

	private static void commandLineError(String BMName) {
		if ("DT".equals(BMName)) {
			System.err.println("Usage: mpj_init_command " + BMName
					+ " [GRAPH=]{BH|WH|SH} CLASS={A|B|C|D|E|W|S} [DTIMING={true|false}]");
			System.err.println("{BH|WH|SH} is the type of graph and\n"
					+ "{A|B|C|D|E|W|S} is the size class");
		} else {
			System.err.println("Usage: java " + BMName
					+ " CLASS={A|B|C|D|E|W|S} [DTIMING={true|false}]");
			System.err.println("{A|B|C|D|E|W|S} is the size class");
		}
		System.exit(1);
	}

	public static void outOfMemoryMessage() {
		System.err.println("The java maximum heap size is "
				+ "too small to run this benchmark class");
		System.err.println("To allocate more memory, use the -XmxN option"
				+ " where N is the number of bytes to be allocated");
	}

	public static void bannerMPJ(String BMName, char classSize, int numProcess) {
		System.out.println();
		System.out.println(" Java NAS Parallel Benchmarks (NPB-MPJ) - "+BMName+" Benchmark");
		System.out.println();
	}

	public static void bannerSER(String BMName, char classSize) {
		System.out.println();
		System.out.println(" Java NAS Parallel Benchmarks (NPB-SER) - "+BMName+" Benchmark");
		System.out.println();
	}
}
