package NPB.MPJ;

import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Class B of {@link MpjGoldenTest}: same checks and tolerances (CG exact, MG 2 ulp, FT 2e-15, EP 1e-15,
 * SP 1e-15) against NPB 3.4.4 MPI with the same number of processes, but with a subset of np because each run
 * takes from tens of seconds to several minutes. Tag {@code classB}: excluded by default (see pom.xml), run
 * with {@code FMPJ_HOME=... mvn test -Pmpj -DexcludedGroups=classC -Dgroups=classB}. The heap of each run is
 * {@code -Dtest.mpj.xmx.big} (default 8g; see {@link MpjLaunchTest#run}). IS B is already in {@link MpjGoldenTest}.
 * The fixtures are generated with {@code scripts/gen_golden_mpi.py --classes B ...}.
 */
@Tag("classB")
class MpjGoldenClassBTest {

	private static Stream<Arguments> cases(int... nps) {
		Stream.Builder<Arguments> b = Stream.builder();
		for (int np : nps)
			b.add(Arguments.of("B", np));
		return b.build();
	}

	static Stream<Arguments> cg() { return cases(1, 4, 16); }
	static Stream<Arguments> mg() { return cases(1, 8); }
	static Stream<Arguments> ep() { return cases(1, 4, 16); }
	static Stream<Arguments> sp() { return cases(1, 4, 16); }
	static Stream<Arguments> bt() { return cases(1, 4, 16); }
	static Stream<Arguments> lu() { return cases(1, 4, 6, 16); }

	static Stream<Arguments> ft() {
		Stream.Builder<Arguments> b = Stream.builder();
		for (int np : new int[] { 1, 4, 16 })
			b.add(Arguments.of("FT", "B", np));
		return b.build();
	}

	@ParameterizedTest(name = "DT class B {0} np={1}")
	@org.junit.jupiter.params.provider.CsvSource({ "BH, 43", "WH, 43", "SH, 192" })
	@Timeout(value = 30, unit = TimeUnit.MINUTES)
	void dt(String graph, int np) throws Exception {
		MpjGoldenTest.checkDt("B", graph, np);
	}

	@ParameterizedTest(name = "CG class {0} np={1}")
	@MethodSource("cg")
	@Timeout(value = 30, unit = TimeUnit.MINUTES)
	void cg(String cls, int np) throws Exception {
		MpjGoldenTest.checkCg(cls, np);
	}

	@ParameterizedTest(name = "MG class {0} np={1}")
	@MethodSource("mg")
	@Timeout(value = 30, unit = TimeUnit.MINUTES)
	void mg(String cls, int np) throws Exception {
		MpjGoldenTest.checkMg(cls, np);
	}

	@ParameterizedTest(name = "{0} class {1} np={2}")
	@MethodSource("ft")
	@Timeout(value = 30, unit = TimeUnit.MINUTES)
	void ft(String kernel, String cls, int np) throws Exception {
		MpjGoldenTest.checkFt(kernel, cls, np);
	}

	@ParameterizedTest(name = "EP class {0} np={1}")
	@MethodSource("ep")
	@Timeout(value = 30, unit = TimeUnit.MINUTES)
	void ep(String cls, int np) throws Exception {
		MpjGoldenTest.checkEp(cls, np);
	}

	@ParameterizedTest(name = "SP class {0} np={1}")
	@MethodSource("sp")
	@Timeout(value = 30, unit = TimeUnit.MINUTES)
	void sp(String cls, int np) throws Exception {
		MpjGoldenTest.checkSp(cls, np);
	}

	@ParameterizedTest(name = "BT class {0} np={1}")
	@MethodSource("bt")
	@Timeout(value = 30, unit = TimeUnit.MINUTES)
	void bt(String cls, int np) throws Exception {
		MpjGoldenTest.checkBt(cls, np);
	}

	@ParameterizedTest(name = "LU class {0} np={1}")
	@MethodSource("lu")
	@Timeout(value = 30, unit = TimeUnit.MINUTES)
	void lu(String cls, int np) throws Exception {
		MpjGoldenTest.checkLu(cls, np);
	}
}
