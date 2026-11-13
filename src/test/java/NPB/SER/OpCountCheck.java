package NPB.SER;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.Map;

import NPB.common.io.BMResults;

/**
 * Mop/s operation count: recomputes the Mop/s with the OFFICIAL formula (NPB 3.4.4,
 * OMP: cg.f90, ep.f90, ft.f90, is.c, mg.f90) from what {@link BMResults} stores
 * (sizes, niter and the SAME measured time) and compares it with the value
 * reported by Java. It does not depend on the machine: the time is the one that was used.
 *
 * <pre>
 * CG: 1e-6 * 2*niter*na * (3 + nonzer*(nonzer+1) + 25*(5 + nonzer*(nonzer+1)) + 3) / t
 * EP: 2^(m+1) / t / 1e6                      (n1 = 2^(m+1))
 * FT: 1e-6*N * (14.8157 + 7.19641*ln N + (5.23518 + 7.21113*ln N)*niter) / t,  N = nx*ny*nz
 * IS: 1e-6 * TOTAL_KEYS * MAX_ITERATIONS / t
 * MG: 58 * niter * nx*ny*nz * 1e-6 / t
 * SP: 1e-6 * niter * (881.174*N - 4683.91*t^2 + 11484.5*t - 19272.4) / T,  N = nx*ny*nz, t = (nx+ny+nz)/3
 * BT: 1e-6 * niter * (3478.8*N - 17655.7*t^2 + 28023.7*t) / T
 * LU: 1e-6 * itmax * (1984.77*N - 10923.3*t^2 + 27770.9*(nx+ny+nz)/3 - 144010) / T
 * </pre>
 *
 * {@code nonzer} (CG) is not stored in the BMResults: it is taken from the official table.
 */
final class OpCountCheck {

	static final double RTOL = 1e-12;

	private static final Map<Character, Integer> CG_NONZER = new HashMap<>();
	static {
		CG_NONZER.put('S', 7);
		CG_NONZER.put('W', 8);
		CG_NONZER.put('A', 11);
		CG_NONZER.put('B', 13);
		CG_NONZER.put('C', 15);
	}

	private OpCountCheck() {
	}

	static void check(String kernel, char cls, BMResults r) {
		double t = r.getTime()[0];
		assertTrue(t > 0, kernel + " " + cls + ": non-positive time (" + t + ")");
		double n1 = r.getN1(), n2 = r.getN2(), n3 = r.getN3(), niter = r.getNiter();
		double expected;
		switch (kernel) {
		case "CG": {
			double nz = CG_NONZER.get(cls);
			expected = 1.0e-6 * 2 * niter * n1 * (3. + nz * (nz + 1) + 25. * (5. + nz * (nz + 1)) + 3.) / t;
			break;
		}
		case "EP":
			expected = n1 / t / 1.0e6;
			break;
		case "FT": {
			double n = n1 * n2 * n3;
			expected = 1.0e-6 * n * (14.8157 + 7.19641 * Math.log(n) + (5.23518 + 7.21113 * Math.log(n)) * niter) / t;
			break;
		}
		case "IS":
			expected = 1.0e-6 * n1 * niter / t;
			break;
		case "MG":
			expected = 58. * niter * (n1 * n2 * n3) * 1.0e-6 / t;
			break;
		case "SP": {
			double n = n1 * n2 * n3, tt = (n1 + n2 + n3) / 3.0;
			expected = 1.0e-6 * niter * (881.174 * n - 4683.91 * (tt * tt) + 11484.5 * tt - 19272.4) / t;
			break;
		}
		case "BT": {
			double n = n1 * n2 * n3, tt = (n1 + n2 + n3) / 3.0;
			expected = 1.0e-6 * niter * (3478.8 * n - 17655.7 * (tt * tt) + 28023.7 * tt) / t;
			break;
		}
		case "LU": {
			double s3 = (n1 + n2 + n3) / 3.0;
			expected = 1.0e-6 * niter * (1984.77 * n1 * n2 * n3 - 10923.3 * (s3 * s3) + 27770.9 * (n1 + n2 + n3) / 3.0
					- 144010.0) / t;
			break;
		}
		default:
			throw new IllegalArgumentException(kernel);
		}
		double rel = Math.abs(r.getMops() - expected) / expected;
		System.err.printf("OPCOUNT %s class %s relDiff=%.3e%n", kernel, cls, rel);
		assertTrue(rel <= RTOL, kernel + " class " + cls + ": official Mop/s=" + expected + " Java=" + r.getMops()
				+ " relDiff=" + rel); // `!(rel <= tol)` also fails on NaN
	}
}
