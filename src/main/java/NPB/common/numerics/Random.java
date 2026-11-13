/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB-MPJ by Damián Álvarez Mallón
 */
package NPB.common.numerics;

public class Random {

	/* default seed */
	private double tran = 314159265.0; // First 9 digits of PI

	/* constants */
	private static final double d2m46 = Math.pow(0.5, 46);
	private static final long i246m1 = 70368744177663L; //(long) Math.pow(2, 46) - 1;
	private static final double r23 = Math.pow(0.5, 23);
	public static final double r46 = Math.pow(r23, 2);
	private static final double t23 = Math.pow(2.0, 23);
	private static final double t46 = Math.pow(t23, 2);

	/*
	 * Random number generator with an external seed
	 */
	public double randlc(double x, double a) {
		double t1, t2, t3, t4, a1, a2, x1, x2, z;

		/*
		 * Break A into two parts such that A = 2^23 * A1 + A2.
		 */
		t1 = r23 * a;
		a1 = (int) t1;
		a2 = a - t23 * a1;
		/*
		 * Break X into two parts such that X = 2^23 * X1 + X2, compute Z = A1 *
		 * X2 + A2 * X1 (mod 2^23), and then X = 2^23 * Z + A2 * X2 (mod 2^46).
		 */
		t1 = r23 * x;
		x1 = (int) t1;
		x2 = x - t23 * x1;
		t1 = a1 * x2 + a2 * x1;
		t2 = (int) (r23 * t1);
		z = t1 - t23 * t2;
		t3 = t23 * z + a2 * x2;
		t4 = (int) (r46 * t3);
		x = t3 - t46 * t4;
		return x;
	}

	/*
	 * Random number generator with an external seed that return the seed and
	 * the random number
	 */
	public double[] vrandlc(double x, double a) {
		double t1, t2, t3, t4, a1, a2, x1, x2, z;
		/*
		 * Break A into two parts such that A = 2^23 * A1 + A2.
		 */
		t1 = r23 * a;
		a1 = (int) t1;
		a2 = a - t23 * a1;
		/*
		 * Break X into two parts such that X = 2^23 * X1 + X2, compute Z = A1 *
		 * X2 + A2 * X1 (mod 2^23), and then X = 2^23 * Z + A2 * X2 (mod 2^46).
		 */
		t1 = r23 * x;
		x1 = (int) t1;
		x2 = x - t23 * x1;
		t1 = a1 * x2 + a2 * x1;
		t2 = (int) (r23 * t1);
		z = t1 - t23 * t2;
		t3 = t23 * z + a2 * x2;
		t4 = (int) (r46 * t3);
		x = t3 - t46 * t4;
		double values[] = { x, r46 * x };
		return values;
	}

	/*
	 * Random number generator with an internal seed
	 */
	public double randlc(double a) {
		double t1, t2, t3, t4, a1, a2, x1, x2, z;

		/*
		 * Break A into two parts such that A = 2^23 * A1 + A2.
		 */
		t1 = r23 * a;
		a1 = (int) t1;
		a2 = a - t23 * a1;
		/*
		 * Break X into two parts such that X = 2^23 * X1 + X2, compute Z = A1 *
		 * X2 + A2 * X1 (mod 2^23), and then X = 2^23 * Z + A2 * X2 (mod 2^46).
		 */
		t1 = r23 * tran;
		x1 = (int) t1;
		x2 = tran - t23 * x1;
		t1 = a1 * x2 + a2 * x1;
		t2 = (int) (r23 * t1);
		z = t1 - t23 * t2;
		t3 = t23 * z + a2 * x2;
		t4 = (int) (r46 * t3);
		tran = t3 - t46 * t4;
		return (r46 * tran);
	}

	public double vranlc(double n, double x, double a, double y[], int offset) {
		long Lx = (long) x;
		long La = (long) a;

		for (int i = 0; i < n; i++) {
			Lx = (Lx * La) & (i246m1);
			y[offset + i] = (double) (d2m46 * Lx);
		}
		return (double) Lx;
	}

	public double complexvranlc(double n, double x, double a, double y[][], int offset) {
		long Lx = (long) x;
		long La = (long) a;

		for (int i = 0; i < n; i++) {
			Lx = (Lx * La) & (i246m1);
			y[0][offset + i] = (double) (d2m46 * Lx);
			Lx = (Lx * La) & (i246m1);
			y[1][offset + i] = (double) (d2m46 * Lx);
		}
		return (double) Lx;
	}

	public double complexvranlc(double n, double x, double a, double y[], int offset) {
		long Lx = (long) x;
		long La = (long) a;

		for (int i = 0; i < n; i++) {
			Lx = (Lx * La) & (i246m1);
			y[2 * (offset + i)] = (double) (d2m46 * Lx);
			Lx = (Lx * La) & (i246m1);
			y[1 + 2 * (offset + i)] = (double) (d2m46 * Lx);
		}
		return (double) Lx;
	}

	/*
	 * Power raises an integer, disguised as a double precision real, to an
	 * integer power
	 */
	public double power(double a, int n) {

		double aj, ajj, pow, seed;
		int nj;

		pow = 1.0;
		nj = n;
		aj = a;
		while (nj != 0) {
			if (nj % 2 == 1) {
				seed = randlc(pow, aj);
				pow = seed;
			}
			ajj = aj;
			seed = randlc(aj, ajj);
			aj = seed;
			nj = nj / 2;
		}
		return pow;
	}

	/*
	 * Power raises an integer, disguised as a double precision real, to an
	 * integer power. This version tries to avoid integer overflow by treating
	 * it as expressed in a form of "n1*n2+n3".
	 */
	public double power(double a, int n1, int n2, int n3) {

		double aj, ajj, pow, seed;
		int n1j, n2j, nj;

		pow = 1.0;
		aj = a;
		nj = n3;
		n1j = n1;
		n2j = n2;

		while (true) {
			if (n2j > 0){			      
				if (n2j % 2 == 1)
					nj = nj + n1j;

				n2j = n2j / 2;
			}
			else if(nj == 0)
				break;

			if (nj % 2 == 1) {
				seed = randlc(pow, aj);
				pow = seed;
			}
			ajj = aj;
			seed = randlc(aj, ajj);
			aj = seed;
			nj = nj / 2;
		}
		return pow;
	}
}
