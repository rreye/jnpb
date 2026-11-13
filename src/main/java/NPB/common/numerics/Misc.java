/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB-MPJ by Damián Álvarez Mallón
 */
package NPB.common.numerics;

public class Misc {

	public static int ipow2(int i) {
		int pow2 = 1;
		if (i < 0)
			return -1;
		if (i == 0)
			return 1;
		while (i-- > 0) {
			pow2 *= 2;
		}
		return pow2;
	}

	public static int ilog2(int i) {
		int log2;
		int exp2 = 1;
		if (i <= 0)
			return (-1);

		for (log2 = 0; log2 < 20; log2++) {
			if (exp2 == i)
				return (log2);
			exp2 *= 2;
		}
		return (-1);
	}

	public static int isqrt(int i) {
		int root;
		int square;

		if (i <= 0)
			return (-1);
		square = 0;
		for (root = 1; square <= i; root++) {
			square = root * root;
			if (square == i)
				return root;
		}
		return -1;
	}
}
