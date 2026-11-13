/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB 3.4.4 MPI (common/get_active_nprocs.f90)
 */
package NPB.MPJ;

import mpi.Intracomm;
import mpi.MPI;

import NPB.exceptions.IncorrectNumProcessException;

final class ActiveProcs {

	/** Topology kinds, with the numbering of get_active_nprocs. */
	static final int SQUARE = 1;
	static final int GRID = 2;
	static final int POW2 = 3;

	/** Processes launched (np_total in NPB-MPI). */
	final int total;
	/** Rank of this process in MPI.COMM_WORLD. */
	final int rank;
	/** Processes that take part (a square or a power of two). */
	final int active;
	/** Factors of {@link #active}: np1 >= np2 (as the arguments np1, np2 of get_active_nprocs). */
	final int np1, np2;
	/** False for the excess processes masked out. */
	final boolean isActive;
	/** Communicator of the active processes (COMM_WORLD when none is masked out). */
	final Intracomm comm;

	private ActiveProcs(int total, int rank, int[] topo, boolean isActive, Intracomm comm) {
		this.total = total;
		this.rank = rank;
		this.np1 = topo[0];
		this.np2 = topo[1];
		this.active = topo[2];
		this.isActive = isActive;
		this.comm = comm;
	}

	/** {np1, np2, npa} for a topology kind and a number of processes (no communication). */
	static int[] topology(int pkind, int nprocs) {
		int np1, np2;
		if (pkind <= SQUARE) {
			np2 = (int) Math.sqrt(nprocs + 1.0e-3);
			np1 = np2;
		} else {
			np1 = (int) (Math.log(nprocs + 1.0e-3) / Math.log(2.0));
			np2 = np1 / 2;
			np1 = np1 - np2;
			np1 = 1 << np1;
			np2 = 1 << np2;
		}
		int npa = np1 * np2;

		if (pkind == GRID && npa < nprocs) {
			int np1w = (int) Math.sqrt(nprocs + 1.0e-3);
			int np2w = (int) Math.sqrt(nprocs * 2 + 1.0e-3);
			for (int n = np1w; n <= np2w; n++) {
				int npaw = nprocs / n * n;
				if (n == 1 && nprocs == 3)
					npaw = 2;
				if (npaw > npa) {
					npa = npaw;
					np1 = npa / n;
					if (np1 < n) {
						np2 = np1;
						np1 = n;
					} else {
						np2 = n;
					}
				}
			}
		}
		return new int[] { np1, np2, npa };
	}

	/**
	 * Whether the process count is enforced given the value of NPB_NPROCS_STRICT.
	 * Enforced by default; (0, off, OFF, and anything starting with n, N, f, F or -)
	 * turns it off.
	 */
	static boolean isStrict(String value) {
		if (value == null || value.length() == 0)
			return true;
		if (value.equals("0") || value.charAt(0) == '-')
			return false;
		if (value.equals("off") || value.equals("OFF") || "nNfF".indexOf(value.charAt(0)) >= 0)
			return false;
		return true;
	}

	/** Message of the official error for an unsupported number of processes. */
	static String errorMessage(int pkind, int nprocs, int npa) {
		return "Error determining processor topology for " + nprocs + " processes: expecting a "
				+ (pkind <= SQUARE ? "square" : pkind == GRID ? "grid (nx*ny, nx/2<=ny<=nx)" : "power-of-two")
				+ " number of processes (such as " + npa + ")";
	}

	/**
	 * Selects the active processes. Collective over MPI.COMM_WORLD (only when the
	 * number of processes is not the supported one). Throws
	 * IncorrectNumProcessException on every process if the count is enforced; the
	 * caller must finalize MPI.
	 */
	static ActiveProcs select(int pkind) throws IncorrectNumProcessException {
		Intracomm world = MPI.COMM_WORLD;
		int total = world.Size();
		int rank = world.Rank();
		int[] topo = topology(pkind, total);

		if (total == topo[2])
			return new ActiveProcs(total, rank, topo, true, world);

		int strict[] = { 1 };
		if (rank == 0)
			strict[0] = isStrict(System.getenv("NPB_NPROCS_STRICT")) ? 1 : 0;
		world.Bcast(strict, 0, 1, MPI.INT, 0);

		if (strict[0] != 0)
			throw new IncorrectNumProcessException(errorMessage(pkind, total, topo[2]));

		boolean isActive = rank < topo[2];
		Intracomm comm = world.Split(isActive ? 0 : 1, rank);
		return new ActiveProcs(total, rank, topo, isActive, comm);
	}
}
