/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB-MPJ by Damián Álvarez Mallón
 */
package NPB.common.io;

import java.io.*;

public class BMResults implements Serializable {

	private static final long serialVersionUID = 5871282998562180374L;
	private String name;
	private char classSize;
	private int niter;
	private double mops;
	private long n1, n2, n3;
	private double time[] = { 0, 0 };
	private String opType;
	private int numProcess;
	private int activeProcesses; /* 0 = not reported (as nprocs_active in NPB-MPI c_print_results) */
	private int verified;
	private boolean detailedTiming;

	public BMResults(String bname, char CLASS, long bn1, long bn2, long bn3,
			int bniter, double btime[], double bmops, String boptype,
			int passedVerification, int numProcess, boolean detailedTiming) {

		name = bname;
		classSize = CLASS;
		n1 = bn1;
		n2 = bn2;
		n3 = bn3;
		niter = bniter;
		time[0] = btime[0];
		time[1] = btime[1];
		mops = bmops;
		opType = boptype;
		verified = passedVerification;
		this.numProcess = numProcess;
		this.detailedTiming = detailedTiming;
	}

	/**
	 * Same as above, also reporting the number of ACTIVE processes when it
	 * differs from the total (NPB-MPI 3.4 masks the excess ranks when the number
	 * of processes is not a power of two and NPB_NPROCS_STRICT is off).
	 */
	public BMResults(String bname, char CLASS, long bn1, long bn2, long bn3,
			int bniter, double btime[], double bmops, String boptype,
			int passedVerification, int numProcess, int activeProcesses,
			boolean detailedTiming) {

		this(bname, CLASS, bn1, bn2, bn3, bniter, btime, bmops, boptype,
				passedVerification, numProcess, detailedTiming);
		this.activeProcesses = activeProcesses;
	}

	public void printMPJ()
	{
		System.out.println();
		System.out.println(" "+name+" Benchmark completed.");
		System.out.println(" Class           =\t"+classSize);		

		if (n2 == 0 && n3 == 0)
			System.out.println(" Size            =\t"+n1);
		else
			System.out.println(" Size            =\t"+n1+"x "+n2+"x "+n3);

		System.out.println(" Iterations      =\t"+niter);

		if (!detailedTiming)
			System.out.format(java.util.Locale.ENGLISH," Time in seconds =\t%.2f\n",time[0]);
		else
			System.out.format(java.util.Locale.ENGLISH," Time in seconds =\t%.2f (COMP/COMM: %.2f/%.2f)\n",time[0],time[0]-time[1],time[1]);

		System.out.println(" Total processes =\t"+numProcess);
		if (activeProcesses != 0)
			System.out.println(" Active processes=\t"+activeProcesses);
		System.out.format(java.util.Locale.ENGLISH," Mop/s total     =\t%.2f\n",mops);
		System.out.format(java.util.Locale.ENGLISH," Mop/s/process   =\t%.2f\n",mops/numProcess);
		System.out.println(" Operation type  =\t"+opType);

		if (verified == 1)
			System.out.println(" Verification    =\tSUCCESSFUL");
		else if (verified == 0)
			System.out.println(" Verification    =\tUNSUCCESSFUL");
		else
			System.out.println(" Verification    =\tNOT PERFORMED");

		System.out.println();
	}

	public void printSER()
	{
		System.out.println();
		System.out.println(" "+name+" Benchmark completed.");
		System.out.println(" Class           =\t"+classSize);

		if (n2 == 0 && n3 == 0)
			System.out.println(" Size            =\t"+n1);
		else
			System.out.println(" Size            =\t"+n1+"x "+n2+"x "+n3);

		System.out.println(" Iterations      =\t"+niter);
		System.out.format(java.util.Locale.ENGLISH," Time in seconds =\t%.2f\n",time[0]);
		System.out.format(java.util.Locale.ENGLISH," Mop/s total     =\t%.2f\n",mops);		
		System.out.println(" Operation type  =\t"+opType);

		if (verified == 1)
			System.out.println(" Verification    =\tSUCCESSFUL");
		else if (verified == 0)
			System.out.println(" Verification    =\tUNSUCCESSFUL");
		else
			System.out.println(" Verification    =\tNOT PERFORMED");

		System.out.println();
	}

	public static void printVerificationStatus(char clss, int verified) {
		if (clss == 'U' || verified == -1) {
			verified = -1;
			System.out.println(" Problem size unknown");
			System.out.println(" NO VERIFICATION PERFORMED");
		} else if (verified == 1) {
			System.out.println(" VERIFICATION SUCCESSFUL");
		} else {
			System.out.println(" VERIFICATION FAILED");
		}
	}

	public static int printComparisonStatus(char classSize, int verified,
			double epsilon, double xcr[], double xcrref[], double xcrdif[]) {
		for (int m = 0; m < xcr.length; m++) {
			if (classSize == 'U') {
				System.out.println(m + ". " + xcr[m]);
			} else {
				if (xcrdif[m] <= epsilon) {
					if (verified == -1)
						verified = 1;
				} else {
					verified = 0;
					System.out.print("FAILURE: ");
				}
				System.out.println(m + ". " + xcr[m] + " " + xcrref[m] + " "
						+ xcrdif[m]);
			}
		}
		return verified;
	}

	public static int printComparisonStatus(char classSize, int verified,
			double epsilon, double xcr, double xcrref, double xcrdif) {
		if (classSize == 'U') {
			System.out.println(" " + xcr);
		} else {
			if (xcrdif <= epsilon) {
				if (verified == -1)
					verified = 1;
			} else {
				verified = 0;
				System.out.print("FAILURE: ");
			}
			System.out.println(xcr + " " + xcrref + " " + xcrdif);
		}
		return verified;
	}

	public String getName() {
		return name;
	}

	public char getClassSize() {
		return classSize;
	}

	public long getN1() {
		return n1;
	}

	public long getN2() {
		return n2;
	}

	public long getN3() {
		return n3;
	}

	public int getNiter() {
		return niter;
	}

	public double getMops() {
		return mops;
	}

	public double[] getTime() {
		return time;
	}

	public String getOpType() {
		return opType;
	}

	public int getNumProcess() {
		return numProcess;
	}

	public int getVerified() {
		return verified;
	}

	public boolean getDetailedTiming() {
		return detailedTiming;
	}
}
