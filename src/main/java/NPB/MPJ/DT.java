/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB-MPJ by Damián Álvarez Mallón
 */
package NPB.MPJ;

import mpi.MPI;
import NPB.Benchmark;

import NPB.common.io.BMArgs;
import NPB.common.io.BMResults;
import NPB.common.numerics.Misc;
import NPB.common.numerics.Random;
import NPB.common.timer.Timer;
import NPB.exceptions.BadClassException;
import NPB.exceptions.IncorrectNumProcessException;
import NPB.exceptions.NPBException;

public class DT implements Benchmark{

	private static final String BM_NAME = "DT";
	private char CLASS;
	private boolean detailedTiming;
	private boolean nodeTiming = false;
	private String graph;
	private Timer timer;
	private Timer nodeTimer;
	private int verified;

	private int NUM_SAMPLES = 0;
	private int STD_DEVIATION = 0;
	private int NUM_SOURCES = 0;

	protected static int BLOCK_SIZE = 128;
	protected static int SMALL_BLOCK_SIZE = 32;

	private final int fieldDim = 4;

	private Random rng;
	private BMResults res;

	/*
	 * Different timers
	 */
	private final int T_COMP = 0;
	private final int T_COMM = 1;
	
	/*
	 * MPJ variables
	 */
	private int myRank;
	private int numProcess;

	public static void main(String[] args) throws NPBException {
		DT dt = null;

		BMArgs.parseCmdLineArgs(args, BM_NAME);
		try {
			dt = new DT(BMArgs.CLASS, BMArgs.DETAILED_TIMING, BMArgs.GRAPH);
		} catch (OutOfMemoryError e) {
			BMArgs.outOfMemoryMessage();
			throw e;
		}
		dt.runBenchmark(args);
	}

	public BMResults getResults(){
		return res;
	}
	
	public DT(char classSize, boolean dtiming, String graph) {
		detailedTiming = dtiming;
		CLASS = classSize;
		this.graph = graph;

	}
	
	public void build(String[] args) throws NPBException {
		
		MPI.Init(args);
		
		myRank = MPI.COMM_WORLD.Rank();
		numProcess = MPI.COMM_WORLD.Size();

		rng = new Random();

		switch (CLASS) {
		case 'S':
			NUM_SAMPLES = 1728;
			STD_DEVIATION = 128;
			NUM_SOURCES = 4;
			break;
		case 'W':
			NUM_SAMPLES = 13824;
			STD_DEVIATION = 256;
			NUM_SOURCES = 8;
			break;
		case 'A':
			NUM_SAMPLES = 110592;
			STD_DEVIATION = 512;
			NUM_SOURCES = 16;
			break;
		case 'B':
			NUM_SAMPLES = 884736;
			STD_DEVIATION = 1024;
			NUM_SOURCES = 32;
			break;
		case 'C':
			NUM_SAMPLES = 7077888;
			STD_DEVIATION = 2048;
			NUM_SOURCES = 64;
			break;
		case 'D':
			NUM_SAMPLES = 56623104;
			STD_DEVIATION = 4096;
			NUM_SOURCES = 128;
			break;
		default:
			terminate();
			throw new BadClassException("Especified class (" + CLASS
					+ ") not supported by DT. Valid classes: S, W, A, B, C, D");
		}

	}

	public void runBenchmark(String[] args) throws NPBException {
		
		this.build(args);
		DGraph dg = null;
		int featNum;
		double bytesSent = 2.0;

		switch (graph.charAt(0)) {
		case 'B':
			dg = buildBH(CLASS);
			break;
		case 'W':
			dg = buildWH(CLASS);
			break;
		case 'S':
			dg = buildSH(CLASS);
			break;
		}
		if (nodeTiming && dg.numNodes + 1 > Timer.MAX_TIMERS) {
			nodeTiming = false;
			if (myRank == 0)
				System.err.println("Not enough timers. Node timeing is off.");
		}

		nodeTimer = new Timer();
		timer = new Timer();
		timer.resetAll();
		nodeTimer.resetAll();
		if (dg.numNodes > numProcess) {
			MPI.Finalize();
			throw new IncorrectNumProcessException("The number of MPJ processes should not be less than the number of nodes in the graph\n"+
					"Number nodes in the graph = "+dg.numNodes);
		}
		for (int i = 0; i < dg.numNodes; i++) {
			dg.node[i].address = i;
		}
		if (myRank == 0) {
			System.out
					.printf("\n\n Unofficial NAS Parallel Benchmarks -- DT Benchmark\n\n");
			graphShow(dg, 0);
			timer.start(T_COMP);
		}
		processNodes(dg);

		featNum = NUM_SAMPLES * fieldDim;
		bytesSent = featNum * dg.numArcs;
		bytesSent /= 1048576;

		if (myRank == 0) {
			timer.stop(T_COMP);
			double times[] = { 0, 0 };
			times[0] = timer.read(T_COMP);
			times[1] = timer.read(T_COMM);

			BMResults.printVerificationStatus(CLASS, verified);
			res = new BMResults(dg.name, CLASS, featNum, 0, 0,
					dg.numNodes, times, bytesSent / times[0],
					"Bytes transmitted", verified, numProcess, detailedTiming);
			res.printMPJ();
		}

		terminate();

	}

	/*
	 * Nodes of the graph must be topologically sorted to avoid MPI
	 * deadlock.
	 */
	DGraph buildBH(char cls) {
		int numSources = NUM_SOURCES, maxInDeg = 4;
		int numLayerNodes = numSources, firstLayerNode = 0;
		DGraph dg;
		DGNode nd = null, snd = null, sink = null;
		DGArc ar = null;
		int toT_COMParators = 0;
		int numPrevLayerNodes = numLayerNodes;
		int id = 0, sid = 0;

		dg = new DGraph("DT_BH." + cls);
		int k = 0;
		for (k = 0; k < numSources; k++) {
			nd = new DGNode("Source." + k);
			AttachNode(dg, nd);
		}
		while (numLayerNodes > maxInDeg) {
			numLayerNodes = numLayerNodes / maxInDeg;
			if (numLayerNodes * maxInDeg < numPrevLayerNodes)
				numLayerNodes++;
			for (int i = 0; i < numLayerNodes; i++) {
				nd = new DGNode("Comparator." + toT_COMParators);
				toT_COMParators++;
				id = AttachNode(dg, nd);
				for (int j = 0; j < maxInDeg; j++) {
					sid = i * maxInDeg + j;
					if (sid >= numPrevLayerNodes)
						break;
					snd = dg.node[firstLayerNode + sid];
					ar = new DGArc(snd, dg.node[id]);
					AttachArc(dg, ar);
				}
			}
			firstLayerNode += numPrevLayerNodes;
			numPrevLayerNodes = numLayerNodes;
		}
		sink = new DGNode("Sink");
		AttachNode(dg, sink);
		for (int i = 0; i < numPrevLayerNodes; i++) {
			nd = dg.node[firstLayerNode + i];
			ar = new DGArc(nd, sink);
			AttachArc(dg, ar);
		}
		return dg;

	}

	/*
	 * Nodes of the graph must be topologically sorted to avoid MPI
	 * deadlock.
	 */
	DGraph buildWH(char cls) {
		int numSources = NUM_SOURCES, maxInDeg = 4;
		int numLayerNodes = numSources, firstLayerNode = 0;
		int toT_COMParators = 0;
		int numPrevLayerNodes = numLayerNodes;
		int id = 0, sid = 0;
		DGraph dg;
		DGNode nd = null, source = null, tmp = null, snd = null;
		DGArc ar = null;

		dg = new DGraph("DT_WH." + cls);

		for (int i = 0; i < numSources; i++) {
			nd = new DGNode("Sink." + i);
			AttachNode(dg, nd);
		}
		toT_COMParators = 0;
		numPrevLayerNodes = numLayerNodes;
		while (numLayerNodes > maxInDeg) {
			numLayerNodes = numLayerNodes / maxInDeg;
			if (numLayerNodes * maxInDeg < numPrevLayerNodes)
				numLayerNodes++;
			for (int i = 0; i < numLayerNodes; i++) {
				nd = new DGNode("Comparator." + toT_COMParators);
				toT_COMParators++;
				id = AttachNode(dg, nd);
				for (int j = 0; j < maxInDeg; j++) {
					sid = i * maxInDeg + j;
					if (sid >= numPrevLayerNodes)
						break;
					snd = dg.node[firstLayerNode + sid];
					ar = new DGArc(dg.node[id], snd);
					AttachArc(dg, ar);
				}
			}
			firstLayerNode += numPrevLayerNodes;
			numPrevLayerNodes = numLayerNodes;
		}
		source = new DGNode("Source");
		AttachNode(dg, source);
		for (int i = 0; i < numPrevLayerNodes; i++) {
			nd = dg.node[firstLayerNode + i];
			ar = new DGArc(source, nd);
			AttachArc(dg, ar);
		}

		for (int i = 0; i < dg.numNodes / 2; i++) { /* Topological sorting */
			tmp = dg.node[i];
			dg.node[i] = dg.node[dg.numNodes - 1 - i];
			dg.node[i].id = i;
			dg.node[dg.numNodes - 1 - i] = tmp;
			dg.node[dg.numNodes - 1 - i].id = dg.numNodes - 1 - i;
		}
		return dg;

	}

	/*
	 * Nodes of the graph must be topologically sorted to avoid MPI
	 * deadlock.
	 */
	DGraph buildSH(char cls) {
		DGraph dg;
		int numSources = NUM_SOURCES; /* must be power of 2 */
		int numOfLayers = 0, tmpS = numSources >> 1;
		int firstLayerNode = 0;
		DGArc ar = null;
		DGNode nd = null;
		int mask = 0x0, ndid = 0, ndoff = 0;

		dg = new DGraph("DT_SH." + cls);

		while (tmpS > 1) {
			numOfLayers++;
			tmpS >>= 1;
		}
		for (int i = 0; i < numSources; i++) {
			nd = new DGNode("Source." + i);
			AttachNode(dg, nd);
		}
		for (int j = 0; j < numOfLayers; j++) {
			mask = 0x00000001 << j;
			for (int i = 0; i < numSources; i++) {
				nd = new DGNode("Comparator." + (i + j * firstLayerNode));
				AttachNode(dg, nd);
				ndoff = i & (~mask);
				ndid = firstLayerNode + ndoff;
				ar = new DGArc(dg.node[ndid], nd);
				AttachArc(dg, ar);
				ndoff += mask;
				ndid = firstLayerNode + ndoff;
				ar = new DGArc(dg.node[ndid], nd);
				AttachArc(dg, ar);
			}
			firstLayerNode += numSources;
		}
		mask = 0x00000001 << numOfLayers;
		for (int i = 0; i < numSources; i++) {
			nd = new DGNode("Sink." + i);
			AttachNode(dg, nd);
			ndoff = i & (~mask);
			ndid = firstLayerNode + ndoff;
			ar = new DGArc(dg.node[ndid], nd);
			AttachArc(dg, ar);
			ndoff += mask;
			ndid = firstLayerNode + ndoff;
			ar = new DGArc(dg.node[ndid], nd);
			AttachArc(dg, ar);
		}
		return dg;
	}

	void graphShow(DGraph dg, int detailsLevel) {
		System.err.printf("%d.%s: (%d,%d)\n", dg.id, dg.name, dg.numNodes,
				dg.numArcs);
		if (detailsLevel < 1)
			return;
		for (int i = 0; i < dg.numNodes; i++) {
			DGNode focusNode = dg.node[i];
			if (detailsLevel >= 2) {
				for (int j = 0; j < focusNode.inDegree; j++) {
					System.err.printf("\t ");
					nodeShow(focusNode.inArc[j].tail);
				}
			}
			nodeShow(focusNode);
			if (detailsLevel < 2)
				continue;
			for (int j = 0; j < focusNode.outDegree; j++) {
				System.err.printf("\t ");
				nodeShow(focusNode.outArc[j].head);
			}
			System.err.printf("---\n");
		}
		System.err.printf("----------------------------------------\n");
		if (detailsLevel < 3)
			return;
	}

	void processNodes(DGraph dg) {
		double chksum = 0.0;
		Arr feat = null;
		int tag;
		DGNode nd = null;
		double rchksum = 0.0;

		for (int i = 0; i < dg.numNodes; i++) {
			nd = dg.node[i];
			if (nd.address != myRank)
				continue;
			if (nd.name.indexOf("Source") >= 0) {
				nd.feat = randomFeatures(dg.name, fieldDim, nd.id);
				sendResults(dg, nd, nd.feat);
			} else if (nd.name.indexOf("Sink") >= 0) {
				chksum = reduceStreams(dg, nd);
				tag = dg.numArcs + nd.id; /*
											 * make these to avoid clash with
											 * arc tags
											 */
				double sendbuff[] = { chksum };
				if (detailedTiming)
					timer.start(T_COMM);
				MPI.COMM_WORLD.Send(sendbuff, 0, 1, MPI.DOUBLE, 0, tag);
				if (detailedTiming)
					timer.stop(T_COMM);
			} else {
				feat = combineStreams(dg, nd);
				sendResults(dg, nd, feat);
			}
		}
		if (myRank == 0) { /* Report node */
			rchksum = 0.0;
			chksum = 0.0;
			for (int i = 0; i < dg.numNodes; i++) {
				nd = dg.node[i];
				if (!(nd.name.indexOf("Sink") >= 0))
					continue;
				tag = dg.numArcs + nd.id; /*
											 * make these to avoid clash with
											 * arc tags
											 */
				double recvbuff[] = { 0 };
				if (detailedTiming)
					timer.start(T_COMM);
				MPI.COMM_WORLD
						.Recv(recvbuff, 0, 1, MPI.DOUBLE, nd.address, tag);
				if (detailedTiming)
					timer.stop(T_COMM);
				rchksum = recvbuff[0];
				chksum += rchksum;
			}
			verified = verify(dg.name, chksum);
		}
	}

	void nodeShow(DGNode nd) {
		System.err.printf("%3d.%s: (%d,%d)\n", nd.id, nd.name, nd.inDegree,
				nd.outDegree);
	}

	private void terminate() {
		MPI.Finalize();
	}

	int AttachNode(DGraph dg, DGNode nd) {
		int len = 0;
		DGNode nds[] = null, tmpnd = null;
		DGArc ar[] = null;

		if (dg.numNodes == dg.maxNodes - 1) {
			dg.maxNodes += BLOCK_SIZE;
			nds = new DGNode[dg.maxNodes];
			System.arraycopy(dg.node, 0, nds, 0, dg.maxNodes - BLOCK_SIZE);
			dg.node = nds;
		}

		len = nd.name.length();
		for (int i = 0; i < dg.numNodes; i++) {
			tmpnd = dg.node[i];
			ar = null;
			if (tmpnd.name.length() != len)
				continue;
			if (strncmp(nd.name, tmpnd.name, len))
				continue;
			if (nd.inDegree > 0) {
				tmpnd.maxInDegree += nd.maxInDegree;
				ar = new DGArc[tmpnd.maxInDegree];
				System.arraycopy(tmpnd.inArc, 0, ar, 0, tmpnd.inDegree);
				tmpnd.inArc = ar;
				for (int j = 0; j < nd.inDegree; j++) {
					nd.inArc[j].head = tmpnd;
				}
				System.arraycopy(nd.inArc, 0, tmpnd.inArc, tmpnd.inDegree,
						nd.inDegree);
				tmpnd.inDegree += nd.inDegree;
			}
			if (nd.outDegree > 0) {
				tmpnd.maxOutDegree += nd.maxOutDegree;
				ar = new DGArc[tmpnd.maxOutDegree];
				System.arraycopy(tmpnd.outArc, 0, ar, 0, tmpnd.outDegree);
				tmpnd.outArc = ar;
				for (int j = 0; j < nd.outDegree; j++) {
					nd.outArc[j].tail = tmpnd;
				}
				System.arraycopy(nd.outArc, 0, tmpnd.outArc, tmpnd.outDegree,
						nd.outDegree);
				tmpnd.outDegree += nd.outDegree;
			}
			return i;
		}
		nd.id = dg.numNodes;
		dg.node[dg.numNodes] = nd;
		dg.numNodes++;
		return nd.id;
	}

	int AttachArc(DGraph dg, DGArc nar) {
		int arcId = -1;
		int newNumber = 0;
		DGNode head = nar.head, tail = nar.tail;
		DGArc ars[] = null, probe = null;
		if (tail == null || head == null)
			return arcId;
		if (dg.numArcs == dg.maxArcs - 1) {
			dg.maxArcs += BLOCK_SIZE;
			ars = new DGArc[dg.maxArcs];
			System.arraycopy(dg.arc, 0, ars, 0, dg.maxArcs - BLOCK_SIZE);
			dg.arc = ars;
		}
		for (int i = 0; i < tail.outDegree; i++) { /* parallel arc */
			probe = tail.outArc[i];
			if (probe.head == head && probe.length == nar.length) {
				return probe.id;
			}
		}

		nar.id = dg.numArcs;
		arcId = dg.numArcs;
		dg.arc[dg.numArcs] = nar;
		dg.numArcs++;

		head.inArc[head.inDegree] = nar;
		head.inDegree++;
		if (head.inDegree >= head.maxInDegree) {
			newNumber = head.maxInDegree + SMALL_BLOCK_SIZE;
			ars = new DGArc[newNumber];
			System.arraycopy(head.inArc, 0, ars, 0, head.inDegree);
			head.inArc = ars;
			head.maxInDegree = newNumber;
		}
		tail.outArc[tail.outDegree] = nar;
		tail.outDegree++;
		if (tail.outDegree >= tail.maxOutDegree) {
			newNumber = tail.maxOutDegree + SMALL_BLOCK_SIZE;
			ars = new DGArc[newNumber];
			System.arraycopy(tail.outArc, 0, ars, 0, tail.outDegree);
			tail.outArc = ars;
			tail.maxOutDegree = newNumber;
		}
		return arcId;
	}

	int verify(String bmname, double rnm2) {
		double verifyValue = 0.0;
		double epsilon = 1.0E-8;
		char cls = CLASS;
		int verified = -1;
		if (cls != 'U') {
			if (cls == 'S') {
				if (bmname.indexOf("BH") >= 0) {
					verifyValue = 30892725.0;
				} else if (bmname.indexOf("WH") >= 0) {
					verifyValue = 67349758.0;
				} else if (bmname.indexOf("SH") >= 0) {
					verifyValue = 58875767.0;
				} else {
					System.err.printf("No such benchmark as %s.\n", bmname);
				}
				verified = 0;
			} else if (cls == 'W') {
				if (bmname.indexOf("BH") >= 0) {
					verifyValue = 4102461.0;
				} else if (bmname.indexOf("WH") >= 0) {
					verifyValue = 204280762.0;
				} else if (bmname.indexOf("SH") >= 0) {
					verifyValue = 186944764.0;
				} else {
					System.err.printf("No such benchmark as %s.\n", bmname);
				}
				verified = 0;
			} else if (cls == 'A') {
				if (bmname.indexOf("BH") >= 0) {
					verifyValue = 17809491.0;
				} else if (bmname.indexOf("WH") >= 0) {
					verifyValue = 1289925229.0;
				} else if (bmname.indexOf("SH") >= 0) {
					verifyValue = 610856482.0;
				} else {
					System.err.printf("No such benchmark as %s.\n", bmname);
				}
				verified = 0;
			} else if (cls == 'B') {
				if (bmname.indexOf("BH") >= 0) {
					verifyValue = 4317114.0;
				} else if (bmname.indexOf("WH") >= 0) {
					verifyValue = 7877279917.0;
				} else if (bmname.indexOf("SH") >= 0) {
					verifyValue = 1836863082.0;
				} else {
					System.err.printf("No such benchmark as %s.\n", bmname);
				}
				// Deviation from the official code: NPB 3.4.4 (C and the original NPB-MPJ) leave
				// verified = -1 for class B, although its verification values exist (README).
				verified = 0;
			} else if (cls == 'C') {
				if (bmname.indexOf("BH") >= 0) {
					verifyValue = 0.0;
				} else if (bmname.indexOf("WH") >= 0) {
					verifyValue = 0.0;
				} else if (bmname.indexOf("SH") >= 0) {
					verifyValue = 0.0;
				} else {
					System.err.printf("No such benchmark as %s.\n", bmname);
					verified = -1;
				}
			} else if (cls == 'D') {
				if (bmname.indexOf("BH") >= 0) {
					verifyValue = 0.0;
				} else if (bmname.indexOf("WH") >= 0) {
					verifyValue = 0.0;
				} else if (bmname.indexOf("SH") >= 0) {
					verifyValue = 0.0;
				} else {
					System.err.printf("No such benchmark as %s.\n", bmname);
				}
				verified = -1;
			} else {
				System.err.printf("No such class as %c.\n", cls);
			}
			System.err.printf(" %s L2 Norm = %f\n", bmname, rnm2);
			if (verified == -1) {
				System.err.printf(" No verification was performed.\n");
			} else if (rnm2 - verifyValue < epsilon
					&& rnm2 - verifyValue > -epsilon) {
				verified = 1;
				System.err.printf(" Deviation = %f\n", (rnm2 - verifyValue));
			} else {
				verified = 0;
				System.err.printf(" The correct verification value = %f\n",
						verifyValue);
				System.err.printf(" Got value = %f\n", rnm2);
			}
		} else {
			verified = -1;
		}
		return verified;
	}

	Arr randomFeatures(String bmname, int fdim, int id) {
		int len = getFeatureNum(bmname, id) * fdim;
		Arr feat = new Arr(len);
		int nxg = 2, nyg = 2, nzg = 2, nfg = 5;
		int nx = 421, ny = 419, nz = 1427, nf = 3527;
		long expon = (len * (id + 1)) % 3141592;
		int seedx = ipowMod(nxg, expon, nx), seedy = ipowMod(nyg, expon, ny), seedz = ipowMod(
				nzg, expon, nz), seedf = ipowMod(nfg, expon, nf);
		int i = 0;
		if (nodeTiming) {
			nodeTimer.reset(id + 1);
			nodeTimer.start(id + 1);
		}
		for (i = 0; i < len; i += fdim) {
			seedx = (seedx * nxg) % nx;
			seedy = (seedy * nyg) % ny;
			seedz = (seedz * nzg) % nz;
			seedf = (seedf * nfg) % nf;
			feat.val[i] = seedx;
			feat.val[i + 1] = seedy;
			feat.val[i + 2] = seedz;
			feat.val[i + 3] = seedf;
		}
		if (nodeTiming) {
			nodeTimer.stop(id + 1);
			System.err.printf("** RandomFeatures time in node %d = %f\n", id,
					nodeTimer.read(id + 1));
		}
		return feat;
	}

	int getFeatureNum(String mbname, int id) {
		double tran = 314159265.0;
		double A = 2 * id + 1;
		double values[] = rng.vrandlc(tran, A);
		tran = values[0];
		double denom = values[1];
		char cval = 'S';
		int rtfs = 0, len = 0;
		rtfs = ipowMod((int) (1 / denom) * (int) cval, (long) (2 * id + 1),
				2 * STD_DEVIATION);
		if (rtfs < 0)
			rtfs = -rtfs;
		len = NUM_SAMPLES - STD_DEVIATION + rtfs;
		return len;
	}

	int ipowMod(int a, long n, int md) {
		int seed = 1, q = a, r = 1;
		if (n < 0) {
			System.err.printf(
					"ipowMod: exponent must be nonnegative exp=%lld\n", n);
			n = -n;
		}
		if (md <= 0) {
			System.err.printf("ipowMod: module must be positive mod=%d", md);
			return 1;
		}
		if (n == 0)
			return 1;
		while (n > 1) {
			int n2 = (int) n / 2;
			if (n2 * 2 == n) {
				seed = (q * q) % md;
				q = seed;
				n = n2;
			} else {
				seed = (r * q) % md;
				r = seed;
				n = n - 1;
			}
		}
		seed = (r * q) % md;
		return seed;
	}

	int sendResults(DGraph dg, DGNode nd, Arr feat) {
		int tag = 0;
		DGArc ar = null;
		DGNode head = null;
		if (feat == null)
			return 0;
		for (int i = 0; i < nd.outDegree; i++) {
			ar = nd.outArc[i];
			if (!ar.tail.equals(nd))
				continue;
			head = ar.head;
			tag = ar.id;
			if (head.address != nd.address) {
				if (detailedTiming)
					timer.start(T_COMM);
				int sendbuff[] = { feat.len };
				MPI.COMM_WORLD.Send(sendbuff, 0, 1, MPI.INT, head.address, tag);
				MPI.COMM_WORLD.Send(feat.val, 0, feat.len, MPI.DOUBLE,
						head.address, tag);
				if (detailedTiming)
					timer.stop(T_COMM);
			}
		}
		return 1;
	}

	double reduceStreams(DGraph dg, DGNode nd) {
		double csum = 0.0;
		int len = 0, tag = 0;
		DGArc ar = null;
		DGNode tail = null;
		Arr feat = null;
		double retv = 0.0;

		for (int i = 0; i < nd.inDegree; i++) {
			ar = nd.inArc[i];
			if (ar.head != nd)
				continue;
			if (!ar.head.equals(nd))
				continue;
			tail = ar.tail;
			if (tail.address != nd.address) {
				len = 0;
				tag = ar.id;
				int recvbuff[] = { 0 };
				if (detailedTiming)
					timer.start(T_COMM);
				MPI.COMM_WORLD.Recv(recvbuff, 0, 1, MPI.INT, tail.address, tag);
				if (detailedTiming)
					timer.stop(T_COMM);
				len = recvbuff[0];
				feat = new Arr(len);
				if (detailedTiming)
					timer.start(T_COMM);
				MPI.COMM_WORLD.Recv(feat.val, 0, feat.len, MPI.DOUBLE,
						tail.address, tag);
				if (detailedTiming)
					timer.stop(T_COMM);
				csum += Reduce(feat, (nd.id + 1));
			} else {
				csum += Reduce(tail.feat, (nd.id + 1));
			}
		}
		if (nd.inDegree > 0)
			csum = (((long) csum) / nd.inDegree);
		retv = (nd.id + 1) * csum;
		return retv;
	}

	double Reduce(Arr a, int w) {
		double retv = 0.0;
		if (nodeTiming) {
			nodeTimer.reset(w);
			nodeTimer.start(w);
		}

		/* The casting needed for node and array dependent verifcation */
		retv = (int) (w * checkVal(a));

		if (nodeTiming) {
			nodeTimer.stop(w);
			System.err.printf("** Reduce time in node %d = %f\n", (w - 1),
					nodeTimer.read(w));
		}
		return retv;
	}

	double checkVal(Arr feat) {
		double csum = 0.0;
		for (int i = 0; i < feat.len; i++) {
			/* The truncation does not work since result will be 0 for large len */
			csum += feat.val[i] * feat.val[i] / feat.len;
		}
		return csum;
	}

	Arr combineStreams(DGraph dg, DGNode nd) {
		// (optimization) lazy accumulator: the original created a zero accumulator of NUM_SAMPLES*fieldDim and
		// windowFilter resampled it (to zeros) to the length of the first input if that was larger. It is now created with
		// the final length, max(NUM_SAMPLES*fieldDim, len): same result, without one allocation and one resampling.
		Arr resfeat = null;
		int len = 0, tag = 0;
		DGArc ar = null;
		DGNode tail = null;
		Arr feat = null, featp = null;

		if (nd.inDegree == 0)
			return null;
		for (int i = 0; i < nd.inDegree; i++) {
			ar = nd.inArc[i];
			if (ar.head != nd)
				continue;
			tail = ar.tail;
			if (tail.address != nd.address) {
				len = 0;
				tag = ar.id;
				int recvbuff[] = { 0 };
				if (detailedTiming)
					timer.start(T_COMM);
				MPI.COMM_WORLD.Recv(recvbuff, 0, 1, MPI.INT, tail.address, tag);
				if (detailedTiming)
					timer.stop(T_COMM);
				len = recvbuff[0];
				feat = new Arr(len);
				if (detailedTiming)
					timer.start(T_COMM);
				MPI.COMM_WORLD.Recv(feat.val, 0, feat.len, MPI.DOUBLE,
						tail.address, tag);
				if (detailedTiming)
					timer.stop(T_COMM);
				if (resfeat == null)
					resfeat = new Arr(Math.max(NUM_SAMPLES * fieldDim, feat.len));
				resfeat = windowFilter(resfeat, feat, nd.id);
			} else {
				featp = tail.feat;
				feat = new Arr(featp.len);
				System.arraycopy(featp.val, 0, feat.val, 0, featp.len);
				if (resfeat == null)
					resfeat = new Arr(Math.max(NUM_SAMPLES * fieldDim, feat.len));
				resfeat = windowFilter(resfeat, feat, nd.id);
			}
		}
		if (resfeat == null)
			resfeat = new Arr(NUM_SAMPLES * fieldDim);
		for (int i = 0; i < resfeat.len; i++)
			resfeat.val[i] = ((int) resfeat.val[i]) / nd.inDegree;
		nd.feat = resfeat;
		return nd.feat;
	}

	Arr windowFilter(Arr a, Arr b, int w) {
		int i = 0, j = 0, k = 0;
		double rms0 = 0.0, rms1 = 0.0, rmsm1 = 0.0;
		double weight = ((double) (w + 1)) / (w + 2);

		w += 1;
		if (nodeTiming) {
			nodeTimer.read(w);
			nodeTimer.start(w);
		}
		if (a.len < b.len)
			resample(a, b.len);
		if (a.len > b.len)
			resample(b, a.len);
		for (i = fieldDim; i < a.len - fieldDim; i += fieldDim) {
			rms0 = (a.val[i] - b.val[i]) * (a.val[i] - b.val[i])
					+ (a.val[i + 1] - b.val[i + 1])
					* (a.val[i + 1] - b.val[i + 1])
					+ (a.val[i + 2] - b.val[i + 2])
					* (a.val[i + 2] - b.val[i + 2])
					+ (a.val[i + 3] - b.val[i + 3])
					* (a.val[i + 3] - b.val[i + 3]);
			j = i + fieldDim;
			rms1 = (a.val[j] - b.val[j]) * (a.val[j] - b.val[j])
					+ (a.val[j + 1] - b.val[j + 1])
					* (a.val[j + 1] - b.val[j + 1])
					+ (a.val[j + 2] - b.val[j + 2])
					* (a.val[j + 2] - b.val[j + 2])
					+ (a.val[j + 3] - b.val[j + 3])
					* (a.val[j + 3] - b.val[j + 3]);
			j = i - fieldDim;
			rmsm1 = (a.val[j] - b.val[j]) * (a.val[j] - b.val[j])
					+ (a.val[j + 1] - b.val[j + 1])
					* (a.val[j + 1] - b.val[j + 1])
					+ (a.val[j + 2] - b.val[j + 2])
					* (a.val[j + 2] - b.val[j + 2])
					+ (a.val[j + 3] - b.val[j + 3])
					* (a.val[j + 3] - b.val[j + 3]);
			k = 0;
			if (rms1 < rms0) {
				k = 1;
				rms0 = rms1;
			}
			if (rmsm1 < rms0)
				k = -1;
			if (k == 0) {
				j = i + fieldDim;
				a.val[i] = weight * b.val[i];
				a.val[i + 1] = weight * b.val[i + 1];
				a.val[i + 2] = weight * b.val[i + 2];
				a.val[i + 3] = weight * b.val[i + 3];
			} else if (k == 1) {
				j = i + fieldDim;
				a.val[i] = weight * b.val[j];
				a.val[i + 1] = weight * b.val[j + 1];
				a.val[i + 2] = weight * b.val[j + 2];
				a.val[i + 3] = weight * b.val[j + 3];
			} else { /* if(k==-1) */
				j = i - fieldDim;
				a.val[i] = weight * b.val[j];
				a.val[i + 1] = weight * b.val[j + 1];
				a.val[i + 2] = weight * b.val[j + 2];
				a.val[i + 3] = weight * b.val[j + 3];
			}
		}
		if (nodeTiming) {
			nodeTimer.stop(w);
			System.err.printf("** WindowFilter time in node %d = %f\n",
					(w - 1), nodeTimer.read(w));
		}
		return a;
	}

	void resample(Arr a, int blen) {
		int jlo = 0, jhi = 0;
		double avval = 0.0;
		double nval[] = new double[blen];
		// (optimization) no redundant zeroing, and blen / a.len hoisted out of the loop: same result
		final int q = blen / a.len;
		for (int i = 1; i < a.len - 1; i++) {
			jlo = (int) (0.5 * (2 * i - 1) * q);
			jhi = (int) (0.5 * (2 * i + 1) * q);

			avval = a.val[i] / (jhi - jlo + 1);
			for (int j = jlo; j <= jhi; j++) {
				nval[j] += avval;
			}
		}
		nval[0] = a.val[0];
		nval[blen - 1] = a.val[a.len - 1];
		a.val = nval;
		a.len = blen;
	}

	private boolean strncmp(String s1, String s2, int len) {
		if (s1.length() < len || s2.length() < len)
			return true;
		else {
			for (int i = 0; i < len; i++) {
				if (s1.charAt(i) != s2.charAt(i))
					return true;
			}
		}
		return false;
	}

}

final class DGraph {
	protected int maxNodes, maxArcs, numNodes, numArcs;
	protected int id;
	protected String name;
	protected DGNode node[];
	protected DGArc arc[];

	DGraph(String name) {
		numNodes = 0;
		numArcs = 0;
		maxNodes = DT.BLOCK_SIZE;
		maxArcs = DT.BLOCK_SIZE;
		node = new DGNode[maxNodes];
		arc = new DGArc[maxArcs];
		this.name = name;
	}

}

final class DGArc {
	protected int id;
	protected DGNode tail, head;
	int length, width, attribute, maxWidth;

	DGArc(DGNode tail, DGNode head) {
		this.tail = tail;
		this.head = head;
	}

}

final class DGNode {
	int maxInDegree, maxOutDegree;
	int inDegree, outDegree;
	int id;
	String name;
	DGArc inArc[], outArc[];
	int depth, height, width;
	int color, attribute, address, verified;
	Arr feat;

	DGNode(String name) {
		attribute = 0;
		color = 0;
		inDegree = 0;
		outDegree = 0;
		maxInDegree = DT.SMALL_BLOCK_SIZE;
		maxOutDegree = DT.SMALL_BLOCK_SIZE;
		inArc = new DGArc[maxInDegree];
		outArc = new DGArc[maxOutDegree];
		this.name = name;
		feat = null;
	}

}

final class Arr {
	int len;
	double val[];

	Arr(int len) {
		this.len = len;
		val = new double[len];
	}

}
