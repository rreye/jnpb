/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB-MPJ by Damián Álvarez Mallón
 */
package NPB;

import NPB.common.io.BMResults;

import NPB.exceptions.NPBException;

public interface Benchmark {
	
	public void runBenchmark(String[] args) throws NPBException;
	public void build(String[] args) throws NPBException;
	public BMResults getResults();	
}
