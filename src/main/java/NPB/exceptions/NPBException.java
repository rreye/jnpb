/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB-MPJ by Damián Álvarez Mallón
 */
package NPB.exceptions;

public abstract class NPBException extends Exception {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -6452345393574101768L;

	protected NPBException(String message){
		super("NPB Exception: "+message);
	}
}
