/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB-MPJ by Damián Álvarez Mallón
 */
package NPB.exceptions;

public class BadInputException extends NPBException {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6097934731317009937L;

	public BadInputException(String message){
		super(message);
	}
}
