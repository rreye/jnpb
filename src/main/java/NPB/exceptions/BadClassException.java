/*
 * Copyright (c) Universidade da Coruña
 * SPDX-License-Identifier: MIT
 * Based on NPB-MPJ by Damián Álvarez Mallón
 */
package NPB.exceptions;

public class BadClassException extends NPBException {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5571060765719973865L;

	public BadClassException(String message){
		super(message);
	}
}
