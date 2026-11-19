package com.recsys.client;

/** The API answered with an error code (bad request, auth failure, unknown app...). */
public class RecsysException extends RuntimeException {
    private final int code;

    public RecsysException(int code, String message) {
        super(code + ": " + message);
        this.code = code;
    }

    public int getCode() { return code; }
}
