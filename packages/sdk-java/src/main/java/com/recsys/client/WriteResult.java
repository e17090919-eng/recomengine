package com.recsys.client;

import java.util.ArrayList;
import java.util.List;

public final class WriteResult {
    /** A rejected row; {@code index} is its position in the list you passed. */
    public static final class RowError {
        private final int index;
        private final String message;

        RowError(int index, String message) {
            this.index = index;
            this.message = message;
        }

        public int getIndex() { return index; }
        public String getMessage() { return message; }

        @Override
        public String toString() { return "#" + index + ": " + message; }
    }

    int accepted;
    final List<RowError> errors = new ArrayList<>();

    public int getAccepted() { return accepted; }
    public List<RowError> getErrors() { return errors; }
    public boolean isOk() { return errors.isEmpty(); }
}
