package com.recsys.client;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class PredictResult {
    public static final class Item {
        private final String id, transData;
        private final int rank;
        private final Double score;

        Item(String id, int rank, String transData, Double score) {
            this.id = id;
            this.rank = rank;
            this.transData = transData;
            this.score = score;
        }

        public String getId() { return id; }
        public int getRank() { return rank; }
        /** Pass back unchanged with impression / click events (attribution). */
        public String getTransData() { return transData; }
        /** Ranker score, when the server exposes it; may be null. */
        public Double getScore() { return score; }

        @Override
        public String toString() { return "Item{" + id + " #" + rank + "}"; }
    }

    private final String requestId, error;
    private final List<Item> items;
    private final List<String> degraded;
    private final boolean fallback;

    PredictResult(String requestId, List<Item> items, List<String> degraded, boolean fallback, String error) {
        this.requestId = requestId;
        this.items = Collections.unmodifiableList(items);
        this.degraded = Collections.unmodifiableList(degraded);
        this.fallback = fallback;
        this.error = error;
    }

    static PredictResult fallback(List<String> ids, Integer size, String err) {
        List<Item> items = new ArrayList<>();
        if (ids != null) {
            int n = size != null && size > 0 ? Math.min(size, ids.size()) : ids.size();
            for (int i = 0; i < n; i++) items.add(new Item(ids.get(i), i + 1, "", null));
        }
        return new PredictResult("", items, Collections.emptyList(), true, err);
    }

    public String getRequestId() { return requestId; }
    public List<Item> getItems() { return items; }
    /** Server-side degradation reasons (e.g. a recall channel timed out); empty when fully served. */
    public List<String> getDegraded() { return degraded; }
    /** True when the SDK served the caller's fallback list instead of the API's answer. */
    public boolean isFallback() { return fallback; }
    public String getError() { return error; }

    public List<String> ids() {
        List<String> out = new ArrayList<>(items.size());
        for (Item i : items) out.add(i.getId());
        return out;
    }
}
