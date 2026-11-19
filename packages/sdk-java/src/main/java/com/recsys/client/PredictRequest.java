package com.recsys.client;

import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One recommendation request. Build with
 * {@code PredictRequest.of("u123", "cdf$##$home$##$feed$##$0").size(20).market("CN-HAINAN").fallbackIds(hot)}.
 */
public final class PredictRequest {
    final String userId, spm;
    Integer size;
    String market, parentItemId;
    Double allowanceRemaining, priceMin, priceMax;
    List<String> candidateIds = Collections.emptyList(), filterIds = Collections.emptyList(),
            cartIds = Collections.emptyList(), categories = Collections.emptyList(), brands = Collections.emptyList();
    Map<String, String> experiment = Collections.emptyMap();
    /** Served when the API is unreachable / times out / errors; null = no fallback (auth errors then throw). */
    List<String> fallbackIds;
    Duration timeout;

    private PredictRequest(String userId, String spm) {
        this.userId = userId;
        this.spm = spm;
    }

    /** {@code spm} = app$##$page$##$block$##$position, e.g. {@code cdf$##$home$##$feed$##$0}. */
    public static PredictRequest of(String userId, String spm) {
        return new PredictRequest(userId, spm);
    }

    public PredictRequest size(int v) { this.size = v; return this; }
    public PredictRequest market(String v) { this.market = v; return this; }
    /** Detail-page recommendations: the item being viewed. */
    public PredictRequest parentItemId(String v) { this.parentItemId = v; return this; }
    public PredictRequest candidateIds(List<String> v) { this.candidateIds = RecsysClient.copy(v); return this; }
    /** Items that must not be returned (e.g. already on screen). */
    public PredictRequest filterIds(List<String> v) { this.filterIds = RecsysClient.copy(v); return this; }
    public PredictRequest cartIds(List<String> v) { this.cartIds = RecsysClient.copy(v); return this; }
    /** Remaining duty-free allowance for this user (market currency). */
    public PredictRequest allowanceRemaining(double v) { this.allowanceRemaining = v; return this; }
    public PredictRequest categories(List<String> v) { this.categories = RecsysClient.copy(v); return this; }
    public PredictRequest brands(List<String> v) { this.brands = RecsysClient.copy(v); return this; }
    public PredictRequest priceMin(double v) { this.priceMin = v; return this; }
    public PredictRequest priceMax(double v) { this.priceMax = v; return this; }
    /** Force an experiment bucket (debug / QA): layer -> group. */
    public PredictRequest experiment(Map<String, String> v) { this.experiment = new LinkedHashMap<>(v); return this; }
    public PredictRequest fallbackIds(List<String> v) { this.fallbackIds = RecsysClient.copy(v); return this; }
    public PredictRequest timeout(Duration v) { this.timeout = v; return this; }
}
