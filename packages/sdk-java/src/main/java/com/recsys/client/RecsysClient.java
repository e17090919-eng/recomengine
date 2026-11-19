package com.recsys.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Thread-safe blocking client for the recsys API. Mirrors the Python SDK (packages/sdk-python):
 * <ul>
 *   <li>signing ({@code Tenant-Signature}) when a token is set;</li>
 *   <li>{@link #predict} never throws on transport errors / timeouts / 5xx: it returns the
 *       caller's fallback list with {@code fallback=true};</li>
 *   <li>writes are chunked and retried on 5xx and transport errors.</li>
 * </ul>
 * Build one per process and share it: {@code RecsysClient.builder(url, appId).token(t).build()}.
 */
public final class RecsysClient implements AutoCloseable {
    public static final String SPM_SEP = "$##$";
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final SecureRandom RANDOM = new SecureRandom();

    private final String baseUrl, appId, tenantId, token;
    private final Duration timeout, writeTimeout;
    private final int retries, chunkSize;
    private final HttpClient http;

    private RecsysClient(Builder b) {
        this.baseUrl = b.baseUrl.replaceAll("/+$", "");
        this.appId = b.appId;
        this.tenantId = b.tenantId != null ? b.tenantId : b.appId;
        this.token = b.token;
        this.timeout = b.timeout;
        this.writeTimeout = b.writeTimeout;
        this.retries = b.retries;
        this.chunkSize = b.chunkSize;
        this.http = b.http != null ? b.http
                : HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).version(HttpClient.Version.HTTP_1_1).build();
    }

    public static Builder builder(String baseUrl, String appId) {
        return new Builder(baseUrl, appId);
    }

    /** hex(SHA256(token || body || tenantId || ts || nonce)) — identical to the server's check. */
    public static String sign(String token, byte[] body, String tenantId, String ts, String nonce) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            md.update(token.getBytes(StandardCharsets.UTF_8));
            md.update(body);
            md.update(tenantId.getBytes(StandardCharsets.UTF_8));
            md.update(ts.getBytes(StandardCharsets.UTF_8));
            md.update(nonce.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte x : md.digest()) sb.append(String.format("%02x", x));
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    // ------------------------------------------------------------------ predict / callback

    public PredictResult predict(PredictRequest req) {
        ObjectNode body;
        try {
            body = predictBody(req);
        } catch (IllegalArgumentException e) {
            if (req.fallbackIds == null) throw e;
            return PredictResult.fallback(req.fallbackIds, req.size, e.toString());
        }
        Duration t = req.timeout != null ? req.timeout : timeout;
        try {
            JsonNode r = post("/predict/api/" + appId + "/default", body, t, 0);
            return parsePredict(r);
        } catch (RecsysException e) {
            if (req.fallbackIds == null && (e.getCode() == 401 || e.getCode() == 404)) throw e;
            return PredictResult.fallback(req.fallbackIds, req.size, e.toString());
        } catch (IOException e) {
            return PredictResult.fallback(req.fallbackIds, req.size, e.toString());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return PredictResult.fallback(req.fallbackIds, req.size, e.toString());
        }
    }

    /** Report which of the recommended items were actually shown (exposure). */
    public void callback(String userId, String requestId, List<String> itemIds, String spm)
            throws IOException, InterruptedException {
        ObjectNode body = JSON.createObjectNode();
        body.put("uid", userId).put("request_id", requestId);
        if (spm != null) body.put("spm", spm); else body.putNull("spm");
        ArrayNode items = body.putArray("items");
        for (String id : itemIds) items.addObject().put("id", id);
        post("/predict/api/" + appId + "/callback", body, timeout, retries);
    }

    // ------------------------------------------------------------------ writes

    /** Write rows to {@code behavior}, {@code item} or {@code user}. Rows are chunked; per-row errors come back in the result. */
    public WriteResult write(String table, List<? extends Map<String, ?>> rows) throws IOException, InterruptedException {
        WriteResult total = new WriteResult();
        for (int start = 0; start < rows.size(); start += chunkSize) {
            List<? extends Map<String, ?>> chunk = rows.subList(start, Math.min(rows.size(), start + chunkSize));
            JsonNode r = post("/data/api/" + appId + "/" + table + "/write", JSON.valueToTree(chunk), writeTimeout, retries);
            int code = r.path("code").asInt(0);
            total.accepted += r.path("accepted").asInt(0);
            for (JsonNode e : r.path("errors")) {
                total.errors.add(new WriteResult.RowError(start + e.path("index").asInt(0), e.path("message").asText("")));
            }
            if ((code == 400 && r.path("errors").size() == 0) || code == 401 || code == 404) {
                throw new RecsysException(code, r.path("message").asText(""));
            }
        }
        return total;
    }

    public WriteResult writeBehaviors(List<? extends Map<String, ?>> rows) throws IOException, InterruptedException {
        return write("behavior", rows);
    }

    public WriteResult writeItems(List<? extends Map<String, ?>> rows) throws IOException, InterruptedException {
        return write("item", rows);
    }

    public WriteResult writeUsers(List<? extends Map<String, ?>> rows) throws IOException, InterruptedException {
        return write("user", rows);
    }

    /** Mark a day's data for {@code table} complete (date = yyyyMMdd); the daily batch may then start. */
    public void done(String table, String date) throws IOException, InterruptedException {
        ObjectNode body = JSON.createObjectNode().put("table", table).put("date", date);
        JsonNode r = post("/data/api/" + appId + "/done", body, writeTimeout, retries);
        if (r.path("code").asInt(0) != 200) throw new RecsysException(r.path("code").asInt(0), r.path("message").asText(""));
    }

    @Override
    public void close() {
        // java.net.http.HttpClient has no close() before JDK 21; connections are released on GC.
    }

    // ------------------------------------------------------------------ internals

    ObjectNode predictBody(PredictRequest q) {
        Objects.requireNonNull(q.userId, "userId");
        Objects.requireNonNull(q.spm, "spm");
        if (q.spm.split(java.util.regex.Pattern.quote(SPM_SEP), -1).length != 4) {
            throw new IllegalArgumentException("spm must be 'app$##$page$##$block$##$position', got " + q.spm);
        }
        ObjectNode body = JSON.createObjectNode();
        body.putObject("user").put("uid", q.userId);
        ObjectNode ctx = body.putObject("context");
        ctx.put("spm", q.spm);
        ObjectNode feature = ctx.putObject("feature");
        ObjectNode sf = feature.putObject("stringFeature");
        ObjectNode df = feature.putObject("doubleFeature");
        ObjectNode saf = feature.putObject("stringArrayFeature");
        if (q.market != null) sf.put("market", q.market);
        if (q.allowanceRemaining != null) df.put("allowance_remaining", q.allowanceRemaining);
        if (!q.cartIds.isEmpty()) values(saf, "cart_items", q.cartIds);
        ObjectNode flt = ctx.putObject("filter");
        ObjectNode fsa = flt.putObject("stringArrayFeature");
        ObjectNode fd = flt.putObject("doubleFeature");
        if (!q.categories.isEmpty()) values(fsa, "category", q.categories);
        if (!q.brands.isEmpty()) values(fsa, "brand", q.brands);
        if (q.priceMin != null) fd.put("price_min", q.priceMin);
        if (q.priceMax != null) fd.put("price_max", q.priceMax);
        ObjectNode extra = ctx.putObject("extra");
        q.experiment.forEach((k, v) -> extra.put("exp." + k, v));
        ArrayNode cand = body.putArray("candidateItems");
        q.candidateIds.forEach(i -> cand.addObject().put("id", i));
        ArrayNode filt = body.putArray("filterItems");
        q.filterIds.forEach(i -> filt.addObject().put("id", i));
        if (q.parentItemId != null) body.putObject("parentItem").put("id", q.parentItemId);
        if (q.size != null && q.size > 0) body.put("size", q.size);
        return body;
    }

    private static void values(ObjectNode parent, String key, List<String> vals) {
        ArrayNode a = parent.putObject(key).putArray("values");
        vals.forEach(a::add);
    }

    static PredictResult parsePredict(JsonNode r) {
        int code = r.path("code").asInt(0);
        if (code != 200) throw new RecsysException(code, r.path("message").asText(""));
        List<PredictResult.Item> items = new ArrayList<>();
        int n = 0;
        for (JsonNode x : r.path("value").path("items")) {
            n++;
            JsonNode score = x.path("extra").path("score");
            Double s = (score.isMissingNode() || score.isNull() || score.asText().isEmpty()) ? null : score.asDouble();
            items.add(new PredictResult.Item(x.path("id").asText(), x.path("rank").asInt(n), x.path("transData").asText(""), s));
        }
        List<String> degraded = new ArrayList<>();
        r.path("degraded").forEach(d -> degraded.add(d.asText()));
        return new PredictResult(r.path("request_id").asText(""), items, degraded, false, null);
    }

    private JsonNode post(String path, JsonNode obj, Duration t, int maxRetries) throws IOException, InterruptedException {
        byte[] body = JSON.writeValueAsBytes(obj);
        IOException last = null;
        for (int attempt = 0; attempt <= maxRetries; attempt++) {
            HttpRequest.Builder rb = HttpRequest.newBuilder(URI.create(baseUrl + path))
                    .timeout(t)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofByteArray(body));
            if (token != null) {
                String ts = Long.toString(System.currentTimeMillis() / 1000);
                byte[] nb = new byte[8];
                RANDOM.nextBytes(nb);
                StringBuilder nonce = new StringBuilder();
                for (byte x : nb) nonce.append(String.format("%02x", x));
                rb.header("Tenant-Id", tenantId).header("Tenant-Ts", ts).header("Tenant-Nonce", nonce.toString())
                  .header("Tenant-Signature", sign(token, body, tenantId, ts, nonce.toString()));
            }
            try {
                HttpResponse<byte[]> resp = http.send(rb.build(), HttpResponse.BodyHandlers.ofByteArray());
                int sc = resp.statusCode();
                if (sc >= 500) {
                    last = new IOException("HTTP " + sc);
                } else if (sc >= 400) {
                    JsonNode j = tryParse(resp.body());
                    throw new RecsysException(j != null ? j.path("code").asInt(sc) : sc,
                            j != null ? j.path("message").asText("HTTP " + sc) : "HTTP " + sc);
                } else {
                    return JSON.readTree(resp.body());
                }
            } catch (IOException e) {
                last = e;
            }
            if (attempt < maxRetries) Thread.sleep(Math.min(50L << attempt, 1000L));
        }
        throw last;
    }

    private static JsonNode tryParse(byte[] b) {
        try {
            return JSON.readTree(b);
        } catch (IOException e) {
            return null;
        }
    }

    // ------------------------------------------------------------------ builder

    public static final class Builder {
        private final String baseUrl, appId;
        private String tenantId, token;
        private Duration timeout = Duration.ofMillis(300), writeTimeout = Duration.ofSeconds(5);
        private int retries = 1, chunkSize = 1000;
        private HttpClient http;

        private Builder(String baseUrl, String appId) {
            this.baseUrl = Objects.requireNonNull(baseUrl);
            this.appId = Objects.requireNonNull(appId);
        }

        public Builder tenantId(String v) { this.tenantId = v; return this; }
        public Builder token(String v) { this.token = v; return this; }
        public Builder timeout(Duration v) { this.timeout = v; return this; }
        public Builder writeTimeout(Duration v) { this.writeTimeout = v; return this; }
        public Builder retries(int v) { this.retries = v; return this; }
        public Builder chunkSize(int v) { if (v < 1) throw new IllegalArgumentException("chunkSize"); this.chunkSize = v; return this; }
        /** Inject a shared / instrumented HttpClient. */
        public Builder httpClient(HttpClient v) { this.http = v; return this; }

        public RecsysClient build() { return new RecsysClient(this); }
    }

    static List<String> copy(List<String> v) {
        return v == null ? Collections.emptyList() : List.copyOf(v);
    }
}
