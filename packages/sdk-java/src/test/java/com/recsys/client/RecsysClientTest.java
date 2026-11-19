package com.recsys.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

/** Unit tests against an in-process fake server (no network, no engine). */
class RecsysClientTest {
    static final ObjectMapper JSON = new ObjectMapper();
    HttpServer server;
    final List<Map<String, Object>> seen = new CopyOnWriteArrayList<>();
    volatile Function<JsonNode, Object[]> handler; // -> {status, body}
    String url;

    @BeforeEach
    void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", ex -> {
            byte[] in = ex.getRequestBody().readAllBytes();
            JsonNode body = JSON.readTree(in);
            seen.add(Map.of("path", ex.getRequestURI().getPath(), "body", body,
                    "sig", String.valueOf(ex.getRequestHeaders().getFirst("Tenant-Signature")),
                    "ts", String.valueOf(ex.getRequestHeaders().getFirst("Tenant-Ts")),
                    "nonce", String.valueOf(ex.getRequestHeaders().getFirst("Tenant-Nonce")),
                    "raw", new String(in, StandardCharsets.UTF_8)));
            Object[] r = handler.apply(body);
            byte[] out = JSON.writeValueAsBytes(r[1]);
            ex.sendResponseHeaders((Integer) r[0], out.length);
            ex.getResponseBody().write(out);
            ex.close();
        });
        server.start();
        url = "http://127.0.0.1:" + server.getAddress().getPort();
    }

    @AfterEach
    void stop() {
        server.stop(0);
    }

    @Test
    void signatureMatchesServerImplementation() {
        // reference value from recsys_core.signing.sign('tok', b'{"a":1}', 'cdf', '1700000000', 'abcd')
        assertEquals("41e0a6811f6ffe971e88301a397fdf6843e1e0ffa69d387f573d7e9901cd7791",
                RecsysClient.sign("tok", "{\"a\":1}".getBytes(StandardCharsets.UTF_8), "cdf", "1700000000", "abcd"));
    }

    @Test
    void predictBuildsVolcengineShapedBodyAndParsesItems() {
        handler = b -> new Object[]{200, Map.of("code", 200, "request_id", "r1", "degraded", List.of("recall:swing"),
                "value", Map.of("items", List.of(
                        Map.of("id", "g1", "rank", 1, "transData", "t1", "extra", Map.of("score", "0.9")),
                        Map.of("id", "g2", "rank", 2))))};
        RecsysClient rc = RecsysClient.builder(url, "cdf").token("tok").build();
        PredictResult r = rc.predict(PredictRequest.of("u1", "cdf$##$home$##$feed$##$0").size(2).market("NZ-AKL")
                .cartIds(List.of("g9")).allowanceRemaining(1200).brands(List.of("b1")).priceMax(500)
                .parentItemId("g7").experiment(Map.of("rank", "gbdt")));
        assertFalse(r.isFallback());
        assertEquals("r1", r.getRequestId());
        assertEquals(List.of("g1", "g2"), r.ids());
        assertEquals("t1", r.getItems().get(0).getTransData());
        assertEquals(0.9, r.getItems().get(0).getScore(), 1e-9);
        assertNull(r.getItems().get(1).getScore());
        assertEquals(List.of("recall:swing"), r.getDegraded());

        Map<String, Object> req = seen.get(0);
        assertEquals("/predict/api/cdf/default", req.get("path"));
        JsonNode b = (JsonNode) req.get("body");
        assertEquals("u1", b.at("/user/uid").asText());
        assertEquals("NZ-AKL", b.at("/context/feature/stringFeature/market").asText());
        assertEquals(1200.0, b.at("/context/feature/doubleFeature/allowance_remaining").asDouble());
        assertEquals("g9", b.at("/context/feature/stringArrayFeature/cart_items/values/0").asText());
        assertEquals("b1", b.at("/context/filter/stringArrayFeature/brand/values/0").asText());
        assertEquals(500.0, b.at("/context/filter/doubleFeature/price_max").asDouble());
        assertEquals("gbdt", b.at("/context/extra/exp.rank").asText());
        assertEquals("g7", b.at("/parentItem/id").asText());
        assertEquals(2, b.at("/size").asInt());
        // signature is over the exact bytes sent
        assertEquals(RecsysClient.sign("tok", ((String) req.get("raw")).getBytes(StandardCharsets.UTF_8), "cdf",
                (String) req.get("ts"), (String) req.get("nonce")), req.get("sig"));
    }

    @Test
    void predictFallsBackOnTimeoutAnd5xx() {
        handler = b -> {
            try { Thread.sleep(500); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            return new Object[]{200, Map.of("code", 200)};
        };
        RecsysClient rc = RecsysClient.builder(url, "cdf").timeout(Duration.ofMillis(100)).build();
        PredictResult r = rc.predict(PredictRequest.of("u1", "a$##$b$##$c$##$0").size(2).fallbackIds(List.of("h1", "h2", "h3")));
        assertTrue(r.isFallback());
        assertEquals(List.of("h1", "h2"), r.ids());

        handler = b -> new Object[]{503, Map.of("code", 503, "message", "busy")};
        r = rc.predict(PredictRequest.of("u1", "a$##$b$##$c$##$0").fallbackIds(List.of("h1")));
        assertTrue(r.isFallback());
        assertEquals(List.of("h1"), r.ids());
    }

    @Test
    void authErrorThrowsWithoutFallbackAndBadSpmIsRejected() {
        handler = b -> new Object[]{200, Map.of("code", 401, "message", "bad signature")};
        RecsysClient rc = RecsysClient.builder(url, "cdf").build();
        RecsysException e = assertThrows(RecsysException.class, () -> rc.predict(PredictRequest.of("u1", "a$##$b$##$c$##$0")));
        assertEquals(401, e.getCode());
        assertThrows(IllegalArgumentException.class, () -> rc.predict(PredictRequest.of("u1", "home-feed")));
        assertTrue(rc.predict(PredictRequest.of("u1", "home-feed").fallbackIds(List.of("h"))).isFallback());
    }

    @Test
    void writesAreChunkedRetriedAndErrorIndexesAreGlobal() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        handler = b -> {
            if (calls.incrementAndGet() == 1) return new Object[]{502, Map.of()}; // first attempt fails -> retried
            List<Object> errs = new ArrayList<>();
            if (b.size() == 2 && b.get(1).path("item_id").asText().equals("bad")) errs.add(Map.of("index", 1, "message", "unknown item"));
            return new Object[]{200, Map.of("code", 200, "accepted", b.size() - errs.size(), "errors", errs)};
        };
        RecsysClient rc = RecsysClient.builder(url, "cdf").chunkSize(2).retries(2).build();
        List<Map<String, Object>> rows = List.of(
                Map.of("user_id", "u", "item_id", "g1", "event_type", "click"),
                Map.of("user_id", "u", "item_id", "g2", "event_type", "click"),
                Map.of("user_id", "u", "item_id", "g3", "event_type", "click"),
                Map.of("user_id", "u", "item_id", "bad", "event_type", "click"));
        WriteResult r = rc.writeBehaviors(rows);
        assertEquals(3, r.getAccepted());
        assertEquals(1, r.getErrors().size());
        assertEquals(3, r.getErrors().get(0).getIndex());
        assertEquals(3, calls.get()); // 1 failed + 2 chunks
        assertEquals("/data/api/cdf/behavior/write", seen.get(0).get("path"));
    }

    @Test
    void doneAndCallback() throws Exception {
        handler = b -> new Object[]{200, Map.of("code", 200)};
        RecsysClient rc = RecsysClient.builder(url, "cdf").build();
        rc.done("behavior", "20261118");
        rc.callback("u1", "r1", List.of("g1", "g2"), null);
        assertEquals("/data/api/cdf/done", seen.get(0).get("path"));
        assertEquals("20261118", ((JsonNode) seen.get(0).get("body")).path("date").asText());
        assertEquals("/predict/api/cdf/callback", seen.get(1).get("path"));
        assertEquals("g2", ((JsonNode) seen.get(1).get("body")).at("/items/1/id").asText());
    }
}
