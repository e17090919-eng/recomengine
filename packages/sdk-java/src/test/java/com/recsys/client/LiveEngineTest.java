package com.recsys.client;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end against a running engine: {@code make serve} then
 * {@code mvn test -Drecsys.it.url=http://127.0.0.1:8000} (make sdk-java-it). Skipped otherwise.
 */
class LiveEngineTest {
    @Test
    void predictAndWriteAgainstRealEngine() throws Exception {
        String url = System.getProperty("recsys.it.url", "");
        Assumptions.assumeTrue(!url.isBlank() && !url.startsWith("${"), "recsys.it.url not set");
        String app = System.getProperty("recsys.it.app", "demo");
        String market = System.getProperty("recsys.it.market", "CN-HAINAN");
        RecsysClient rc = RecsysClient.builder(url, app).timeout(Duration.ofSeconds(2)).build();

        PredictResult r = rc.predict(PredictRequest.of("u00001", app + "$##$home$##$feed$##$0").size(10).market(market));
        assertFalse(r.isFallback(), "engine should answer: " + r.getError());
        assertFalse(r.getItems().isEmpty());
        assertFalse(r.getRequestId().isEmpty());

        String item = r.getItems().get(0).getId();
        WriteResult w = rc.writeBehaviors(List.of(Map.of(
                "user_id", "u00001", "item_id", item, "event_type", "click", "market", market,
                "request_id", r.getRequestId(), "spm", app + "$##$home$##$feed$##$0")));  // no ts: the engine stamps its own clock
        assertTrue(w.isOk(), "write errors: " + w.getErrors());
        assertEquals(1, w.getAccepted());
    }
}
