# recsys-client (Java)

Java SDK for the recsys API — same behaviour as the Python SDK (`packages/sdk-python`).
Java ≥ 11, one dependency (Jackson). Build: `make sdk-java-test` (JDK + Maven from `make infra-native-install`).

```java
RecsysClient rc = RecsysClient.builder("https://rec.example.com", "cdf")
        .token(System.getenv("RECSYS_TOKEN"))     // adds Tenant-Signature
        .timeout(Duration.ofMillis(300))
        .build();                                  // thread-safe: one per process

PredictResult res = rc.predict(PredictRequest.of("u123", "cdf$##$home$##$feed$##$0")
        .size(20).market("CN-HAINAN")
        .fallbackIds(cachedHotList));              // served if the API is down / times out
for (PredictResult.Item it : res.getItems()) show(it.getId(), it.getTransData());

rc.writeBehaviors(List.of(Map.of(
        "user_id", "u123", "item_id", "g001", "event_type", "click",
        "request_id", res.getRequestId(), "spm", "cdf$##$home$##$feed$##$3")));
rc.done("behavior", "20261114");                   // day complete -> daily batch may start
```

- `predict` never throws on transport errors, timeouts or 5xx: it returns `fallbackIds` with `isFallback() == true`.
  Auth / unknown-app errors (401/404) throw `RecsysException` when no fallback list is given.
- Writes are chunked (`chunkSize`, default 1000) and retried on 5xx / transport errors; rejected rows come back in
  `WriteResult.getErrors()` with their index in the list you passed.
- Signature = `hex(SHA256(token ‖ body ‖ tenantId ‖ ts ‖ nonce))`, checked against the server implementation in the tests.

Tests: `RecsysClientTest` uses an in-process fake server; `LiveEngineTest` runs against a real engine
(`make serve` or `make serve-infra`, then `make sdk-java-it`) and is skipped otherwise.
