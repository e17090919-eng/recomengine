-- Behaviour log for reporting / experiment metrics / data exploration.
-- Real-time path: Kafka topic behavior-events (JSON rows in the canonical schema) -> behavior.

CREATE DATABASE IF NOT EXISTS recsys;

CREATE TABLE IF NOT EXISTS recsys.behavior
(
    request_id      String,
    user_id         String,
    item_id         String,
    event_type      LowCardinality(String),
    ts              DateTime('UTC'),
    spm             String,
    market          LowCardinality(String),
    parent_item_id  String,
    paid_amount     Float64
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (market, event_type, user_id, ts)
TTL ts + INTERVAL 400 DAY;

CREATE TABLE IF NOT EXISTS recsys.behavior_kafka
(
    request_id      String,
    user_id         String,
    item_id         String,
    event_type      String,
    ts              Int64,
    spm             String,
    market          String,
    parent_item_id  String,
    paid_amount     Float64
)
ENGINE = Kafka
SETTINGS kafka_broker_list = 'kafka:19092',
         kafka_topic_list = 'behavior-events',
         kafka_group_name = 'clickhouse-behavior',
         kafka_format = 'JSONEachRow',
         kafka_skip_broken_messages = 100;

CREATE MATERIALIZED VIEW IF NOT EXISTS recsys.behavior_mv TO recsys.behavior AS
SELECT request_id, user_id, item_id, event_type, toDateTime(ts, 'UTC') AS ts, spm, market, parent_item_id, paid_amount
FROM recsys.behavior_kafka;

-- daily funnel per slot (first three SPM segments) — the base of the effect report
CREATE VIEW IF NOT EXISTS recsys.daily_funnel AS
SELECT
    toDate(ts) AS day,
    market,
    arrayStringConcat(arraySlice(splitByString('$##$', spm), 1, 3), '$##$') AS slot_spm,
    countIf(event_type = 'impression') AS impressions,
    countIf(event_type = 'click')      AS clicks,
    countIf(event_type = 'add_cart')   AS carts,
    countIf(event_type = 'purchase')   AS purchases,
    sumIf(paid_amount, event_type = 'purchase') AS gmv,
    clicks / nullIf(impressions, 0)    AS ctr
FROM recsys.behavior
GROUP BY day, market, slot_spm;
