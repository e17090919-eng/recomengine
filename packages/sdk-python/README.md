# recsys-client

Python SDK for the recsys API. Only depends on `httpx`; Python ≥ 3.9.

```python
from recsys_client import RecsysClient

rc = RecsysClient("https://rec.example.com", app_id="cdf", tenant_id="cdf", token="***", timeout=0.3)

res = rc.predict(
    user_id="u123", spm="cdf$##$home$##$feed$##$0", size=20, market="CN-HAINAN", fallback_ids=cached_hot_list
)  # served if the API is down / times out
for it in res.items:
    show(it.id, trans_data=it.trans_data)  # pass trans_data back with impression/click events

rc.write_behaviors(
    [
        {
            "user_id": "u123",
            "item_id": "g001",
            "event_type": "click",
            "request_id": res.request_id,
            "spm": "cdf$##$home$##$feed$##$3",
        }
    ]
)
rc.done("behavior", "20261114")  # day complete -> daily batch may start
```

- Signing (`Tenant-Signature`) is added automatically when `token` is set.
- `predict` never raises on transport errors or timeouts: it returns `fallback_ids` with `res.fallback = True`.
- Writes are chunked (`chunk_size`, default 1000) and retried on 5xx / transport errors.
- `AsyncRecsysClient` has the same methods as coroutines.
