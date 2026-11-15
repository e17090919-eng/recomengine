"""Sync and async clients. Request/response shapes follow the server's Volcengine-compatible
predict API; see the server's README for field semantics."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import httpx

SPM_SEP = "$##$"


class RecsysError(Exception):
    def __init__(self, code: int, message: str, payload: Optional[dict] = None):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.payload = code, message, payload or {}


def sign(token: str, body: bytes, tenant_id: str, ts: str, nonce: str) -> str:
    """hex(SHA256(token || body || tenant_id || ts || nonce))"""
    h = hashlib.sha256()
    for part in (token.encode(), body, tenant_id.encode(), ts.encode(), nonce.encode()):
        h.update(part)
    return h.hexdigest()


@dataclass
class Item:
    id: str
    rank: int
    trans_data: str = ""
    score: Optional[float] = None


@dataclass
class PredictResult:
    request_id: str
    items: List[Item]
    degraded: List[str] = field(default_factory=list)  # server-side degradation reasons
    fallback: bool = False  # True when the SDK served `fallback_ids` (API unreachable / error)
    error: Optional[str] = None

    @property
    def ids(self) -> List[str]:
        return [i.id for i in self.items]


@dataclass
class WriteResult:
    accepted: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)  # {"index": i (in the rows you passed), "message"}

    @property
    def ok(self) -> bool:
        return not self.errors


class _Base:
    def __init__(
        self,
        base_url: str,
        app_id: str,
        tenant_id: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 0.3,
        write_timeout: float = 5.0,
        retries: int = 1,
        chunk_size: int = 1000,
    ):
        self.base_url, self.app_id = base_url.rstrip("/"), app_id
        self.tenant_id, self.token = tenant_id or app_id, token
        self.timeout, self.write_timeout, self.retries, self.chunk_size = timeout, write_timeout, retries, chunk_size

    # ---------------------------------------------------------------- request building
    def _headers(self, body: bytes) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.token:
            ts, nonce = str(int(time.time())), secrets.token_hex(8)
            h.update(
                {
                    "Tenant-Id": self.tenant_id,
                    "Tenant-Ts": ts,
                    "Tenant-Nonce": nonce,
                    "Tenant-Signature": sign(self.token, body, self.tenant_id, ts, nonce),
                }
            )
        return h

    @staticmethod
    def _body(obj: Any) -> bytes:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode()

    def _predict_body(
        self,
        user_id: str,
        spm: str,
        size: Optional[int],
        market: Optional[str],
        parent_item_id: Optional[str],
        candidate_ids: Optional[Sequence[str]],
        filter_ids: Optional[Sequence[str]],
        cart_ids: Optional[Sequence[str]],
        allowance_remaining: Optional[float],
        categories: Optional[Sequence[str]],
        brands: Optional[Sequence[str]],
        price_min: Optional[float],
        price_max: Optional[float],
        experiment: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        if len(spm.split(SPM_SEP)) != 4:
            raise ValueError(f"spm must be 'app{SPM_SEP}page{SPM_SEP}block{SPM_SEP}position', got {spm!r}")
        feature: Dict[str, Any] = {"stringFeature": {}, "doubleFeature": {}, "stringArrayFeature": {}}
        if market:
            feature["stringFeature"]["market"] = market
        if allowance_remaining is not None:
            feature["doubleFeature"]["allowance_remaining"] = float(allowance_remaining)
        if cart_ids:
            feature["stringArrayFeature"]["cart_items"] = {"values": list(cart_ids)}
        flt: Dict[str, Any] = {"stringArrayFeature": {}, "doubleFeature": {}}
        if categories:
            flt["stringArrayFeature"]["category"] = {"values": list(categories)}
        if brands:
            flt["stringArrayFeature"]["brand"] = {"values": list(brands)}
        if price_min is not None:
            flt["doubleFeature"]["price_min"] = float(price_min)
        if price_max is not None:
            flt["doubleFeature"]["price_max"] = float(price_max)
        body: Dict[str, Any] = {
            "user": {"uid": user_id},
            "context": {
                "spm": spm,
                "feature": feature,
                "filter": flt,
                "extra": {f"exp.{k}": v for k, v in (experiment or {}).items()},
            },
            "candidateItems": [{"id": i} for i in candidate_ids or ()],
            "filterItems": [{"id": i} for i in filter_ids or ()],
        }
        if parent_item_id:
            body["parentItem"] = {"id": parent_item_id}
        if size:
            body["size"] = int(size)
        return body

    @staticmethod
    def _parse_predict(r: Dict[str, Any]) -> PredictResult:
        if r.get("code") != 200:
            raise RecsysError(int(r.get("code", 0)), str(r.get("message", "")), r)
        items = [
            Item(
                id=str(x["id"]),
                rank=int(x.get("rank", n + 1)),
                trans_data=str(x.get("transData", "")),
                score=float(x["extra"]["score"]) if x.get("extra", {}).get("score") else None,
            )
            for n, x in enumerate(r.get("value", {}).get("items", []))
        ]
        return PredictResult(str(r.get("request_id", "")), items, list(r.get("degraded", [])))

    @staticmethod
    def _fallback(ids: Optional[Sequence[str]], size: Optional[int], err: str) -> PredictResult:
        ids = list(ids or [])[: size or None]
        return PredictResult("", [Item(i, n + 1) for n, i in enumerate(ids)], fallback=True, error=err)

    def _chunks(self, rows: Sequence[Dict[str, Any]]):
        for start in range(0, len(rows), self.chunk_size):
            yield start, list(rows[start : start + self.chunk_size])

    @staticmethod
    def _merge_write(total: WriteResult, start: int, r: Dict[str, Any]) -> None:
        total.accepted += int(r.get("accepted", 0))
        for e in r.get("errors", []):
            total.errors.append({**e, "index": start + int(e.get("index", 0))})
        if r.get("code") in (400,) and not r.get("errors"):
            raise RecsysError(400, str(r.get("message")), r)
        if r.get("code") in (401, 404):
            raise RecsysError(int(r["code"]), str(r.get("message")), r)

    @staticmethod
    def _retryable(e: Exception) -> bool:
        if isinstance(e, httpx.HTTPStatusError):
            return e.response.status_code >= 500
        return isinstance(e, httpx.TransportError)


class RecsysClient(_Base):
    """Blocking client. Pass `http` to reuse / inject an httpx.Client (e.g. for tests)."""

    def __init__(self, base_url: str, app_id: str, *, http: Optional[httpx.Client] = None, **kw: Any):
        super().__init__(base_url, app_id, **kw)
        self.http = http or httpx.Client(limits=httpx.Limits(max_keepalive_connections=50, max_connections=200))

    def __enter__(self) -> RecsysClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.http.close()

    def _post(self, path: str, obj: Any, timeout: float, retries: int) -> Dict[str, Any]:
        body = self._body(obj)
        last: Exception = RuntimeError("unreachable")
        for attempt in range(retries + 1):
            try:
                r = self.http.post(self.base_url + path, content=body, headers=self._headers(body), timeout=timeout)
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                last = e
                if not self._retryable(e) or attempt == retries:
                    break
                time.sleep(min(0.05 * 2**attempt, 1.0))
        raise last

    def predict(
        self,
        user_id: str,
        spm: str,
        size: Optional[int] = None,
        *,
        market: Optional[str] = None,
        parent_item_id: Optional[str] = None,
        candidate_ids: Optional[Sequence[str]] = None,
        filter_ids: Optional[Sequence[str]] = None,
        cart_ids: Optional[Sequence[str]] = None,
        allowance_remaining: Optional[float] = None,
        categories: Optional[Sequence[str]] = None,
        brands: Optional[Sequence[str]] = None,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        experiment: Optional[Dict[str, str]] = None,
        fallback_ids: Optional[Sequence[str]] = None,
        timeout: Optional[float] = None,
    ) -> PredictResult:
        body = self._predict_body(
            user_id, spm, size, market, parent_item_id, candidate_ids, filter_ids, cart_ids,
            allowance_remaining, categories, brands, price_min, price_max, experiment,
        )  # fmt: skip
        try:
            r = self._post(f"/predict/api/{self.app_id}/default", body, timeout or self.timeout, 0)
            return self._parse_predict(r)
        except (httpx.HTTPError, RecsysError, ValueError) as e:
            if fallback_ids is None and isinstance(e, RecsysError) and e.code in (401, 404):
                raise
            return self._fallback(fallback_ids, size, repr(e))

    def callback(self, user_id: str, request_id: str, item_ids: Sequence[str], spm: Optional[str] = None) -> None:
        body = {"uid": user_id, "request_id": request_id, "spm": spm, "items": [{"id": i} for i in item_ids]}
        self._post(f"/predict/api/{self.app_id}/callback", body, self.timeout, self.retries)

    def write(self, table: str, rows: Sequence[Dict[str, Any]]) -> WriteResult:
        total = WriteResult()
        for start, chunk in self._chunks(rows):
            r = self._post(f"/data/api/{self.app_id}/{table}/write", chunk, self.write_timeout, self.retries)
            self._merge_write(total, start, r)
        return total

    def write_behaviors(self, rows: Sequence[Dict[str, Any]]) -> WriteResult:
        return self.write("behavior", rows)

    def write_items(self, rows: Sequence[Dict[str, Any]]) -> WriteResult:
        return self.write("item", rows)

    def write_users(self, rows: Sequence[Dict[str, Any]]) -> WriteResult:
        return self.write("user", rows)

    def done(self, table: str, date: str) -> None:
        r = self._post(
            f"/data/api/{self.app_id}/done", {"table": table, "date": date}, self.write_timeout, self.retries
        )
        if r.get("code") != 200:
            raise RecsysError(int(r.get("code", 0)), str(r.get("message")), r)


class AsyncRecsysClient(_Base):
    """asyncio client with the same methods as RecsysClient (as coroutines)."""

    def __init__(self, base_url: str, app_id: str, *, http: Optional[httpx.AsyncClient] = None, **kw: Any):
        super().__init__(base_url, app_id, **kw)
        self.http = http or httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=50, max_connections=200))

    async def __aenter__(self) -> AsyncRecsysClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _post(self, path: str, obj: Any, timeout: float, retries: int) -> Dict[str, Any]:
        import asyncio

        body = self._body(obj)
        last: Exception = RuntimeError("unreachable")
        for attempt in range(retries + 1):
            try:
                r = await self.http.post(
                    self.base_url + path, content=body, headers=self._headers(body), timeout=timeout
                )
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                last = e
                if not self._retryable(e) or attempt == retries:
                    break
                await asyncio.sleep(min(0.05 * 2**attempt, 1.0))
        raise last

    async def predict(self, user_id: str, spm: str, size: Optional[int] = None, **kw: Any) -> PredictResult:
        fallback_ids = kw.pop("fallback_ids", None)
        timeout = kw.pop("timeout", None)
        args = [kw.pop(k, None) for k in (
            "market", "parent_item_id", "candidate_ids", "filter_ids", "cart_ids", "allowance_remaining",
            "categories", "brands", "price_min", "price_max", "experiment",
        )]  # fmt: skip
        if kw:
            raise TypeError(f"unexpected arguments {sorted(kw)}")
        body = self._predict_body(user_id, spm, size, *args)
        try:
            r = await self._post(f"/predict/api/{self.app_id}/default", body, timeout or self.timeout, 0)
            return self._parse_predict(r)
        except (httpx.HTTPError, RecsysError, ValueError) as e:
            if fallback_ids is None and isinstance(e, RecsysError) and e.code in (401, 404):
                raise
            return self._fallback(fallback_ids, size, repr(e))

    async def write(self, table: str, rows: Sequence[Dict[str, Any]]) -> WriteResult:
        total = WriteResult()
        for start, chunk in self._chunks(rows):
            r = await self._post(f"/data/api/{self.app_id}/{table}/write", chunk, self.write_timeout, self.retries)
            self._merge_write(total, start, r)
        return total

    async def done(self, table: str, date: str) -> None:
        r = await self._post(
            f"/data/api/{self.app_id}/done", {"table": table, "date": date}, self.write_timeout, self.retries
        )
        if r.get("code") != 200:
            raise RecsysError(int(r.get("code", 0)), str(r.get("message")), r)
