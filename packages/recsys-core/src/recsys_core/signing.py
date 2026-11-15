"""Request signing shared by the server and the SDKs.

signature = hex(SHA256(token || body || tenant_id || ts || nonce))
(same construction as Volcengine/BytePlus predict, so existing clients keep working)."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time


def sign(token: str, body: bytes, tenant_id: str, ts: str, nonce: str) -> str:
    h = hashlib.sha256()
    for part in (token.encode(), body, tenant_id.encode(), ts.encode(), nonce.encode()):
        h.update(part)
    return h.hexdigest()


def headers(token: str, tenant_id: str, body: bytes) -> dict[str, str]:
    """Client helper: headers for a signed request."""
    ts, nonce = str(int(time.time())), secrets.token_hex(8)
    return {
        "Tenant-Id": tenant_id,
        "Tenant-Ts": ts,
        "Tenant-Nonce": nonce,
        "Tenant-Signature": sign(token, body, tenant_id, ts, nonce),
        "Content-Type": "application/json",
    }


def verify(token: str, body: bytes, tenant_id: str, ts: str, nonce: str, signature: str) -> bool:
    return hmac.compare_digest(sign(token, body, tenant_id, ts, nonce), signature)
