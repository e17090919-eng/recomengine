"""Layered A/B bucketing: hash(uid + layer salt) % 100. Layers are orthogonal;
variants inside a layer are mutually exclusive."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class Assignment:
    layer: str
    variant: str
    params: dict


def bucket(uid: str, salt: str, n: int = 100) -> int:
    return int.from_bytes(hashlib.sha1(f"{salt}:{uid}".encode()).digest()[:8], "big") % n


def assign(experiments: dict, uid: str, slot: str, force: dict[str, str] | None = None) -> list[Assignment]:
    out = []
    for layer, exp in (experiments or {}).items():
        if slot not in exp.get("slots", []):
            continue
        variants = exp["variants"]
        chosen = None
        if force and layer in force:
            chosen = next((v for v in variants if v["name"] == force[layer]), None)
        if chosen is None:
            b, acc = bucket(uid, exp.get("salt", layer)), 0
            for v in variants:
                acc += v["traffic"]
                if b < acc:
                    chosen = v
                    break
        if chosen is not None:
            out.append(Assignment(layer, chosen["name"], dict(chosen.get("params", {}))))
    return out
