"""Dependency-free validation: a draft-07 subset checker plus semantic rules.

The structural checker supports only the keywords used by the bundled schema
(type, required, properties, additionalProperties, items, enum, const,
minItems, minLength, $ref to #/definitions). It is not a general JSON Schema
implementation; use the `jsonschema` package for that.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_SCHEMA = (
    Path(__file__).resolve().parents[2] / "schemas" / "quantized_emitter_review.schema.json"
)
_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "integer": int, "number": (int, float),
}
KNOWN = {"known-source", "known-measured", "known-documented"}


def load_schema(path: str | Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or DEFAULT_SCHEMA).read_text(encoding="utf-8"))


def _resolve(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported $ref: {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _check(inst: Any, sch: dict[str, Any], root: dict[str, Any], path: str, errs: list[str]) -> None:
    if "$ref" in sch:
        sch = _resolve(sch["$ref"], root)
    t = sch.get("type")
    if t:
        ok = isinstance(inst, _TYPES[t])
        if t in ("integer", "number") and isinstance(inst, bool):
            ok = False
        if not ok:
            errs.append(f"{path}: expected {t}")
            return
    if "const" in sch and inst != sch["const"]:
        errs.append(f"{path}: must equal {sch['const']!r}")
    if "enum" in sch and inst not in sch["enum"]:
        errs.append(f"{path}: {inst!r} not in {sch['enum']}")
    if isinstance(inst, str) and len(inst) < sch.get("minLength", 0):
        errs.append(f"{path}: shorter than minLength {sch['minLength']}")
    if isinstance(inst, list):
        if len(inst) < sch.get("minItems", 0):
            errs.append(f"{path}: fewer than {sch['minItems']} items")
        if "items" in sch:
            for i, item in enumerate(inst):
                _check(item, sch["items"], root, f"{path}[{i}]", errs)
    if isinstance(inst, dict):
        props = sch.get("properties", {})
        for key in sch.get("required", []):
            if key not in inst:
                errs.append(f"{path}: missing required '{key}'")
        if sch.get("additionalProperties") is False:
            for key in inst:
                if key not in props:
                    errs.append(f"{path}: unexpected property '{key}'")
        for key, sub in props.items():
            if key in inst:
                _check(inst[key], sub, root, f"{path}.{key}", errs)


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def semantic_errors(doc: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for rules a JSON Schema cannot express."""
    errs: list[str] = []
    warns: list[str] = []

    for i, c in enumerate(doc.get("known", [])):
        cls = str(c.get("evidence_class", ""))
        if cls not in KNOWN:
            errs.append(f"known[{i}]: section 'known' may only hold known-* claims")
        elif not c.get("source"):
            errs.append(f"known[{i}]: known-* claim requires a source")
    for i, c in enumerate(doc.get("proposed_unmeasured", {}).get("claims", [])):
        if c.get("evidence_class") != "proposed-unmeasured":
            errs.append(f"proposed_unmeasured.claims[{i}]: must be proposed-unmeasured")

    verdict = doc.get("scope_verdict", {})
    blocker = verdict.get("primary_blocker", {})
    if blocker.get("evidence_class") in KNOWN and not blocker.get("source"):
        errs.append("scope_verdict.primary_blocker: known-* claim requires a source")
    for typ, v in verdict.get("per_type", {}).items():
        if v == "near-term-in-scope" and blocker.get("evidence_class") not in KNOWN:
            errs.append(f"scope_verdict.{typ}: near-term-in-scope needs a known-* primary blocker assessment")

    if not doc.get("verification_limits"):
        errs.append("verification_limits: must not be empty")

    n = _sentences(doc.get("bottom_line", ""))
    if not 2 <= n <= 4:
        errs.append(f"bottom_line: expected 2-4 sentences, found {n}")

    for i, c in enumerate(doc.get("required_changes", {}).get("type_system", [])):
        if c.get("extensibility_verified"):
            warns.append(f"required_changes.type_system[{i}]: extensibility_verified=true; confirm the material shows this")
    return errs, warns


def validate_review(doc: dict[str, Any], schema: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    schema = schema or load_schema()
    errs: list[str] = []
    _check(doc, schema, schema, "$", errs)
    if errs:
        return errs, []
    return semantic_errors(doc)
