"""Dependency-free validation: a draft-07 subset checker plus semantic rules.

The structural checker supports only the keywords used by the bundled schema
(type, required, properties, additionalProperties, items, enum, const,
minItems, minLength, pattern, $ref to #/definitions). It is not a general JSON
Schema implementation; use the `jsonschema` package for that.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_SCHEMA = Path(__file__).resolve().parent / "data" / "quantized_emitter_review.schema.json"
_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "integer": int, "number": (int, float),
}
KNOWN = {"known-source", "known-measured", "known-documented"}
BLOCKER_OK = KNOWN | {"unverified", "assumption"}
_ABBR = ("e.g.", "i.e.", "etc.", "vs.", "cf.", "Fig.", "Sec.", "No.")
CHANGE_LISTS = ("type_system", "emitter_lowering", "operators", "validation")


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
        sch = {**_resolve(sch["$ref"], root), **{k: v for k, v in sch.items() if k != "$ref"}}
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
    if isinstance(inst, str):
        if len(inst) < sch.get("minLength", 0):
            errs.append(f"{path}: shorter than minLength {sch['minLength']}")
        if "pattern" in sch and not re.search(sch["pattern"], inst):
            errs.append(f"{path}: does not match pattern {sch['pattern']!r} (blank?)")
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
    t = text.strip()
    for a in _ABBR:
        t = t.replace(a, a.replace(".", "\x00"))
    return len([s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()])


def _dupes(label: str, texts: list[str], errs: list[str]) -> None:
    seen: set[str] = set()
    for i, raw in enumerate(texts):
        key = " ".join(raw.lower().split())
        if key in seen:
            errs.append(f"{label}[{i}]: duplicate of an earlier entry")
        seen.add(key)


def semantic_errors(doc: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for rules a JSON Schema cannot express."""
    errs: list[str] = []
    warns: list[str] = []
    known = doc.get("known", [])
    proposed = doc.get("proposed_unmeasured", {}).get("claims", [])

    for i, c in enumerate(known):
        cls = str(c.get("evidence_class", ""))
        if cls not in KNOWN:
            errs.append(f"known[{i}]: section 'known' may only hold known-* claims")
        elif not c.get("source"):
            errs.append(f"known[{i}]: known-* claim requires a source")
    for i, c in enumerate(proposed):
        if c.get("evidence_class") != "proposed-unmeasured":
            errs.append(f"proposed_unmeasured.claims[{i}]: must be proposed-unmeasured")

    verdict = doc.get("scope_verdict", {})
    blocker = verdict.get("primary_blocker", {})
    bclass = blocker.get("evidence_class")
    if bclass not in BLOCKER_OK:
        errs.append(f"scope_verdict.primary_blocker: evidence_class must be one of {sorted(BLOCKER_OK)}, not {bclass!r}")
    if bclass in KNOWN and not blocker.get("source"):
        errs.append("scope_verdict.primary_blocker: known-* claim requires a source")
    per_type = verdict.get("per_type", {})
    for typ, v in per_type.items():
        if v == "near-term-in-scope" and bclass not in KNOWN:
            errs.append(f"scope_verdict.{typ}: near-term-in-scope needs a known-* primary blocker assessment")
    text = (verdict.get("summary", "") + " " + doc.get("bottom_line", "")).lower()
    if "near-term-in-scope" in per_type.values() and ("out of scope" in text or "out-of-scope" in text):
        errs.append("scope_verdict: a type is near-term-in-scope but the summary or bottom line says 'out of scope'")

    n = _sentences(doc.get("bottom_line", ""))
    if not 2 <= n <= 4:
        errs.append(f"bottom_line: expected 2-4 sentences, found {n}")

    _dupes("known", [c.get("statement", "") for c in known], errs)
    _dupes("proposed_unmeasured.claims", [c.get("statement", "") for c in proposed], errs)
    _dupes("open_design_questions", [q.get("question", "") for q in doc.get("open_design_questions", [])], errs)
    changes = doc.get("required_changes", {})
    for name in CHANGE_LISTS:
        items = changes.get(name, [])
        _dupes(f"required_changes.{name}", [c.get("statement", "") for c in items], errs)
        for i, c in enumerate(items):
            if c.get("extensibility_verified"):
                warns.append(f"required_changes.{name}[{i}]: extensibility_verified=true; confirm the material shows this")
    return errs, warns


def validate_review(
    doc: dict[str, Any],
    schema: dict[str, Any] | None = None,
    structural_only: bool = False,
) -> tuple[list[str], list[str]]:
    """Validate doc. With structural_only, apply only the schema checks (use this
    for any schema other than the bundled one; the semantic rules assume its fields)."""
    schema = schema or load_schema()
    errs: list[str] = []
    _check(doc, schema, schema, "$", errs)
    if errs or structural_only:
        return errs, []
    return semantic_errors(doc)
