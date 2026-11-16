"""Dependency-free validation: a documented JSON Schema subset plus semantic rules.

The structural checker supports the keywords used by the bundled draft-07
schema: type, required, properties, additionalProperties, items, enum, const,
minItems, minLength, pattern, and local $ref references. It is not a general
JSON Schema implementation; use the optional test extra for differential
checks against the jsonschema package.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

DEFAULT_SCHEMA = Path(__file__).resolve().parent / "data" / "quantized_emitter_review.schema.json"
_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
}
KNOWN = {"known-source", "known-measured", "known-documented"}
BLOCKER_OK = KNOWN | {"unverified", "assumption"}
_ABBR = (
    "e.g.", "i.e.", "etc.", "vs.", "cf.", "Fig.", "Sec.", "No.",
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "U.S.", "U.K.",
)
CHANGE_LISTS = ("type_system", "emitter_lowering", "operators", "validation")


class SchemaDefinitionError(ValueError):
    """The supplied schema uses an invalid or unsupported schema construct."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def load_json(path: str | Path) -> Any:
    """Read strict JSON (reject duplicate keys and NaN/Infinity)."""
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonfinite,
    )


def load_schema(path: str | Path | None = None) -> dict[str, Any]:
    schema = load_json(path or DEFAULT_SCHEMA)
    if not isinstance(schema, dict):
        raise SchemaDefinitionError("schema root must be a JSON object")
    return schema


def _resolve(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise SchemaDefinitionError(f"unsupported $ref: {ref!r}")
    node: Any = root
    for raw_part in ref[2:].split("/"):
        if re.search(r"~(?![01])", raw_part):
            raise SchemaDefinitionError(f"invalid JSON Pointer escape in $ref: {ref!r}")
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise SchemaDefinitionError(f"unresolved $ref: {ref!r}")
        node = node[part]
    if not isinstance(node, dict):
        raise SchemaDefinitionError(f"$ref target must be a schema object: {ref!r}")
    return node


def _json_equal(left: Any, right: Any) -> bool:
    """JSON equality: booleans are not numbers, unlike Python's bool/int equality."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return (
            left.keys() == right.keys()
            and all(_json_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _matches_type(instance: Any, expected: str) -> bool:
    python_type = _TYPES[expected]
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "integer":
        if isinstance(instance, bool):
            return False
        if isinstance(instance, int):
            return True
        return isinstance(instance, float) and math.isfinite(instance) and instance.is_integer()
    return isinstance(instance, python_type)


def _check(
    inst: Any,
    sch: dict[str, Any],
    root: dict[str, Any],
    path: str,
    errs: list[str],
) -> None:
    if not isinstance(sch, dict):
        errs.append(f"{path}: schema node must be an object")
        return

    # Draft-07 says sibling keywords beside $ref are ignored.
    if "$ref" in sch:
        try:
            sch = _resolve(sch["$ref"], root)
        except SchemaDefinitionError as exc:
            errs.append(f"{path}: invalid schema: {exc}")
            return

    expected = sch.get("type")
    if expected is not None:
        allowed = [expected] if isinstance(expected, str) else expected
        if not isinstance(allowed, list) or not allowed or any(
            not isinstance(item, str) or item not in _TYPES for item in allowed
        ):
            errs.append(f"{path}: schema has unsupported type declaration {expected!r}")
            return
        if not any(_matches_type(inst, item) for item in allowed):
            errs.append(f"{path}: expected {' or '.join(allowed)}")
            return

    if "const" in sch and not _json_equal(inst, sch["const"]):
        errs.append(f"{path}: must equal {sch['const']!r}")
    if "enum" in sch:
        values = sch["enum"]
        if not isinstance(values, list) or not any(
            _json_equal(inst, candidate) for candidate in values
        ):
            errs.append(f"{path}: {inst!r} not in {values!r}")

    if isinstance(inst, str):
        minimum = sch.get("minLength", 0)
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
            errs.append(f"{path}: schema has invalid minLength {minimum!r}")
        elif len(inst) < minimum:
            errs.append(f"{path}: shorter than minLength {minimum}")
        if "pattern" in sch:
            pattern = sch["pattern"]
            if not isinstance(pattern, str):
                errs.append(f"{path}: schema pattern must be a string")
            else:
                try:
                    matched = re.search(pattern, inst)
                except re.error as exc:
                    errs.append(f"{path}: schema has invalid pattern {pattern!r}: {exc}")
                else:
                    if matched is None:
                        errs.append(f"{path}: does not match pattern {pattern!r}")

    if isinstance(inst, list):
        minimum = sch.get("minItems", 0)
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
            errs.append(f"{path}: schema has invalid minItems {minimum!r}")
        elif len(inst) < minimum:
            errs.append(f"{path}: fewer than minItems {minimum}")
        item_schema = sch.get("items")
        if item_schema is not None:
            for index, item in enumerate(inst):
                _check(item, item_schema, root, f"{path}[{index}]", errs)

    if isinstance(inst, dict):
        required = sch.get("required", [])
        if not isinstance(required, list) or any(not isinstance(k, str) for k in required):
            errs.append(f"{path}: schema required must be an array of strings")
        else:
            for key in required:
                if key not in inst:
                    errs.append(f"{path}: missing required {key!r}")

        props = sch.get("properties", {})
        if not isinstance(props, dict):
            errs.append(f"{path}: schema properties must be an object")
            return
        additional = sch.get("additionalProperties", True)
        for key, value in inst.items():
            if key in props:
                _check(value, props[key], root, f"{path}.{key}", errs)
            elif additional is False:
                errs.append(f"{path}: unexpected property {key!r}")
            elif isinstance(additional, dict):
                _check(value, additional, root, f"{path}.{key}", errs)
            elif additional is not True:
                errs.append(f"{path}: unsupported additionalProperties value {additional!r}")


def _sentences(text: str) -> int:
    value = text.strip()
    for abbreviation in _ABBR:
        value = value.replace(abbreviation, abbreviation.replace(".", "\x00"))
    return len([part for part in re.split(r"(?<=[.!?])\s+", value) if part.strip()])


def _dupes(label: str, texts: list[str], errs: list[str]) -> None:
    seen: set[str] = set()
    for index, raw in enumerate(texts):
        key = " ".join(raw.lower().split())
        if key in seen:
            errs.append(f"{label}[{index}]: duplicate of an earlier entry")
        seen.add(key)


def _contradicts_near_term(per_type: dict[str, Any], text: str) -> bool:
    """Detect direct type/scope associations while respecting explicit negation."""
    text = text.lower()
    for type_name, verdict in per_type.items():
        if verdict != "near-term-in-scope":
            continue
        escaped = re.escape(type_name)
        type_then_scope = re.search(
            rf"\b{escaped}\b(?:\s+\w+){{0,5}}\s+out[- ]of[- ]scope\b",
            text,
        )
        if type_then_scope and not re.search(r"\bnot\b", type_then_scope.group(0)):
            return True

        scope_then_type = re.search(
            rf"\bout[- ]of[- ]scope\s+(?:for|in)\s+(?:the\s+)?\b{escaped}\b",
            text,
        )
        if scope_then_type:
            prefix = text[max(0, scope_then_type.start() - 32):scope_then_type.start()]
            if not re.search(r"\bnot(?:\s+\w+){0,2}\s+$", prefix):
                return True

        global_scope = re.search(
            r"\b(?:all(?:\s+three)?|quantized)\s+types?\b"
            r"(?:\s+\w+){0,5}\s+out[- ]of[- ]scope\b",
            text,
        )
        if global_scope and not re.search(r"\bnot\b", global_scope.group(0)):
            return True
    return False

def semantic_errors(doc: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for rules that are not expressible in the schema."""
    errs: list[str] = []
    warns: list[str] = []
    known = doc.get("known", [])
    proposed = doc.get("proposed_unmeasured", {}).get("claims", [])

    for index, claim in enumerate(known):
        evidence_class = claim.get("evidence_class", "")
        if evidence_class not in KNOWN:
            errs.append("known[{}]: section 'known' may only hold known-* claims".format(index))
        elif not claim.get("source"):
            errs.append(f"known[{index}]: known-* claim requires a source")

    for index, claim in enumerate(proposed):
        if claim.get("evidence_class") != "proposed-unmeasured":
            errs.append(
                f"proposed_unmeasured.claims[{index}]: must be proposed-unmeasured"
            )

    verdict = doc.get("scope_verdict", {})
    blocker = verdict.get("primary_blocker", {})
    blocker_class = blocker.get("evidence_class")
    if blocker_class not in BLOCKER_OK:
        errs.append(
            "scope_verdict.primary_blocker: evidence_class must be one of "
            f"{sorted(BLOCKER_OK)}, not {blocker_class!r}"
        )
    if blocker_class in KNOWN and not blocker.get("source"):
        errs.append("scope_verdict.primary_blocker: known-* claim requires a source")

    per_type = verdict.get("per_type", {})
    for type_name, scope in per_type.items():
        if scope == "near-term-in-scope" and blocker_class not in KNOWN:
            errs.append(
                f"scope_verdict.{type_name}: near-term-in-scope needs a known-* "
                "primary blocker assessment"
            )

    verdict_text = (verdict.get("summary", "") + " " + doc.get("bottom_line", "")).lower()
    if _contradicts_near_term(per_type, verdict_text):
        errs.append(
            "scope_verdict: a near-term-in-scope type is described as saying "
            "'out of scope' in the summary or bottom line"
        )

    sentence_count = _sentences(doc.get("bottom_line", ""))
    if not 2 <= sentence_count <= 4:
        errs.append(f"bottom_line: expected 2-4 sentences, found {sentence_count}")

    _dupes("known", [claim.get("statement", "") for claim in known], errs)
    _dupes(
        "proposed_unmeasured.claims",
        [claim.get("statement", "") for claim in proposed],
        errs,
    )
    _dupes(
        "open_design_questions",
        [question.get("question", "") for question in doc.get("open_design_questions", [])],
        errs,
    )
    changes = doc.get("required_changes", {})
    for name in CHANGE_LISTS:
        items = changes.get(name, [])
        _dupes(
            f"required_changes.{name}",
            [change.get("statement", "") for change in items],
            errs,
        )
        for index, change in enumerate(items):
            if change.get("extensibility_verified"):
                warns.append(
                    f"required_changes.{name}[{index}]: extensibility_verified=true; "
                    "confirm the material shows this"
                )
    return errs, warns


def validate_review(
    doc: dict[str, Any],
    schema: dict[str, Any] | None = None,
    structural_only: bool = False,
) -> tuple[list[str], list[str]]:
    """Validate a review; structural_only skips bundled-schema semantic rules."""
    if schema is None:
        schema = load_schema()
    errs: list[str] = []
    _check(doc, schema, schema, "$", errs)
    if errs or structural_only:
        return errs, []
    return semantic_errors(doc)
