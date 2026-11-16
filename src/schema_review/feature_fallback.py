"""Deterministic capability and fallback decisions for ONNX-to-Cmajor features.

This module classifies evidence supplied by an adapter probe. It does not probe
ONNX or Cmajor itself and never performs a conversion. A returned fallback
decision is permission to execute a separately implemented, validated adapter
path—not proof that the conversion code exists.
"""
from __future__ import annotations

import math
from typing import Any

FEATURES = ("ternary", "int8", "float16")
NATIVE_GATES = (
    "dtype_recognized",
    "initializer_decoded",
    "representation_supported",
    "emitter_lowering_supported",
    "operator_coverage_verified",
    "codegen_compiles",
    "runtime_parity_verified",
)
FALLBACK_GATES = (
    "dtype_recognized",
    "initializer_decoded",
    "operator_coverage_verified",
    "codegen_compiles",
    "runtime_parity_verified",
)
FALLBACK_TRANSFORMS = {
    "ternary": "decode_ternary_to_float32",
    "int8": "dequantize_int8_to_float32",
    "float16": "promote_float16_to_float32",
}


def _evidence_complete(evidence: Any, required: tuple[str, ...]) -> bool:
    return (
        isinstance(evidence, dict)
        and all(isinstance(evidence.get(key), str) and evidence[key].strip() for key in required)
    )


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def classify_feature(feature: str, report: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic native/fallback/blocked/unsupported decision."""
    if feature not in FEATURES:
        raise ValueError(f"unsupported feature name: {feature!r}")
    if not isinstance(report, dict):
        raise TypeError("report must be an object")

    capabilities = report.get("capabilities")
    evidence = report.get("evidence")
    if not isinstance(capabilities, dict):
        capabilities = {}
    if not isinstance(evidence, dict):
        evidence = {}

    missing_native = [gate for gate in NATIVE_GATES if capabilities.get(gate) is not True]
    native_evidence_ok = _evidence_complete(evidence, NATIVE_GATES)
    if not missing_native and native_evidence_ok:
        return {
            "feature": feature,
            "decision": "native",
            "path": "native",
            "reason_codes": [],
            "required_action": "emit_native",
        }

    fallback = report.get("fallback")
    if isinstance(fallback, dict) and fallback.get("enabled") is True:
        expected_transform = FALLBACK_TRANSFORMS[feature]
        transform_ok = fallback.get("transform") == expected_transform
        ternary_encoding_ok = (
            feature != "ternary" or fallback.get("ternary_encoding_verified") is True
        )
        fallback_evidence = fallback.get("evidence")
        fallback_evidence_ok = (
            isinstance(fallback_evidence, list)
            and bool(fallback_evidence)
            and all(isinstance(item, str) and item.strip() for item in fallback_evidence)
        )
        error_ok = (
            _finite_nonnegative(fallback.get("error_tolerance"))
            and _finite_nonnegative(fallback.get("measured_error"))
            and fallback["measured_error"] <= fallback["error_tolerance"]
        )
        fallback_gates_ok = (
            all(capabilities.get(gate) is True for gate in FALLBACK_GATES)
            and _evidence_complete(evidence, FALLBACK_GATES)
        )
        guards_ok = (
            transform_ok
            and ternary_encoding_ok
            and fallback.get("metadata_preserved") is True
            and fallback.get("validation_passed") is True
            and fallback_gates_ok
            and fallback_evidence_ok
            and error_ok
        )
        if guards_ok:
            return {
                "feature": feature,
                "decision": "fallback",
                "path": expected_transform,
                "reason_codes": ["native_path_unavailable"],
                "required_action": "emit_via_validated_fallback",
            }

    unknown = [
        gate for gate in NATIVE_GATES
        if gate not in capabilities or capabilities.get(gate) is None
    ]
    if unknown or not native_evidence_ok:
        reasons = []
        if unknown:
            reasons.append("capability_unknown:" + ",".join(unknown))
        if not native_evidence_ok:
            reasons.append("native_evidence_incomplete")
        return {
            "feature": feature,
            "decision": "blocked",
            "path": None,
            "reason_codes": reasons,
            "required_action": "collect_capability_evidence",
        }

    false_gates = [gate for gate in NATIVE_GATES if capabilities.get(gate) is False]
    return {
        "feature": feature,
        "decision": "unsupported",
        "path": None,
        "reason_codes": ["native_capability_false:" + ",".join(false_gates)],
        "required_action": "reject_or_add_validated_fallback",
    }


def classify_model(report: dict[str, Any]) -> dict[str, Any]:
    """Classify each requested feature independently; readiness is all-or-nothing."""
    if not isinstance(report, dict):
        raise TypeError("report must be an object")
    requested = report.get("requested_features")
    if not isinstance(requested, list) or not requested:
        raise ValueError("requested_features must be a non-empty list")
    if any(not isinstance(feature, str) for feature in requested):
        raise ValueError("requested_features entries must be strings")
    if len(set(requested)) != len(requested):
        raise ValueError("requested_features contains duplicates")
    feature_reports = report.get("features")
    if not isinstance(feature_reports, dict):
        raise ValueError("features must be an object")
    results = {}
    for feature in requested:
        if feature not in FEATURES:
            raise ValueError(f"unsupported feature name: {feature!r}")
        results[feature] = classify_feature(feature, feature_reports.get(feature, {}))
    return {
        "results": results,
        "model_decision": (
            "ready" if all(item["decision"] in {"native", "fallback"} for item in results.values())
            else "blocked"
        ),
    }
