"""Tests for the ONNX-to-Cmajor feature/fallback decision model."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft7Validator

from schema_review.feature_fallback import (
    FALLBACK_TRANSFORMS,
    NATIVE_GATES,
    classify_feature,
    classify_model,
)

ROOT = Path(__file__).resolve().parents[1]


def native_report():
    return {
        "capabilities": {gate: True for gate in NATIVE_GATES},
        "evidence": {gate: f"artifact:{gate}" for gate in NATIVE_GATES},
    }


def fallback_report(feature):
    report = native_report()
    report["capabilities"]["representation_supported"] = False
    report["capabilities"]["emitter_lowering_supported"] = False
    report["fallback"] = {
        "enabled": True,
        "transform": FALLBACK_TRANSFORMS[feature],
        "ternary_encoding_verified": feature == "ternary",
        "metadata_preserved": True,
        "validation_passed": True,
        "error_tolerance": 1e-5,
        "measured_error": 1e-6,
        "evidence": ["test-run:123", "model-hash:abc"],
    }
    return report


class FeatureFallbackTests(unittest.TestCase):
    def test_native_requires_all_gates_and_evidence(self):
        self.assertEqual(classify_feature("int8", native_report())["decision"], "native")

    def test_missing_capability_is_blocked_not_assumed_false(self):
        report = native_report()
        del report["capabilities"]["codegen_compiles"]
        result = classify_feature("float16", report)
        self.assertEqual(result["decision"], "blocked")
        self.assertIn("capability_unknown:codegen_compiles", result["reason_codes"])

    def test_true_capability_without_evidence_is_blocked(self):
        report = native_report()
        del report["evidence"]["runtime_parity_verified"]
        self.assertEqual(classify_feature("int8", report)["decision"], "blocked")

    def test_explicit_false_without_fallback_is_unsupported(self):
        report = native_report()
        report["capabilities"]["emitter_lowering_supported"] = False
        result = classify_feature("float16", report)
        self.assertEqual(result["decision"], "unsupported")

    def test_int8_fallback_requires_correct_transform(self):
        report = fallback_report("int8")
        self.assertEqual(classify_feature("int8", report)["decision"], "fallback")
        report["fallback"]["transform"] = "promote_float16_to_float32"
        self.assertEqual(classify_feature("int8", report)["decision"], "unsupported")

    def test_float16_promotion_fallback(self):
        self.assertEqual(classify_feature("float16", fallback_report("float16"))["decision"], "fallback")

    def test_ternary_fallback_requires_verified_encoding(self):
        report = fallback_report("ternary")
        self.assertEqual(classify_feature("ternary", report)["decision"], "fallback")
        report["fallback"]["ternary_encoding_verified"] = False
        self.assertEqual(classify_feature("ternary", report)["decision"], "unsupported")

    def test_fallback_rejects_lost_quantization_metadata(self):
        report = fallback_report("int8")
        report["fallback"]["metadata_preserved"] = False
        self.assertEqual(classify_feature("int8", report)["decision"], "unsupported")

    def test_fallback_rejects_failed_numerical_validation(self):
        report = fallback_report("int8")
        report["fallback"]["validation_passed"] = False
        self.assertEqual(classify_feature("int8", report)["decision"], "unsupported")

    def test_fallback_rejects_error_above_tolerance(self):
        report = fallback_report("int8")
        report["fallback"]["measured_error"] = 1e-3
        self.assertEqual(classify_feature("int8", report)["decision"], "unsupported")

    def test_fallback_requires_operator_and_runtime_gates(self):
        report = fallback_report("float16")
        report["capabilities"]["runtime_parity_verified"] = False
        self.assertEqual(classify_feature("float16", report)["decision"], "unsupported")

    def test_fallback_requires_evidence_for_shared_gates(self):
        report = fallback_report("int8")
        del report["evidence"]["operator_coverage_verified"]
        self.assertEqual(classify_feature("int8", report)["decision"], "blocked")

    def test_model_readiness_is_all_requested_features(self):
        report = {
            "requested_features": ["int8", "float16"],
            "features": {
                "int8": native_report(),
                "float16": fallback_report("float16"),
            },
        }
        self.assertEqual(classify_model(report)["model_decision"], "ready")
        report["features"]["float16"]["fallback"]["validation_passed"] = False
        self.assertEqual(classify_model(report)["model_decision"], "blocked")

    def test_duplicate_requested_features_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            classify_model({"requested_features": ["int8", "int8"], "features": {}})

    def test_non_string_requested_feature_rejected_cleanly(self):
        with self.assertRaisesRegex(ValueError, "entries must be strings"):
            classify_model({"requested_features": [{"dtype": "int8"}], "features": {}})

    def test_unknown_feature_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported feature"):
            classify_feature("int4", native_report())

    def test_report_schema_accepts_valid_fallback_and_rejects_bad_shape(self):
        schema = json.loads((ROOT / "src/schema_review/data/feature_fallback.schema.json").read_text())
        valid = {
            "requested_features": ["int8"],
            "features": {"int8": fallback_report("int8")},
        }
        validator = Draft7Validator(schema)
        self.assertTrue(validator.is_valid(valid))
        invalid = {"requested_features": ["int4"], "features": {}}
        self.assertFalse(validator.is_valid(invalid))


if __name__ == "__main__":
    unittest.main()
