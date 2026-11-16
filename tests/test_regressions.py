"""Regression tests for JSON edge cases and semantic-rule boundaries."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft7Validator

from schema_review import load_schema, validate_review
from schema_review.validate import (
    DEFAULT_SCHEMA,
    _check,
    _contradicts_near_term,
    load_json,
)

ROOT = Path(__file__).resolve().parents[1]


class StructuralRegressionTests(unittest.TestCase):
    def test_boolean_is_not_an_integer(self) -> None:
        errors, _ = validate_review(True, {"type": "integer"}, structural_only=True)
        self.assertTrue(errors)

    def test_boolean_const_is_not_equal_to_integer_one(self) -> None:
        errors, _ = validate_review(1, {"const": True}, structural_only=True)
        self.assertTrue(errors)

    def test_boolean_enum_is_not_equal_to_integer_one(self) -> None:
        errors, _ = validate_review(1, {"enum": [True]}, structural_only=True)
        self.assertTrue(errors)

    def test_integral_float_matches_json_schema_integer_type(self) -> None:
        self.assertEqual(validate_review(1.0, {"type": "integer"}, True)[0], [])
        self.assertTrue(Draft7Validator({"type": "integer"}).is_valid(1.0))

    def test_null_type_is_supported(self) -> None:
        self.assertEqual(validate_review(None, {"type": "null"}, True)[0], [])
        self.assertTrue(validate_review("not-null", {"type": "null"}, True)[0])

    def test_large_integer_does_not_overflow_type_check(self) -> None:
        self.assertEqual(validate_review(10**1000, {"type": "integer"}, True)[0], [])

    def test_integer_and_integral_float_are_json_equal_for_const(self) -> None:
        errors, _ = validate_review(1.0, {"const": 1}, structural_only=True)
        self.assertEqual(errors, [])

    def test_unknown_schema_type_is_reported_not_raised(self) -> None:
        errors, _ = validate_review("x", {"type": "decimal"}, structural_only=True)
        self.assertTrue(any("unsupported type declaration" in error for error in errors))

    def test_explicit_empty_schema_is_not_replaced_by_default(self) -> None:
        errors, _ = validate_review({"anything": True}, {}, structural_only=True)
        self.assertEqual(errors, [])

    def test_additional_properties_schema_is_applied(self) -> None:
        schema = {
            "type": "object",
            "properties": {"fixed": {"type": "string"}},
            "additionalProperties": {"type": "integer"},
        }
        self.assertEqual(validate_review({"fixed": "ok", "extra": 3}, schema, True)[0], [])
        self.assertTrue(validate_review({"fixed": "ok", "extra": "bad"}, schema, True)[0])

    def test_json_pointer_escapes_are_resolved(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "value": {"$ref": "#/definitions/a~1b~0c"},
            },
            "definitions": {"a/b~c": {"type": "integer"}},
        }
        self.assertEqual(validate_review({"value": 4}, schema, True)[0], [])
        self.assertTrue(validate_review({"value": "four"}, schema, True)[0])

    def test_unresolved_reference_is_a_validation_error_not_traceback(self) -> None:
        errors, _ = validate_review({"value": 1}, {"properties": {"value": {"$ref": "#/missing"}}}, True)
        self.assertTrue(any("unresolved $ref" in error for error in errors))

    def test_bundled_schema_matches_draft7_oracle_for_fixture(self) -> None:
        schema = load_schema()
        fixture = json.loads((ROOT / "examples" / "review_example.json").read_text())
        self.assertTrue(Draft7Validator(schema).is_valid(fixture))
        self.assertEqual(validate_review(fixture, schema, structural_only=True)[0], [])

    def test_invalid_regex_is_reported(self) -> None:
        errors, _ = validate_review("value", {"type": "string", "pattern": "["}, True)
        self.assertTrue(any("invalid pattern" in error for error in errors))


class StrictJsonAndCliTests(unittest.TestCase):
    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                load_json(path)

    def test_nonfinite_json_numbers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nan.json"
            path.write_text('{"value": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-standard JSON"):
                load_json(path)

    def test_cli_duplicate_key_returns_input_error_without_traceback(self) -> None:
        env = {"PYTHONPATH": str(ROOT / "src")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"known": [], "known": []}', encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "schema_review", str(path)],
                cwd=ROOT,
                env={**__import__("os").environ, **env},
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_invalid_schema_returns_input_error_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            schema = Path(directory) / "bad-schema.json"
            schema.write_text('{"type": "object", "properties": []}', encoding="utf-8")
            fixture = ROOT / "examples" / "review_example.json"
            result = subprocess.run(
                [sys.executable, "-m", "schema_review", str(fixture), "--schema", str(schema)],
                cwd=ROOT,
                env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)


class SemanticRegressionTests(unittest.TestCase):
    def test_negated_out_of_scope_phrase_is_not_a_contradiction(self) -> None:
        self.assertFalse(
            _contradicts_near_term(
                {"float16": "near-term-in-scope"},
                "float16 is not out of scope; it remains a candidate.",
            )
        )

    def test_out_of_scope_for_different_type_is_not_a_contradiction(self) -> None:
        self.assertFalse(
            _contradicts_near_term(
                {"float16": "near-term-in-scope"},
                "Ternary is out of scope, while float16 is the first candidate.",
            )
        )

    def test_explicit_type_specific_contradiction_is_detected(self) -> None:
        self.assertTrue(
            _contradicts_near_term(
                {"float16": "near-term-in-scope"},
                "Float16 is currently out of scope.",
            )
        )

    def test_global_quantized_type_contradiction_is_detected(self) -> None:
        self.assertTrue(
            _contradicts_near_term(
                {"float16": "near-term-in-scope"},
                "All quantized types are out of scope.",
            )
        )


if __name__ == "__main__":
    unittest.main()
