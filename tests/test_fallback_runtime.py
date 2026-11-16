"""Independent execution and validation tests for fallback transforms."""
from __future__ import annotations

import unittest

from schema_review.fallback_runtime import (
    FallbackError,
    decode_ternary,
    dequantize_int8,
    execute_and_validate,
    promote_float16,
)


class FallbackRuntimeTests(unittest.TestCase):
    def test_float16_executes_and_matches_independent_oracle(self):
        report = execute_and_validate(
            "float16", {"words": [0x3C00, 0xC000, 0x0001], "shape": [3]},
            {"source_dtype": "FLOAT16", "initializer_name": "w"}, 0.0,
        )
        self.assertTrue(report["validation_passed"])
        self.assertEqual(report["output"], [1.0, -2.0, 2 ** -24])
        self.assertIn("oracle:independent-stdlib-reference-v1", report["evidence"])

    def test_float16_rejects_nonfinite(self):
        with self.assertRaises(FallbackError):
            promote_float16([0x7C00], [1])

    def test_float16_rejects_bad_payload_length(self):
        with self.assertRaises(FallbackError):
            promote_float16([0x3C00], [2])

    def test_int8_per_tensor_dequantization(self):
        report = execute_and_validate(
            "int8", {"values": [-128, 0, 127], "shape": [3], "scale": 0.5, "zero_point": -1},
            {"scale": 0.5, "zero_point": -1}, 0.0,
        )
        self.assertTrue(report["validation_passed"])
        self.assertEqual(report["output"], [-63.5, 0.5, 64.0])

    def test_int8_per_axis_dequantization(self):
        report = execute_and_validate(
            "int8", {"values": [1, 2, 3, 4], "shape": [2, 2],
                     "scale": [0.5, 2.0], "zero_point": [0, 1], "axis": 0},
            {"axis": 0, "scale": [0.5, 2.0], "zero_point": [0, 1]}, 0.0,
        )
        self.assertTrue(report["validation_passed"])
        self.assertEqual(report["output"], [0.5, 1.0, 4.0, 6.0])

    def test_int8_rejects_zero_scale(self):
        with self.assertRaises(FallbackError):
            dequantize_int8([1], [1], scale=0.0)

    def test_int8_rejects_invalid_axis_parameter_shape(self):
        with self.assertRaises(FallbackError):
            dequantize_int8([1, 2, 3, 4], [2, 2], scale=[0.5, 1.0, 2.0], axis=0)

    def test_ternary_lsb_decode(self):
        # Symbols 0,1,2 encoded in the low six bits; upper padding is zero.
        report = execute_and_validate(
            "ternary", {"packed": [0b00100100], "values_count": 3, "shape": [3],
                        "codebook": {0: -1.0, 1: 0.0, 2: 1.0}, "bit_order": "lsb"},
            {"encoding": "2bit-lsb"}, 0.0,
        )
        self.assertTrue(report["validation_passed"])
        self.assertEqual(report["output"], [-1.0, 0.0, 1.0])

    def test_ternary_msb_decode(self):
        result = decode_ternary([0b00011000], values_count=3, shape=[3],
                                codebook={0: -1, 1: 0, 2: 1}, bit_order="msb")
        self.assertEqual(result, [-1.0, 0.0, 1.0])

    def test_ternary_requires_explicit_codebook(self):
        with self.assertRaises(FallbackError):
            decode_ternary([0], values_count=1, shape=[1], codebook={0: -1, 1: 0})

    def test_ternary_rejects_reserved_symbol(self):
        with self.assertRaises(FallbackError):
            decode_ternary([0x03], values_count=1, shape=[1],
                           codebook={0: -1, 1: 0, 2: 1})

    def test_ternary_rejects_nonzero_padding(self):
        with self.assertRaises(FallbackError):
            decode_ternary([0b10000100], values_count=3, shape=[3],
                           codebook={0: -1, 1: 0, 2: 1}, bit_order="lsb")

    def test_metadata_mutation_fails_validation(self):
        metadata = {"quantization": {"axis": 0, "scale": [0.5, 1.0]}}
        # The executor owns no reference to mutate, and returns an explicit preservation check.
        result = execute_and_validate(
            "int8", {"values": [2, 3], "shape": [2], "scale": 0.5},
            metadata, 0.0,
        )
        self.assertTrue(result["metadata_preserved"])
        self.assertEqual(metadata["quantization"]["axis"], 0)

    def test_missing_tolerance_is_not_accepted(self):
        with self.assertRaises(FallbackError):
            execute_and_validate("float16", {"words": [0], "shape": [1]}, {}, float("nan"))

    def test_unsupported_feature_fails_closed(self):
        with self.assertRaises(FallbackError):
            execute_and_validate("int4", {}, {}, 0.0)


if __name__ == "__main__":
    unittest.main()
