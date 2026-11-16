"""Regression tests for malformed public fallback payloads."""
import unittest

from schema_review.fallback_runtime import FallbackError, execute_and_validate


class MalformedFallbackPayloadTests(unittest.TestCase):
    def test_missing_required_fields_raise_domain_error(self):
        cases = [
            ("float16", {"shape": [1]}),
            ("int8", {"values": [1], "shape": [1]}),
            ("ternary", {"packed": [0], "values_count": 1, "shape": [1]}),
        ]
        for feature, payload in cases:
            with self.subTest(feature=feature):
                with self.assertRaises(FallbackError):
                    execute_and_validate(feature, payload, {}, 0.0)

    def test_non_iterable_payload_values_raise_domain_error(self):
        with self.assertRaises(FallbackError):
            execute_and_validate("float16", {"words": None, "shape": [1]}, {}, 0.0)
        with self.assertRaises(FallbackError):
            execute_and_validate("int8", {"values": 1, "shape": [1], "scale": 1.0}, {}, 0.0)
        with self.assertRaises(FallbackError):
            execute_and_validate(
                "ternary", {"packed": None, "values_count": 1, "shape": [1],
                            "codebook": {0: -1, 1: 0, 2: 1}}, {}, 0.0
            )

    def test_bool_codebook_keys_are_not_integer_symbols(self):
        with self.assertRaises(FallbackError):
            execute_and_validate(
                "ternary", {"packed": [0], "values_count": 1, "shape": [1],
                            "codebook": {False: -1, 1: 0, 2: 1}}, {}, 0.0
            )

    def test_non_object_payload_and_non_string_feature_fail_closed(self):
        with self.assertRaises(FallbackError):
            execute_and_validate("int8", None, {}, 0.0)
        with self.assertRaises(FallbackError):
            execute_and_validate(None, {}, {}, 0.0)


if __name__ == "__main__":
    unittest.main()
