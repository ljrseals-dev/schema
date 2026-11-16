"""Regression test for the schema-derived independent CLI harness."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "independent_check.py"


class IndependentCheckTests(unittest.TestCase):
    def test_clean_room_and_all_mandatory_mutants(self) -> None:
        spec = importlib.util.spec_from_file_location("independent_check", SCRIPT)
        if spec is None or spec.loader is None:
            self.fail("could not load independent validation harness")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = module.main(os.environ.get("SCHEMA_REVIEW_VALIDATION_REPORT"))

        self.assertEqual(result, 0, output.getvalue()[-4000:])


if __name__ == "__main__":
    unittest.main()
