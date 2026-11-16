import copy
import json
import unittest
from pathlib import Path

from schema_review import load_schema, validate_review

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "examples" / "review_example.json").read_text(encoding="utf-8"))


class ValidateTests(unittest.TestCase):
    def check(self, mutate):
        doc = copy.deepcopy(EXAMPLE)
        mutate(doc)
        return validate_review(doc, load_schema())[0]

    def test_example_is_valid(self):
        errs, _ = validate_review(EXAMPLE, load_schema())
        self.assertEqual(errs, [])

    def test_missing_required_section(self):
        self.assertTrue(self.check(lambda d: d.pop("verification_limits")))

    def test_bad_enum(self):
        self.assertTrue(self.check(lambda d: d["known"][0].update(evidence_class="guess")))

    def test_extra_property_rejected(self):
        self.assertTrue(self.check(lambda d: d.update(extra=1)))

    def test_no_implementation_flag_must_be_true(self):
        self.assertTrue(self.check(lambda d: d["proposed_unmeasured"].update(no_implementation_assumed=False)))

    def test_known_claim_needs_source(self):
        errs = self.check(lambda d: d["known"][0].pop("source"))
        self.assertTrue(any("requires a source" in e for e in errs))

    def test_known_section_rejects_unverified(self):
        errs = self.check(lambda d: d["known"][0].update(evidence_class="unverified"))
        self.assertTrue(any("only hold known" in e for e in errs))

    def test_bottom_line_sentence_count(self):
        errs = self.check(lambda d: d.update(bottom_line="Only one sentence."))
        self.assertTrue(any("bottom_line" in e for e in errs))

    def test_empty_verification_limits(self):
        errs = self.check(lambda d: d.update(verification_limits=[]))
        self.assertTrue(any("verification_limits" in e for e in errs))

    def test_near_term_needs_known_blocker(self):
        def m(d):
            d["scope_verdict"]["per_type"]["float16"] = "near-term-in-scope"
            d["scope_verdict"]["primary_blocker"]["evidence_class"] = "assumption"
        errs = self.check(m)
        self.assertTrue(any("near-term-in-scope" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
