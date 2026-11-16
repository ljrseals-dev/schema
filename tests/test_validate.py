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

    def test_blank_strings_rejected(self):
        self.assertTrue(self.check(lambda d: d.update(verification_limits=["", " "])))
        self.assertTrue(self.check(lambda d: d["known"][0].update(source="   ")))

    def test_abbreviations_not_sentence_breaks(self):
        from schema_review.validate import _sentences
        self.assertEqual(_sentences("Use e.g. float16 first. It is unmeasured."), 2)
        self.assertEqual(_sentences("Costs 1.14 GB. See Fig. 2 for details. Done."), 3)

    def test_empty_known_rejected(self):
        self.assertTrue(self.check(lambda d: d.update(known=[])))

    def test_verdict_summary_contradiction(self):
        def m(d):
            d["scope_verdict"]["per_type"]["float16"] = "near-term-in-scope"
        errs = self.check(m)
        self.assertTrue(any("says 'out of scope'" in e for e in errs))

    def test_blocker_class_restricted(self):
        errs = self.check(lambda d: d["scope_verdict"]["primary_blocker"].update(evidence_class="requires-change"))
        self.assertTrue(any("primary_blocker" in e for e in errs))

    def test_duplicates_rejected(self):
        errs = self.check(lambda d: d["known"].append(copy.deepcopy(d["known"][0])))
        self.assertTrue(any("duplicate" in e for e in errs))

    def test_extensibility_warning_in_every_change_list(self):
        for name in ("type_system", "emitter_lowering", "operators", "validation"):
            doc = copy.deepcopy(EXAMPLE)
            doc["required_changes"][name][0]["extensibility_verified"] = True
            _, warns = validate_review(doc, load_schema())
            self.assertTrue(any(name in w for w in warns), name)

    def test_example_has_no_extensibility_claims(self):
        _, warns = validate_review(EXAMPLE, load_schema())
        self.assertEqual(warns, [])

    def test_schema_ships_inside_package(self):
        from schema_review.validate import DEFAULT_SCHEMA
        self.assertEqual(DEFAULT_SCHEMA.parent.parent.name, "schema_review")
        self.assertTrue(DEFAULT_SCHEMA.exists())


if __name__ == "__main__":
    unittest.main()
