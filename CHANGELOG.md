# Changelog

## 0.2.0
- Reject blank strings (schema `pattern`) and require non-empty `known`.
- Abbreviation-aware sentence counter.
- Primary blocker restricted to known-*, unverified, or assumption.
- Reject duplicate entries within a section.
- Reject a near-term-in-scope verdict that coexists with "out of scope" text.
- Warn on `extensibility_verified: true` in every change list.
- Example review no longer claims verified extensibility.
- Prompt v3: evidence classes apply to claim sections only.
- Schema moved into the package (`src/schema_review/data/`) so non-editable installs work.
