# Integration notes

These files add a validator, tests, CI, and a technical note to a repository
that already has its own prompts, schema, and worked review.

1. **Place at the repository root.** `.github/workflows/ci.yml` only runs from
   the root. If you keep the project in a subfolder, add
   `defaults: run: working-directory: <folder>` to the job and move the
   workflow file to the root `.github/workflows/`.
2. **Schema reconciliation.** The bundled schema lives at
   `src/schema_review/data/quantized_emitter_review.schema.json`. If the
   repository already has a review schema, either replace the bundled file with
   it (and update the semantic rules in `validate.py` to its field names) or
   validate against it with `python -m schema_review <review.json> --schema
   <schema.json> --structural-only`.
3. **Fixture.** `tests/fixtures/review_example.json` matches the bundled schema
   only. It is not the repository's own worked review.
4. **Not yet run:** the GitHub Actions workflow, and validation of any
   pre-existing worked review against any pre-existing schema.
