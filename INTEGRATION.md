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
4. **Independent execution and validation.** Run
   `python scripts/independent_check.py [report.json]` to copy only the
   `schema_review` package into a temporary directory, invoke its CLI from an
   unrelated working directory twice, and compare the results. The harness
   derives structural mutation expectations from the bundled schema and fails
   if any mandatory mutant is accepted. It is deliberately limited to the
   bundled schema and fixture; it does not independently test semantic rules
   or establish general JSON Schema conformance.
5. **CI artifact.** The workflow runs the harness and uploads
   `validation_report.json` as `independent-validation-report`.
6. **Not yet run here:** the hosted GitHub Actions workflow. Its result must be
   read from the Actions run; adding the workflow is not evidence that it ran.
