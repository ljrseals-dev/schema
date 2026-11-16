# Integration and validation notes

The package can be used as a dependency-free runtime validator. The optional
`test` extra installs `jsonschema` for independent Draft-07 differential
validation; it is not a runtime dependency.

1. **Repository root.** The GitHub Actions workflow is at
   `.github/workflows/ci.yml` and runs from the repository root.
2. **Bundled schema.** The semantic rules in `src/schema_review/validate.py`
   are specific to
   `src/schema_review/data/quantized_emitter_review.schema.json`. For another
   schema, pass `--schema <schema.json> --structural-only`; the semantic
   rules are intentionally skipped because they assume the bundled fields.
3. **Fixture.** `examples/review_example.json` is the worked review fixture.
   `tests/fixtures/review_example.json` is kept as the test copy; tests check
   the bundled schema against the example.
4. **Independent execution.** Run
   `python scripts/independent_check.py [report.json]`. The harness copies
   only the package into a temporary directory, invokes the CLI from an
   unrelated working directory twice, and checks deterministic output.
5. **Differential structural oracle.** The harness checks the schema with
   `jsonschema.Draft7Validator`, then compares the project CLI's
   `--structural-only` result with that independent implementation for
   schema-derived mutations. Any disagreement or accepted mandatory-invalid
   mutation fails the run. This is limited to the bundled schema and fixture;
   it does not prove general JSON Schema conformance.
6. **CI report.** The unit-test invocation runs the harness once and writes
   `validation_report.json`. GitHub Actions uploads it as the
   `independent-validation-report` artifact. The report records environment,
   source hashes, mutation counts, and any disagreements.
7. **Semantic boundary.** The mutation oracle validates structural behavior,
   not the independent correctness of semantic rules. Those rules have
   focused regression tests but still need a separately authored semantic
   specification/oracle. Neither layer establishes that review claims are
   true or that cited evidence supports them.
8. **Hosted execution.** Check the pull request's Actions run for the result.
   A workflow definition or a previous successful run is not evidence that a
   later commit has passed.
