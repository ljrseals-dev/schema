# schema

An evidence-classed review schema, master prompt, and validator for assessing
whether quantized weight types (ternary, int8, float16) are in scope for the
Cmajor ONNX emitter (`onnxToCmajor.py`). It also contains a deterministic
feature-capability/fallback decision model for an adapter implementation.

## Contents

| Path | Purpose |
|---|---|
| `src/schema_review/data/quantized_emitter_review.schema.json` | Draft-07 review-output schema |
| `src/schema_review/data/feature_fallback.schema.json` | Feature capability/fallback report contract |
| `src/schema_review/feature_fallback.py` | Deterministic native/fallback/blocked/unsupported classifier |
| `docs/FEATURE_FALLBACK_MODEL.md` | Architecture, per-dtype policy, acceptance criteria and failure behavior |
| `prompts/master_prompt.md` | Reviewer prompt (v3) whose output targets the review schema |
| `examples/review_example.json` | Worked review of the `realm-quant` findings |
| `src/schema_review/` | Dependency-free runtime validator |
| `scripts/independent_check.py` | Clean-room, schema-derived differential check |
| `tests/` | Unit, regression, and independent validation tests |

## Feature/fallback decisions

Use `schema_review.feature_fallback.classify_model(report)` to classify each
requested feature from adapter-supplied capability probes and evidence. The
classifier is deterministic and fail-closed:

- `native`: every native gate is true and has evidence.
- `fallback`: the exact feature-specific transform is enabled, required
  shared gates and evidence exist, metadata is preserved, numerical validation
  passes, and measured error is within the declared tolerance.
- `blocked`: a capability or its evidence is missing/unknown.
- `unsupported`: a required native capability is explicitly false and no
  eligible fallback is available.

The model is ready only if every requested feature is native or fallback.
FLOAT16 promotion, INT8 dequantization, and ternary decoding are candidate
paths—not claims of current emitter support. Ternary requires a verified
encoding; INT8 requires exact quantization metadata preservation. See
[docs/FEATURE_FALLBACK_MODEL.md](docs/FEATURE_FALLBACK_MODEL.md).

This classifier consumes reports; it does not inspect ONNX files, run Cmajor,
perform tensor conversion, or independently validate the evidence supplied to
it. Do not wire it to optimistic defaults.

## Install and use

Runtime installation has no third-party dependencies:

```bash
python -m pip install .
python -m schema_review examples/review_example.json
```

To run the full development test suite, including the independent Draft-07
oracle, install the test extra:

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
python scripts/independent_check.py validation_report.json
```

The CLI returns 0 for a valid review, 1 for validation errors, and 2 when the
review or schema cannot be read or parsed. Warnings are written to stderr.

## What is enforced

The review JSON Schema requires evidence classes, separates prompt-premise
corrections from evidence corrections, requires
`proposed_unmeasured.no_implementation_assumed` to be true, requires each
proposed change to state whether extensibility was verified, and requires
non-empty verification limits.

Additional semantic rules in `validate.py` require known claims to have
known-* classes and sources, constrain the primary blocker, require a known-*
blocker for near-term scope, check explicit scope contradictions, reject
duplicate entries within sections, and require a 2-4 sentence bottom line.
An `extensibility_verified: true` value produces a warning for human review.

## Validation and limits

- The structural checker implements the documented subset used by the
  bundled review schema; it is not a general-purpose JSON Schema implementation.
- CI compares the subset checker with the independent `jsonschema` Draft-07
  implementation over the bundled fixture and generated structural mutations.
  This is differential coverage, not exhaustive conformance testing.
- The semantic scope-consistency and sentence-count checks remain heuristics.
- The feature classifier tests decision logic, not actual ONNX decoding,
  Cmajor compilation, tensor conversion, or runtime parity.
- The validator checks structure and selected consistency rules, not whether
  cited sources actually support claims. Source verification remains a human
  or separate evidence-retrieval task.
- The worked review attributes Cmajor findings to the author's
  `realm-quant` notes; the Cmajor repository itself was not independently read
  as part of this package.
- No reviewer-agreement study or codec-quality evaluation is claimed.

See [INTEGRATION.md](INTEGRATION.md) for existing validator execution boundaries.

## License

MIT. See `LICENSE`.
