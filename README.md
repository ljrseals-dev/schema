# schema

An evidence-classed review schema, master prompt, and validator for assessing
whether quantized weight types (ternary, int8, float16) are in scope for the
Cmajor ONNX emitter (`onnxToCmajor.py`).

The accompanying note is [`Paper.md`](Paper.md).

## Contents

| Path | Purpose |
|---|---|
| `schemas/quantized_emitter_review.schema.json` | JSON Schema (draft-07) for a review output |
| `prompts/master_prompt.md` | Reviewer prompt (v2) whose output targets the schema |
| `examples/review_example.json` | Worked review of the `realm-quant` findings |
| `src/schema_review/` | Dependency-free validator: structural checks plus semantic rules |
| `tests/` | `unittest` suite |

## Use

```bash
python -m pip install -e .
python -m schema_review examples/review_example.json
python -m unittest discover -s tests -v
```

Exit code is 0 when the review validates and 1 otherwise. Warnings go to stderr.

## What the schema enforces

- Every claim carries one `evidence_class`.
- Corrections to the prompt itself live in `prompt_premise_corrections`,
  separate from corrections to the material under review.
- `proposed_unmeasured.no_implementation_assumed` must be `true`.
- Each required change carries `extensibility_verified`.
- `verification_limits` is required and must be non-empty.

Semantic rules beyond JSON Schema (in `validate.py`): `known` holds only
`known-*` claims, each with a `source`; a `near-term-in-scope` verdict needs a
`known-*` primary-blocker assessment; the bottom line is 2-4 sentences.

## Limits

- The validator checks form, not truth. It cannot tell whether a cited source
  supports a claim.
- The structural checker implements only the keywords the bundled schema uses.
  It has not been compared against the `jsonschema` package.
- The example review rests on the author's own notes in `realm-quant`; the
  Cmajor repository itself was not read.
- Nothing here has been evaluated for agreement between reviewers.

## License

MIT. See `LICENSE`.
