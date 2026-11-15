# Quantized Emitter Review Kit

A prompt-and-schema kit for evidence-disciplined review of proposed ternary, INT8, and FLOAT16 support in Cmajor's ONNX emitter for neural audio codec workflows.

## Artifacts

- [Master review prompt](MASTER_PROMPT.md) — scope, constraints, evidence-base requirements, and output instructions.
- [Evidence classifier](CLASSIFIER.md) — ordered rules for deterministic evidence-class assignment.
- [Review output schema](quantized_emitter_review.schema.json) — Draft-07 JSON Schema for the structured review.
- [Independent source verifier prompt](VERIFIER_PROMPT.md) — second-pass source/provenance verification and downgrade rules.

## Important status boundary

This repository contains review tooling, not a quantized Cmajor emitter implementation. It contains no claim that ternary, INT8, or FLOAT16 has been implemented, benchmarked, or validated on an audio codec. A review must inspect the target repository and record exactly which sources and run artifacts were consulted.

## Schema note

The schema requires `source` for known-source, known-measured, and known-documented claims. It intentionally permits `source` to be omitted for `proposed-unmeasured` claims, matching classifier rules R5 and R6. The schema cannot prove that a citation supports a claim; the independent verifier addresses that gap.

## Use

1. Read `MASTER_PROMPT.md` and `CLASSIFIER.md`.
2. Inspect the target Cmajor ONNX emitter, language/compiler sources, operator coverage, and relevant fixtures.
3. Produce JSON conforming to `quantized_emitter_review.schema.json`.
4. Run an independent verification pass using `VERIFIER_PROMPT.md`.
5. Preserve unverified claims as unverified; do not fill evidence gaps with assumptions.
