# Quantized Types in the Cmajor ONNX Emitter — Master Review Prompt

You are reviewing a proposal to extend Cmajor's ONNX emitter to support quantized weights—ternary, INT8, or FLOAT16—in a neural audio codec pipeline.

Treat the following as strict constraints.

## Constraints

1. **Implementation status.** The ternary-quantization proposal has NOT been implemented or measured in the target codebase. Do not assume any empirical result, benchmark, working prototype, or audio-quality result exists. Do not present proposed behavior as observed behavior.
2. **Prior validation.** Ternary quantization has NOT been tested on audio codecs in this context. General external research is not validation inside Cmajor or for codecs.
3. **Type support.** The converter's `getDataType` whitelist (float32, float64, int32, int64, bool) excludes ternary, INT8, and FLOAT16. Language-level type support is UNVERIFIED unless the relevant Cmajor language/compiler artifacts are directly inspected. In this codebase, `onnxToCmajor.py` IS the emitter; do not separate a "converter flag" from an emitter change.
4. **Core question.** Assess whether quantized types are in scope for the Cmajor emitter in the near term, or currently out of scope. Give an independent verdict for ternary, INT8, and FLOAT16.
5. **Evidence standard.** Classify every factual or predictive claim using `CLASSIFIER.md`. Do not blend classes. If evidence is missing, say so.
6. **Correction clause.** Correct mistaken statements about the converter, Cmajor, or prior validation explicitly. Corrections to the prompt itself go in `prompt_premise_corrections`; corrections to evidence go in `assumptions_and_corrections` with kind `evidence-correction`.

## Evidence base — populate first

Before making claims, populate `evidence_base`:
- `sources_consulted`: exact paths/URLs of artifacts actually read in this review session.
- `sources_not_consulted`: relevant artifacts the review depends on but which were not read.
- `all_known_claims_second_hand`: true only if every known-* claim is based on someone else's transcription/notes rather than a primary artifact read directly in this session.

A known-source or known-documented claim whose original artifact was not read directly MUST set `second_hand: true` and include `original_source` when known. Reading notes does not count as reading the original source.

## Classification

Apply the ordered decision procedure in `CLASSIFIER.md`, first match wins, to every claim. The JSON Schema enforces section/class compatibility. `required_changes` entries are structural recommendations, not claims; they are exempt from classification, but every factual basis for a required change must be represented by a known-* claim in `known`. Set `extensibility_verified: true` only when directly inspected evidence establishes that the target type system, emitter, or operator set can be extended as stated; include `extensibility_basis`.

## Verification limits

List every environment or process limit that caps confidence (for example: compiler unavailable, no compile check, runtime unavailable, target model fixtures absent). At least one entry is required. Do not imply a command ran unless its output was observed in this session.

## Output

Return ONLY JSON conforming to `quantized_emitter_review.schema.json`. No prose outside the JSON. No Markdown fence. The `bottom_line` must contain 2–4 short sentences as an array of strings.
