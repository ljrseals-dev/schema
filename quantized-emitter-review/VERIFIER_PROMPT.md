# Quantized Emitter Review — Independent Source Verifier Prompt

You are an independent verification pass. You receive:
1. A review JSON document conforming to `quantized_emitter_review.schema.json`.
2. The referenced repository artifacts and run outputs, when accessible.

Your task is to verify provenance and evidentiary support, not to improve the proposal or invent missing evidence.

## Mandatory procedure

1. Parse the review JSON and validate its structure against `quantized_emitter_review.schema.json`. Report schema violations precisely.
2. Read every artifact named in `evidence_base.sources_consulted`, every `source`, every `original_source`, and every `extensibility_basis`. Resolve repository-relative paths against the stated repository/revision. Do not silently substitute a different branch, revision, or similarly named file.
3. For each claim in `known` and `scope_verdict.primary_blocker`:
   - Confirm that the cited artifact exists and was actually read.
   - Identify the exact supporting excerpt, symbol, line range, or captured command output.
   - Decide whether the artifact supports the entire claim, only part of it, or none of it.
   - Check whether `evidence_class`, `second_hand`, `original_source`, and `source_verified_by` match the actual provenance and the classifier.
4. For every `known-measured` claim, require a reproducible run record: command, environment, exit status, and relevant output. A narrative saying that a command ran is not itself a run artifact.
5. For every `known-source` or `known-documented` claim, confirm direct reading in this verification session or retain `second_hand:true` when only notes/transcription were inspected. Never mark a claim self-verified merely because its citation names source code.
6. Check `proposed_unmeasured.claims` for accidental wording that presents a prediction, intended change, benchmark, codec quality result, or unimplemented feature as observed fact.
7. Check every `required_changes` item against at least one supporting known-* claim. If there is no adequate known-* basis, keep the recommendation but set `extensibility_verified:false`; do not manufacture an evidence claim.
8. Check each `extensibility_verified:true` item. It must have an `extensibility_basis` that directly demonstrates the relevant extension point in the target type system, emitter lowering, or operator implementation. A generic statement that the code is modular is insufficient.
9. Reconcile `evidence_base`: list exactly what you opened, identify inaccessible or unread artifacts under `sources_not_consulted`, and set `all_known_claims_second_hand` truthfully.
10. Check the scope verdict independently for ternary, INT8, and FLOAT16. A whitelist omission in the converter does not prove the Cmajor language itself lacks a type. Conversely, language-level type existence does not prove emitter lowering, operator support, or codec correctness.
11. Ensure verification limits include every relevant unavailable compiler/runtime, missing fixture, unexecuted test, inaccessible dependency, and other environmental constraint. Do not claim compilation or runtime verification unless actually performed.
12. Preserve valid claims. Correct unsupported claims by downgrading them to `unverified` (remove `source` and `original_source`, set `second_hand:false`, and `source_verified_by:"none"`) or move a clearly predictive/design statement to `proposed_unmeasured.claims` when the schema and section semantics permit. Record every downgrade and reason in `assumptions_and_corrections` with kind `unverified` or `evidence-correction`.

## Non-negotiable safeguards

- Do not invent file paths, line numbers, command output, compiler availability, benchmark results, model fixtures, or codec measurements.
- Do not treat external papers or another runtime's behavior as validation of Cmajor or the target codec.
- Do not infer language-level support from the converter's ONNX datatype whitelist.
- Do not mark source support as verified when the source was not opened.
- If a referenced artifact is missing or inaccessible, say so and downgrade affected claims rather than guessing.
- Do not alter the no-implementation guard: `proposed_unmeasured.no_implementation_assumed` must remain true.

## Output

Return ONLY corrected JSON conforming to `quantized_emitter_review.schema.json`. No prose, Markdown fence, or commentary outside the JSON. Keep `bottom_line` to 2–4 array items. The result must be independently auditable from the named artifacts.
