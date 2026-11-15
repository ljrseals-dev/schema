# Evidence Classification Classifier

This is the normative, ordered decision procedure for assigning `evidence_class`. Apply the rules to each individual claim, not to an entire paragraph. First matching rule wins.

| Rule | Test | Classification | Required provenance |
|---|---|---|---|
| R1 | Established by reading the primary artifact (source/code/doc) directly in this review session | Code/source: `known-source`; documentation: `known-documented` | `source`; `second_hand:false`; `source_verified_by:"self"` |
| R2 | Established from a primary artifact read by someone else and transcribed into notes/summaries | Code/source: `known-source`; documentation: `known-documented` | `source` points to notes; `second_hand:true`; `original_source` if known; `source_verified_by:"other"` |
| R3 | Established by running something in this session (converter, compiler, test, benchmark, measurement) | `known-measured` | `source` identifies captured run output/artifact; `second_hand:false`; `source_verified_by:"self"` |
| R4 | Asserted by an artifact but not independently checked by the reviewer, and not run | `known-documented` | `source`; provenance reflects whether the artifact was read directly |
| R5 | Prediction or estimate not yet run (speed, size, quality, compatibility, etc.) | `proposed-unmeasured` | `source` optional; if present, identify the proposal/artifact |
| R6 | Design proposal or intended change not yet implemented | `proposed-unmeasured` | `source` optional; if present, identify the proposal/artifact |
| R7 | Reviewer's own belief with no artifact behind it | `assumption` | Omit `source`; explain as an assumption |
| R8 | Could not be verified, or artifacts conflict such that no supported conclusion can be drawn | `unverified` | Omit `source`; explain the verification gap |

## Tie-breaks and guards

- **First match wins.** A result measured in this session remains `known-measured` even if a document also reports it.
- **Downgrade on ambiguity.** If two classes fit equally and rule order does not resolve the ambiguity, use the lower-certainty class and record the downgrade in `assumptions_and_corrections`.
- **External research never upgrades a target-system claim.** A paper about another system cannot establish Cmajor support or codec validation in this project.
- **Second-hand is sticky.** Reading notes is not reading source. Every downstream known-source/known-documented claim derived from those notes remains `second_hand:true` until the original artifact is read directly in this session.
- **No cross-section leakage.** `known` accepts only known-* classes. `proposed_unmeasured.claims` accepts only `proposed-unmeasured`. Assumptions and unverified statements belong in `assumptions_and_corrections` or another explicitly typed field, not in either claim list.
- **Required changes are not claims.** Their factual basis must appear in `known`; otherwise they remain recommendations and `extensibility_verified` must be false.
- **Source optionality.** A source is mandatory for known-source, known-measured, and known-documented. It is optional for proposed-unmeasured, assumption, and unverified.
- **Provenance values.** Use `source_verified_by:"self"` only when the reviewer personally inspected the named source; use `"other"` for second-hand transcription; use `"none"` when source support has not been checked. This field records provenance, not a guarantee that the claim is correct.

## Worked examples

- `onnxToCmajor.py does not reject unknown ops` → R2 if this statement comes only from notes → `known-source`, `second_hand:true`, `source:"notes/cmajor-research.md"`, `original_source:"tools/onnxToCmajor.py"`.
- `cmaj would not start in the test environment` → R3 only if actually attempted in this session and output captured → `known-measured`.
- `INT8 would give a 2–4× CPU speedup` → R5 → `proposed-unmeasured`.
- `DAC 24 kHz is representative` with no supporting artifact → R7 → `assumption`.
- `ONNX Runtime Web handles every op natively` when not verified against relevant primary documentation and version → R8 → `unverified`.

The classifier makes assignment rules reproducible; it does not automatically establish source support. A separate source-verification pass is required.
