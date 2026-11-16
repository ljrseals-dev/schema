# Evidence-Classed Review of Quantized-Type Proposals for the Cmajor ONNX Emitter

**Author:** Lewis James Seals Jr
**Date:** November 2026
**Status:** Technical note. Describes a schema and validator. Reports no new measurements.

## Abstract

A prior study (`ljrseals-dev/realm-quant`) tested Cmajor's `onnxToCmajor.py`
against the Descript DAC 24 kHz codec and reported that the route is blocked
mainly by operator coverage. A natural follow-on question is whether quantized
weight types (ternary, int8, float16) belong in the emitter. Answering it
invites mixing source-read facts, measurements, external research, and
speculation. This note describes a review schema that forces each statement
into one evidence class, separates corrections to the review prompt from
corrections to the material under review, and requires an explicit list of what
could not be verified. It also describes a small validator and a worked
example. The schema checks the form of a review, not its truth, and has not
been evaluated for reviewer agreement.

## 1. Problem

The review question has five distinct parts that are easy to blur: adding a
dtype to the emitter whitelist, adding emitter lowering logic, adding
language-level types, adding quantized operator implementations, and validating
codec quality. A review that says "ternary needs emitter changes" without
saying which of these it means is not checkable. A second failure is inheriting
an unverified premise from the prompt. The first version of the master prompt
asserted that Cmajor has no ternary, int8, or float16 types. The source-level
evidence supports a narrower statement, covered in Section 4.

## 2. Background: what the prior study reports

All of the following is author-reported from `realm-quant`; the Cmajor
repository was not read for this note.

- Release 1.0.3209 implements sixteen processors in `onnx.cmajor`.
- The shipped `Conv` processor ignores ONNX `pads` and has no stride argument,
  which the study reports would mis-lower all 151 DAC convolutions.
- `getDataType` whitelists float32, float64, int32, int64, and bool. The
  converter aborted on FLOAT16 and UINT8 inputs.
- INT8 variants introduce operators (`ConvInteger`, `DynamicQuantizeLinear`,
  `DequantizeLinear`, `MatMulInteger`) with no Cmajor processors.
- A 5.39x literal expansion was measured on codebook weights. The full-model
  inline sizes were extrapolated, not measured.
- `cmaj` did not start in the test environment, so nothing was compile-checked.

The later `realm-quant` drop adds a Python ternary quantizer (packed 2-bit
symbols with per-output-channel scales, dequantized to float for execution) and
a native ONNX Runtime bridge. Neither modifies the Cmajor emitter or type
system, and the repository states that no native ternary arithmetic is claimed.

## 3. Schema design

A review is a JSON document (`src/schema_review/data/quantized_emitter_review.schema.json`)
with these required parts.

**Evidence classes.** Each claim has one of: `known-source`, `known-measured`,
`known-documented`, `proposed-unmeasured`, `requires-change`, `open-question`,
`assumption`, `unverified`. The classes are an enum so the separation does not
depend on prose.

**Prompt-premise corrections.** A required array, distinct from
`assumptions_and_corrections`, so a mistake in the prompt is not buried among
findings about the material.

**No-implementation guard.** `proposed_unmeasured.no_implementation_assumed`
is pinned to `true`.

**Extensibility flag.** Each required change has `extensibility_verified`. It
should be `false` unless the material shows the target (type system, emitter,
processor set) can be extended that way. Any `true` raises a warning. The
worked example sets every flag to `false`, because the author's notes do not
show extensibility for any of them.

**Verification limits.** A required, non-empty list of what could not be
compiled or run.

## 4. Premise corrections that motivated the schema

1. The prompt said Cmajor has no ternary, int8, or float16 types. The material
   shows only the emitter whitelist and the converter's aborts. Language-level
   support is unverified in both directions.
2. The prompt separated a "converter flag" from an "emitter change". In this
   codebase `onnxToCmajor.py` is the emitter, so the separation does not apply.
3. The prior note says ternary literals eliminate the literal expansion
   "entirely". That is unmeasured, scale metadata adds literals, and it is an
   open question whether short ternary values could emit as existing int32 or
   float32 literals with no new type.

## 5. Validator

`schema_review` performs structural validation with a small draft-07 subset
checker, then applies semantic rules that JSON Schema cannot express: `known`
holds only `known-*` claims with a source; a `near-term-in-scope` verdict needs
a `known-*` blocker assessment and cannot coexist with 'out of scope' wording
in the summary or bottom line; blank strings and duplicate entries are rejected;
the bottom line is two to four sentences; and verification limits are
non-empty. The verdict-consistency and sentence-count rules are heuristics. Unit tests cover these rules. The
validator catches malformed or internally inconsistent reviews. It cannot tell
whether a cited source supports a claim.

## 6. Worked example

`examples/review_example.json` applies the schema to the `realm-quant`
findings. Its verdict is that all three types are currently out of scope for
DAC-class graphs, with operator coverage (the `Conv` processor) as the primary
blocker. The example inherits every limit of its sources: the claims are
author-reported, source-read, and uncompiled.

## 7. Limitations

- No reviewer-agreement study was run; the schema's usefulness for
  consistent reviews is untested.
- The bundled validator has not been compared against the `jsonschema`
  package.
- The example depends on the author's notes rather than the Cmajor source.
- Evidence classes are assigned by the reviewer, so a mislabeled claim can
  still pass validation.

## 8. What this note does not claim

It does not claim that ternary, int8, or float16 work in Cmajor, that any
quantized variant preserves codec quality, or that the schema improves review
accuracy. It records a structure for making those questions answerable.

## References

1. `ljrseals-dev/realm-quant`: `Paper.md`, `notes/cmajor-research.md`, and the
   later Python workbench and bridge.
2. Cmajor, `cmajor-lang/cmajor`, release 1.0.3209 (cited via reference 1; not
   read directly).
