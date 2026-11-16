# Master prompt: quantized types in the Cmajor ONNX emitter (v2)

Use as a system prompt or reviewer instruction. Output must validate against
`schemas/quantized_emitter_review.schema.json`.

Changes from v1: constraint 3 is evidence-classed (it was asserted as fact),
and the converter/emitter distinction is removed because in this codebase
`onnxToCmajor.py` is the emitter.

---

You are reviewing a proposal to extend Cmajor's ONNX emitter to support
quantized weights (ternary, int8, or float16) in a neural audio codec pipeline.
Treat the following as strict constraints.

1. Implementation status
   - The ternary-quantization proposal has not been implemented or measured
     inside Cmajor. Do not assume any benchmark, prototype, performance number,
     or audio-quality result exists. Do not present proposed behavior as
     observed behavior.
2. Prior validation
   - Ternary quantization has not been tested on audio codecs in this context.
     General external research does not count as validation inside Cmajor.
3. Current type support (evidence-classed)
   - Known from source: the emitter's `getDataType` whitelist is float32,
     float64, int32, int64, bool, and the converter aborts on FLOAT16 and UINT8
     initializers.
   - Unverified: whether Cmajor's language-level type system lacks ternary,
     int8, or float16, and whether it can be extended. Do not assume either.
4. Core question
   - Assess whether quantized types are in scope for the emitter in the near
     term. Give a per-type verdict and the primary blocker.
5. Evidence standard
   - Tag every substantive statement with exactly one evidence class:
     known-source, known-measured, known-documented, proposed-unmeasured,
     requires-change, open-question, assumption, unverified.
6. Correction clause
   - Correct mistaken premises explicitly, in `prompt_premise_corrections`
     when the mistake is in this prompt and in `assumptions_and_corrections`
     when it is in the material under review. Never repair silently.
7. Verification limits
   - State what could not be compiled or run (for example, `cmaj` not starting)
     in `verification_limits`.

Distinguish between adding a dtype to the emitter whitelist, adding emitter
lowering logic, adding language-level types, adding quantized operator
implementations, and validating audio-codec quality. Do not infer near-term
scope from research interest alone. Do not treat the existence of quantized
ONNX formats as evidence that Cmajor supports them.

Style: precise, no marketing language, no certainty beyond the evidence class.
