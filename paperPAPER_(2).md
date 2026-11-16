# Evidence-Classed Review of Quantized-Type Proposals for the Cmajor ONNX Emitter: A Schema, a Validator, and an Independent Validation Harness

**Authors:** Lewis Seals and Angela La'Vette Seals
**DOI:** https://zenodo.org/records/24267777
**Date:** November 2026
**Status:** Technical report. It describes a review schema, a validator, and a validation harness. It reports measurements of the validator only. It reports no measurements of Cmajor, of quantized models, or of audio quality.

---

## Abstract

A prior study in the `realm-quant` repository tested Cmajor's `onnxToCmajor.py` converter against the Descript DAC 24 kHz neural audio codec and concluded that the route is blocked mainly by operator coverage, not by any single conceptual flaw. A natural follow-on question is whether quantized weight types (ternary, int8, and float16) belong in the Cmajor ONNX emitter. Questions of this kind are easy to answer badly. Source-read facts, measurements, external research results, and speculation arrive in the same paragraph, and a reader cannot tell which is which. This report describes a structured way to answer the question. It defines a JSON review schema in which every claim carries one of eight evidence classes, in which corrections to the review prompt are kept apart from corrections to the material under review, and in which the reviewer must list what could not be verified. It describes a dependency-free validator that checks both the schema and a set of semantic rules. Most importantly, it describes an independent validation harness that derives its expectations from the schema file alone, drives the validator only through its command-line interface in a clean-room directory, and attempts 435 mutations of a reference review. In the run reported here, all 421 mutations that violate a mandatory schema constraint were rejected, and the seven surviving mutants were all empty-array cases that the schema deliberately permits. The harness establishes that the validator enforces the schema it ships with. It does not establish that any review is true, that the validator agrees with a general-purpose JSON Schema implementation, or that the schema improves the quality of human or machine reviews. Those limits are stated throughout.

---

## 1. Introduction

Neural audio models are large, and the browser is a constrained delivery target. Two broad routes exist for running such a model in a web page. One is a general interpreter, such as ONNX Runtime Web, which executes an ONNX graph directly. The other is a compiler, such as Cmajor's code-as-weights pipeline, which turns the graph into a statically compiled program in which weight values are compile-time constants. The `realm-quant` study measured both routes against one codec and reported that the interpreter route executes correctly but slower than real time on the reference hardware, while the compiler route is blocked before it reaches the question of speed.

Quantization enters this picture because it changes what a weight is. A float32 weight occupies 32 bits. A float16 weight occupies 16. An int8 weight occupies 8, plus metadata. A ternary weight takes one of three values and, in an ideal packing, occupies about 1.58 bits. When weights are emitted as source-code literals, representation matters in a second way: a float32 weight written as a decimal string is longer than its binary form, and the `realm-quant` notes report a measured expansion of 5.39 times on codebook weights. A smaller representation might, in principle, avoid that expansion.

That reasoning is plausible, and plausible reasoning is exactly where a review can go wrong. The conclusion that "ternary weights would fix the literal expansion" is a proposal. It has not been implemented in Cmajor, it has not been measured, and it has not been validated on an audio codec. Yet it is easy to write a paragraph in which that proposal sits next to a source-read fact and a measured number, with nothing to mark the difference.

This report treats the problem as one of structure. If every claim in a review must be filed under one evidence class, and if the review must also list what could not be verified, then the difference between a fact and a proposal becomes visible in the document itself. That is the purpose of the schema. The validator makes the structure enforceable, and the independent harness gives a way to check the validator without trusting the validator's author.

The report has a deliberately narrow scope. It does not claim that ternary, int8, or float16 work in Cmajor. It does not claim that any quantized variant preserves codec quality. It does not claim that the schema makes reviews more accurate. It claims only what it can show: that the validator, as shipped, rejects a large and systematically generated set of malformed reviews and accepts the reference review, and that this was checked by a procedure that does not share code with the validator.

---

## 2. Background

### 2.1 Cmajor and its ONNX converter

Cmajor is a programming language and toolchain for audio processing. Its repository includes a script, `onnxToCmajor.py`, that reads an ONNX file and emits Cmajor source. ONNX itself is an open model-interchange format. It was not created by Cmajor, and the converter is a reader of ONNX files, not part of the format.

According to the `realm-quant` notes, release 1.0.3209 of the Cmajor repository implements sixteen operator processors in its ONNX support template: `Tanh`, `Reshape`, `MatMul`, `Gemm`, `Add`, `Unsqueeze`, `Squeeze`, `Conv`, `GRU`, `LSTM`, `Slice`, `Concat`, `Gather`, `Pad`, `Transpose`, and `Constant`. The same notes report that the converter emits a call to an operator named after each node's type without checking whether such a processor exists, so a missing processor surfaces only when the generated program is compiled. The notes describe this as a deliberate design choice, a permissive converter paired with a strict compiler, and they draw the consequence that a successful conversion is not evidence that the graph can be compiled.

### 2.2 The codec under test

The Descript Audio Codec at 24 kHz has a convolutional encoder, a residual vector quantizer that maps encoder output to codebook indices, and a convolutional decoder with transposed-convolution upsampling and a periodic activation called Snake. In ONNX, Snake uses multiplication, sine, and power operators, and the upsampling uses four transposed convolutions; none of these have processors in the sixteen-processor set. The encoder's nearest-code search uses subtraction, power, reduction, and arg-max operators wrapped in dynamic-shape operators.

### 2.3 What the prior study reports

All statements in this subsection are reported by the `realm-quant` author and were read by us in that repository's notes and paper. We did not read the Cmajor repository directly for this report, and we did not re-run the converter. Each statement is therefore author-reported, and we label it that way wherever it is used.

- The shipped `Conv` processor sets its width padding to zero, never reads the ONNX `pads` attribute, and exposes no stride argument. The study reports that this would mis-lower all 151 convolutions in the codec, because every one of them has non-zero padding, a non-unit stride, or a non-unit dilation.
- The converter's data-type function accepts only float32, float64, int32, int64, and bool. The converter aborted on float16 initializers and on uint8 initializers, the latter appearing in the int8 variants of the models.
- The float16 variants add a single `Cast` node and leave the operator set otherwise unchanged. The int8 variants introduce `ConvInteger`, `DynamicQuantizeLinear`, `DequantizeLinear`, and `MatMulInteger`, none of which have a Cmajor processor.
- A literal expansion of 5.39 times was measured on codebook weights. The full-model figures of roughly 1.14 GB for the decoder and 487 MB for the encoder were extrapolated from that ratio, not measured by emitting the full models.
- The Cmajor toolchain binary did not start in the test environment, so no generated code was compiled and no WebAssembly was produced. Statements about what a processor addition would achieve are source-level readings.

### 2.4 Later work in the same repository

A later version of the `realm-quant` repository adds a Python workbench and a native bridge. The workbench produces float16 and dynamic int8 variants of an ONNX model and a ternary variant stored as packed two-bit symbols with per-output-channel scales, alongside a dequantized float model that a standard runtime can execute. The repository states that this is not native ternary arithmetic and that no speedup is claimed. The bridge lets a Cmajor program call ONNX Runtime through a C interface and does not translate ONNX operators into processors; its documentation states that the native code was not compiled where it was written. Neither addition modifies the Cmajor emitter or type system.

---

## 3. The Problem with Unstructured Review

A review of a quantization proposal has to answer a deceptively simple question: is this in scope for the emitter? Five distinct things hide inside that question, and a review that does not separate them is difficult to check.

1. **A data-type whitelist entry.** The converter decides which ONNX element types it will read. Adding float16 to that list is a change to a function.
2. **Emitter lowering logic.** After reading a tensor, the converter must write it into generated source. For a new type that means deciding how literals are rendered, how scales are attached, and how packed values are unpacked.
3. **Language-level types.** Whether the Cmajor language itself has, or can be given, a half-precision type, an eight-bit integer type, or a ternary type is a question about the language, not the script. Nothing in the material we have read establishes the answer in either direction.
4. **Quantized operator implementations.** Int8 graphs use operators that the sixteen-processor set lacks. Providing them is a separate body of work from the first three items.
5. **Validation of audio quality.** None of the preceding items says anything about whether a quantized codec still sounds right. That requires a metric, a reference, and an experiment.

A paragraph that says "ternary needs emitter changes" collapses these five. A reader cannot tell whether the author means one line in a whitelist or a multi-month project, and cannot tell which parts rest on evidence.

A second failure is inherited premises. A reviewer who is handed a prompt containing a false or unverified statement will often build on it. The first version of the prompt that this project used asserted that "Cmajor currently has no ternary, int8, or float16 types." The evidence supports a narrower statement, discussed in Section 5.

A third failure is silent repair. When a reviewer notices that a premise is wrong, the easy move is to correct it in passing and carry on. The correction then leaves no trace, and later readers cannot tell that the original premise was ever different. The schema is designed against these three failures: blurred layers, inherited premises, and silent repair.

---

## 4. Schema Design

The schema is a JSON Schema in draft-07 form. A review is a single JSON document with nine required top-level fields, listed in Appendix A.

### 4.1 Evidence classes

Every claim in the sections `known`, `proposed_unmeasured.claims`, and the primary blocker carries exactly one of the following classes. The class is an enumerated value, so the separation does not depend on a reader parsing prose.

- `known-source`: stated by a source read directly, such as a script or a source file.
- `known-measured`: stated by a measurement that was actually taken.
- `known-documented`: stated by documentation.
- `proposed-unmeasured`: a proposal for which no implementation or measurement is assumed.
- `requires-change`: a statement that some change would be required.
- `open-question`: a design decision that has not been made.
- `assumption`: something taken as given without support.
- `unverified`: a claim that could not be checked against the provided material.

In the final version of the schema, evidence classes are attached to claim sections only. Other sections are typed by their own structure: entries in the required-changes lists are changes by position, entries in the open-design-questions list are questions by position, and entries in the assumptions-and-corrections list carry a `kind` field. An earlier version of the review prompt told reviewers to tag every substantive statement with a class, which the schema did not support; the mismatch was found during review of this project's own code and resolved by amending the prompt, not the schema.

### 4.2 Prompt-premise corrections

A required array named `prompt_premise_corrections` records mistakes in the review prompt itself. Each entry has a premise, a correction, and a basis. This array exists separately from `assumptions_and_corrections`, which records mistakes in the material under review. The separation makes a prompt-level mistake visible as a prompt-level mistake. A reader who sees an entry here knows that the instruction the reviewer was given was itself wrong, and can discount any earlier review produced under that instruction.

### 4.3 The no-implementation guard

The field `proposed_unmeasured.no_implementation_assumed` is required and fixed to the value `true`. It cannot be set to false. Its purpose is to make the document assert, in a machine-checkable place, that nothing in the proposal section assumes an implementation, a benchmark, a prototype, or an audio-quality result. The guard is blunt. It does not stop a reviewer from smuggling an assumption into the text of a claim. What it does is make the opposite statement impossible to write in the designated field.

### 4.4 Per-type verdicts and the primary blocker

The scope verdict gives one value for each of the three types: `near-term-in-scope`, `out-of-scope`, or `insufficient-evidence`. It also requires a primary blocker, which is itself a claim with an evidence class and, where the class is a known class, a source. Requiring a single named blocker forces the reviewer to say what the main obstacle is, instead of listing many obstacles of unstated weight. The third verdict value matters: a review that cannot decide is allowed to say so, and the schema treats that as a legitimate outcome and not as a failure to answer.

### 4.5 Required changes with an extensibility flag

The required-changes section is divided into four lists: type-system changes, emitter-lowering changes, operator changes, and validation. Each entry names the types it applies to and carries a boolean field, `extensibility_verified`. The flag is true only if the material shows that the target (the type system, the emitter, or the processor set) can in fact be extended in the stated way. The field exists because a statement such as "add a half-precision type" quietly assumes that the language permits it. In the worked review in Section 8, every flag is false, because the material we reviewed does not show extensibility for any of the changes. The validator emits a warning for any true flag, on the principle that a claim of verified extensibility is the kind of claim most likely to be made without checking.

### 4.6 Open design questions

Open design questions are listed with an area drawn from a closed set: representation, packing, scale metadata, literal emission, operator coverage, validation methodology, verification environment, and other. The closed set is a prompt to the reviewer. A review that has no entry under validation methodology has, in effect, not said how any quality claim would be tested.

### 4.7 Verification limits

The `verification_limits` field is required and must contain at least one non-blank entry. It is where the reviewer records what could not be compiled, run, or measured. In the review of the `realm-quant` material, the dominant entry is that the Cmajor toolchain did not start, so every claim about lowering behavior is a reading of source and not an observation of output. Making the field mandatory prevents a review from presenting itself as more verified than it is by omission.

### 4.8 Non-blank strings

Every free-text field is constrained to contain at least one non-whitespace character. This closes a hole in an earlier version, in which an array of empty strings satisfied the requirement that verification limits be non-empty. The hole was found by probing the validator after it had been written, and it is the kind of defect that the harness in Section 7 is designed to catch systematically.

---

## 5. Premise Corrections That Motivated the Design

Three corrections arose while applying the first version of the review prompt to the `realm-quant` material. They are recorded here because they are the kind of error the schema is meant to surface.

**Correction 1: the type-support premise.** The first prompt stated that Cmajor currently has no ternary, int8, or float16 types. The material supports only the following: the converter's data-type function accepts a short list of element types that excludes those three, and the converter aborted on float16 and uint8 inputs. That is a statement about the converter. It is not a statement about the Cmajor language, which may or may not have such types and may or may not be extensible to add them. The corrected prompt separates what is known from source from what is unverified.

**Correction 2: the converter-versus-emitter distinction.** The first prompt distinguished adding a converter flag from changing the emitter. In this codebase the converter script is the emitter. The whitelist lives inside it, and the code that writes generated source is in the same file. The distinction imported from other toolchains does not apply here, and treating it as real would invite the reviewer to look for a second component that does not exist.

**Correction 3: the claim that ternary literals eliminate the expansion.** The prior notes say ternary weights, written as single-character literals, would eliminate the 5.39 times expansion entirely. This is an unmeasured proposal. Ternary weights need scale factors, which add literals. It is open whether values of negative one, zero, and positive one could be emitted as ordinary integer or floating-point literals with no new type, in which case the size benefit would not depend on a type extension. And the full-model sizes were extrapolated, not obtained by emitting full models.

The first two appear in the worked review as prompt-premise corrections. The third appears as an evidence-level correction, because the mistaken statement was in the material under review.

---

## 6. The Validator

### 6.1 Structure

The validator is a Python package named `schema_review`. It has no runtime dependencies beyond the standard library. It exposes a function that takes a parsed review and returns a list of errors and a list of warnings, and a command-line entry point that prints `OK` and exits with status 0 for a valid review, prints each error and exits with status 1 for an invalid one, and exits with status 2 when the input cannot be read or is not valid JSON. The bundled schema is stored inside the package, so a non-editable installation can find it.

Validation proceeds in two stages. If the first stage reports errors, the second stage does not run, because the semantic rules assume a document whose shape is correct.

### 6.2 Structural stage

The first stage is a small checker for the subset of draft-07 that the shipped schema uses: type, required, properties, additionalProperties, items, enum, const, minItems, minLength, pattern, and references into the definitions section. It is not a general implementation of JSON Schema. It resolves references by merging the referenced definition with any sibling keywords, which the shipped schema uses to attach a description to a shared definition.

A structural-only mode skips the second stage. It is intended for use with any schema other than the bundled one, because the semantic rules refer to the bundled schema's field names.

### 6.3 Semantic stage

The second stage applies rules that JSON Schema cannot express. They are summarized in Appendix C.

- The `known` section must be non-empty and may hold only known-class claims, each with a non-blank source.
- The `proposed_unmeasured.claims` section may hold only proposed-unmeasured claims.
- The primary blocker must be a known-class, unverified, or assumption claim. A blocker classed as a required change or a proposal is rejected, because the blocker is meant to describe the current state of things.
- A near-term-in-scope verdict for any type requires a known-class blocker.
- A near-term-in-scope verdict may not coexist with the phrase "out of scope" or "out-of-scope" in the summary or the bottom line. This is a heuristic based on substring matching.
- The bottom line must have between two and four sentences. Sentence counting is a regular-expression split that protects a short list of abbreviations; it is a heuristic.
- No entry may duplicate an earlier entry in the same section, after lower-casing and collapsing whitespace.
- Any entry in any required-changes list with a true extensibility flag produces a warning, not an error, prompting the reviewer to confirm that the material shows it.

### 6.4 Defects found and fixed during development

The first version of the validator passed its own ten unit tests and was nonetheless defective. Review found ten problems. Arrays of blank strings and whitespace-only sources satisfied the non-blank rules. An empty `known` section validated. A verdict could contradict its own summary. A required-change class could serve as a blocker. Duplicate claims were accepted. The extensibility warning fired for only one of four lists. The sentence counter split on abbreviations. The schema was located relative to the repository root, so a non-editable installation would have failed to find it. And the prompt and schema disagreed about which statements carry evidence classes. Each was fixed with a regression test, which took the suite from 10 tests to 19; the harness of Section 7 then brought it to 24.

### 6.5 Known weaknesses

The semantic rules are heuristics where they use text matching. A reviewer who wants to contradict a verdict without tripping the substring check can do so by using different words. The validator checks that a source is present and non-blank; it cannot check that the source supports the claim. A mislabeled claim passes if its label is permitted in its section. These weaknesses are properties of any form-based check, and the validator does not hide them.

---

## 7. Independent Execution and Validation

A validator written by the person who also wrote its tests can share that person's blind spots. If the author forgets that blank strings are a problem, the author's tests will not check for them. Section 6.4 describes exactly this: the first ten tests passed while ten defects remained. The harness in this section addresses the problem by deriving its expectations from a different source and by exercising the validator through a different interface.

### 7.1 Independence by construction

The harness is a single script, `scripts/independent_check.py`. It does not import any validator code. It reads the schema file as plain JSON and uses its own small walker, written separately from the validator's checker, to enumerate what the schema requires. It then builds modified copies of the reference review, writes each to a temporary file, and asks the validator to judge it by running the command-line interface in a subprocess and reading the exit status.

### 7.2 Clean-room execution

Before the mutation sweep, the harness copies only the package directory into a fresh temporary directory, changes to a different working directory, and runs the validator on the reference review with the path to the copied package supplied through the environment. A pass shows that the package does not depend on files outside itself, including the bundled schema, which ships inside the package. The harness runs the command twice and compares standard output, standard error, and exit status, to confirm that the run is deterministic.

### 7.3 Mutation operators

From the reference review, the harness generates mutants by applying the following operators at every position the schema describes.

- **Delete a required key.** Mandatory: the validator must reject.
- **Add an unexpected property** to any object that forbids extras. Mandatory.
- **Replace a value with a value of the wrong type.** Mandatory.
- **Replace an enumerated value with an invalid one.** Mandatory.
- **Violate a constant.** Mandatory.
- **Blank a string** constrained to be non-blank, using both the empty string and a string of spaces. Mandatory.
- **Empty an array** whose schema requires at least one item. Mandatory.
- **Empty an array** with no minimum-item constraint. Informational: the schema permits it.
- **Delete an optional key.** Informational: whether this is accepted depends on the semantic rules.

A mutant in a mandatory category that the validator accepts is a failure. A survivor in an informational category is reported but does not fail the run.

### 7.4 Results of the run

The run reported here executed under Python 3.12.3 on a Linux sandbox. The reference review was valid and the clean-room execution succeeded with status 0 and output `OK`. Two executions produced identical output, so the run was deterministic.

The harness generated 435 mutants in total. The validator rejected 428 of them. Of the 421 mutants in mandatory categories, it rejected all 421.

| Category | Mandatory | Mutants | Rejected |
|---|---|---:|---:|
| Delete a required key | yes | 96 | 96 |
| Add an unexpected property | yes | 37 | 37 |
| Wrong type | yes | 147 | 147 |
| Invalid enumerated value | yes | 36 | 36 |
| Violate a constant | yes | 1 | 1 |
| Blank string | yes | 92 | 92 |
| Empty array with a minimum | yes | 12 | 12 |
| Empty array, no minimum | no | 7 | 0 |
| Delete an optional key | no | 7 | 7 |

The seven survivors were all in the informational category of empty arrays without a minimum. They were the empty forms of the premise-corrections list, the proposed-claims list, the four required-changes lists, and the assumptions-and-corrections list. The schema permits each of these to be empty, which is a design decision: a review with no premise corrections is plausible, and a review with no assumptions is at least possible. The harness therefore reports them without failing. A different design could require at least one entry in some of these lists, and the harness would then classify the corresponding mutants as mandatory.

The harness also records SHA-256 digests of the schema, the reference review, and the validator source, so a later reader can tell whether their files are the ones exercised. The digests as built are in Appendix B.

### 7.5 What the harness shows

It shows that, for this schema and this reference review, the validator enforces every constraint the schema states in the categories above, at every position where the mutation operators apply. It shows that the package runs when copied alone into a clean directory and that its output is deterministic.

### 7.6 What the harness does not show

It does not show that the validator agrees with a general-purpose JSON Schema implementation on arbitrary schemas, because it was only run against the bundled schema. A comparison against the widely used reference library was not possible in the environment where this work was done, which had no network access and did not have that library installed.

It does not test the semantic rules independently. The mutation operators derive from the schema and the semantic rules are not in it; those rules are covered by unit tests written by the same author, and incidentally by some optional-key deletions. An independent test would need an oracle for them, and the only available one is their own description.

It does not show that mutants are representative of real reviewer mistakes. A reviewer who files a true claim under the wrong class, or cites a source that does not say what the review says, produces a document that neither the harness nor the validator can reject.

It does not run the continuous-integration workflow. The workflow file is included in the repository, and its steps correspond to the commands described here, but it has not been executed on a hosted runner.

---

## 8. The Worked Review

The reference review applies the schema to the `realm-quant` material. It is a test fixture and a demonstration. Its claims inherit every limit of their sources: they are author-reported, read from source, and not compiled.

### 8.1 Verdict

The review judges all three types out of scope for DAC-class graphs in the near term. It names one primary blocker: the shipped `Conv` processor ignores padding and has no stride argument, so every convolution in the codec would be lowered incorrectly whatever the weight type. The blocker is classed as known-source, with the `realm-quant` notes as its source. It is stated in terms of operator coverage, not types, and that is the central finding of the review. A quantized type would not unblock the codec because the codec is blocked before weights matter.

### 8.2 Per-type reading

**Float16.** The float16 variants add one `Cast` node and leave the operator set unchanged. The converter aborts on float16 initializers. If the language has a half-precision type, supporting it might be a small change; if not, the converter could promote float16 weights to float32, giving up the size benefit in generated source but not in the file. The review lists both as open questions. Float16 is the smallest change and is still blocked by the convolution problem.

**Int8.** The converter aborts on the unsigned eight-bit initializers, and the int8 graphs use operators with no Cmajor processors, so int8 needs a data-type change, literal rendering, and several new operators. The `realm-quant` data also show the int8 decoder file larger than the float16 one, which weakens the payload argument for int8 on this model.

**Ternary.** Ternary has no counterpart in the exports the study converted. The study's workbench later produced a ternary variant only as a dequantized float model, and the repository claims no native ternary arithmetic. For Cmajor, ternary would need a representation, scale metadata, an unpacking step, and a codec-quality validation. None of this exists in the material, and the review says so in its verification limits.

### 8.3 Required changes by layer

The review separates required changes into type-system, emitter-lowering, operator, and validation lists. The type-system list contains three entries, one per type. The emitter-lowering list contains the whitelist extension, literal rendering, and ternary unpacking with scales. The operator list contains the integer and quantization-related processors and, for every type, the convolution rewrite that precedes everything else. The validation list contains a single entry, a codec-quality evaluation with a defined metric and reference, noting that none exists. Every extensibility flag is false.

### 8.4 Open questions and limits

The review lists six open design questions across representation, packing, scale granularity, literal emission, validation methodology, and the verification environment. It lists three verification limits: the toolchain did not start, lowering behavior is source-read only, and no codec-quality metric was evaluated for any quantized variant. Its bottom line is three sentences long. It states that ternary weights are currently out of scope, that they require type-system and emitter changes and sit behind an unstarted convolution rewrite, and that float16 is the most plausible first step but is unmeasured inside Cmajor.

---

## 9. Analysis of the Technical Question

The schema is a method, and the review is an instance of it. It is worth stating plainly what the instance concludes and what it leaves open.

The conclusion that quantized types are out of scope is conditional on the codec and on the processor state described in the prior notes. It is not a statement that quantized types are a bad idea for Cmajor in general. A model that used only operators in the sixteen-processor set, with zero-padded unit-stride convolutions, would not meet the convolution blocker, and the question of types would then lead. The verdict is a verdict about this graph class.

The review also shows how the layers interact. Representation, operator coverage, and runtime are coupled: a compact representation is worthless if the operators that consume it are missing, and a complete operator set is of limited use if the weights are too large to deliver. The review identifies the convolution rewrite as a prerequisite for every type, which makes it the first change worth making if the compiler route is pursued at all. The bridge in Section 2.4 shows a different route: it delegates execution to a runtime, giving up weights-as-constants for a working path. Which route is preferable depends on goals, and the schema does not decide it.

---

## 10. Threats to Validity

**Source fidelity.** Every claim about Cmajor in the worked review rests on the author's notes. The Cmajor repository was not read for this report. If the notes misread the source, the review inherits the misreading. The evidence classes record that the claims are source-read by the notes' author, but they do not independently confirm it.

**Self-testing.** The validator, its unit tests, and the harness were written by the same project. The harness is independent of the validator's code but not of its authors' understanding of the schema. A misunderstanding shared by the schema's design and the harness's operators would not be caught.

**Heuristic rules.** The sentence counter and contradiction check use text matching and can be evaded or misfire.

**Single reference review.** Mutants derive from one reference review, so constructs it does not exercise produce none, and the counts depend on its shape.

**No reviewer study.** We did not measure whether reviewers using the schema produce more accurate or more consistent reviews than reviewers who do not. The claim that the structure helps is a design rationale, not a result.

**Subset checker.** The structural stage implements only the keywords the shipped schema uses; a schema using others would be silently under-checked, which is why structural-only mode suits only schemas within that subset.

---

## 11. Reproducibility

The repository contains the schema inside the package, the reference review as a test fixture, the validator, a unit-test module, the harness script, and a continuous-integration workflow file. To reproduce Section 7, install the package, run the unit tests with the standard-library runner, and run the harness script, optionally giving it a path for its JSON report. One unit test runs the harness and fails if any mandatory mutant survives. In the environment used here the 24-test suite passed in about 18 seconds, almost all of it the harness's subprocess launches. Anyone who modifies the schema, the reference review, or the validator should expect the mutation counts or digests to change and should compare against Appendix B before assuming a result here applies.

The harness exits with status 0 only if the clean-room run succeeded, the repeated run was identical, and every mandatory mutant was rejected. It writes a report containing the interpreter version, the platform, the SHA-256 digests described above, per-category counts, and the list of survivors.

---

## 12. What This Report Does Not Claim

It does not claim that float16, int8, or ternary weights work in Cmajor. It does not claim that any quantized variant of the codec preserves audio quality. It does not claim that the Cmajor language lacks, or can be extended with, any particular type. It does not claim that the schema improves review accuracy, that the validator matches a general JSON Schema implementation, or that the semantic rules are complete. It does not claim that the continuous-integration workflow has run. It claims that a particular validator rejected a systematically generated set of malformed reviews and accepted a reference review, and that this was established by a procedure that does not import the validator's code.

---

## 13. Future Work

Several steps would strengthen the result.

First, compare the structural stage with a general-purpose JSON Schema implementation on the bundled schema and a corpus of valid and invalid documents; differences would expose a bug in the subset checker or an unhonored keyword.

Second, build an oracle for the semantic rules that is independent of their implementation, for instance by having a second author write a specification from the prose and generate cases from it.

Third, read the Cmajor source directly and replace author-reported evidence with first-hand evidence; whether the type system can be extended could be settled by inspection.

Fourth, run the converter and, if the toolchain can be made to start, compile the decoder variants so that lowering claims become measured.

Fifth, run a reviewer study comparing the consistency of conclusions, and the rate of unsupported claims, with and without the schema. Finally, run the continuous-integration workflow on a hosted runner and record the result.

---

## Author Contributions

Contribution statements are to be supplied by the authors.

## Acknowledgments

This report builds on the `realm-quant` repository and its notes. The conclusions about Cmajor in this report are those of that repository's author as read by us, and any errors in reading them are ours.

## References

1. `ljrseals-dev/realm-quant`: `Paper.md`, `notes/cmajor-research.md`, and the later Python workbench and bridge documentation.
2. Cmajor, `cmajor-lang/cmajor`, release 1.0.3209, as described in reference 1. Not read directly for this report.
3. ONNX, the Open Neural Network Exchange format, an open model-interchange specification.
4. Descript Audio Codec, 24 kHz variant, as described in reference 1.

---

## Appendix A: Required Top-Level Fields

| Field | Content |
|---|---|
| `prompt_premise_corrections` | Mistakes in the review prompt, each with premise, correction, and basis |
| `scope_verdict` | Per-type verdicts, a primary blocker claim, and a summary paragraph |
| `known` | Claims in known classes, each with a source |
| `proposed_unmeasured` | The no-implementation flag, fixed true, and proposal claims |
| `required_changes` | Four lists, each entry naming its types and its extensibility flag |
| `open_design_questions` | Questions, each with an area from a closed set |
| `assumptions_and_corrections` | Entries of kind assumption, evidence-correction, or unverified |
| `verification_limits` | Non-empty list of non-blank strings |
| `bottom_line` | Two to four sentences |

## Appendix B: Digests as Built

SHA-256 digests recorded by the harness in the run reported in Section 7. Any edit to the named file changes its digest.

| File | SHA-256 |
|---|---|
| Schema | `ead8a22f53c1377bd876cd6b012f58da070fbad1eff62656a56d7f27f62b4b21` |
| Reference review | `861f4b8743ed4424baf731be42d3fcb5b447875bb13c629c9c7e3fc68f7dfa4e` |
| `validate.py` | `4e04e48adad11d366b35bd0a11d6855bece41308342794287ac9a9cc83dcdf4d` |

## Appendix C: Semantic Rules at a Glance

| Rule | Kind |
|---|---|
| `known` non-empty, known-class claims only, each with a source | error |
| `proposed_unmeasured.claims` hold proposed-unmeasured claims only | error |
| Primary blocker is known-class, unverified, or assumption | error |
| Near-term-in-scope verdict needs a known-class blocker | error |
| Near-term-in-scope verdict cannot coexist with "out of scope" wording | error, heuristic |
| Bottom line has two to four sentences | error, heuristic |
| No duplicate entries within a section | error |
| Extensibility flag true in any required-changes entry | warning |
