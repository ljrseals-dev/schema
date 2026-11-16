# Evidence-Classed Schema Review and Quantized ONNX-to-Cmajor Fallback

**A research-grounded review validator, deterministic capability policy, and executable numeric fallback library.**

This repository addresses two connected engineering problems:

1. **How do we keep technical reviews honest?** A review must distinguish source facts, measurements, documentation, proposals, assumptions, and unknowns—and make those distinctions machine-checkable.
2. **How do we decide whether a quantized ONNX model can safely move through a Cmajor adapter?** Recognizing a dtype is not enough. Initializer decoding, representation, operator coverage, code generation, compilation, and runtime parity are separate obligations.

The repository contains working software for the review/validation layer, a capability-driven fallback classifier, and executable FLOAT16, INT8, and ternary transforms over caller-decoded tensor payloads. It does **not** yet contain a complete ONNX-to-Cmajor converter.

> **Core engineering principle:** Never turn missing evidence into assumed capability, and never report a partial conversion as a successful model conversion.

---

## At a glance

| Area | What exists | What it does not establish |
|---|---|---|
| Evidence-classed review | Draft-07 review schema, strict JSON loading, structural and semantic validation | Whether cited sources actually prove a review claim |
| Independent validator check | Clean-room CLI execution and differential comparison with jsonschema.Draft7Validator | General Draft-07 certification or independent validation of every semantic heuristic |
| Capability classifier | Deterministic native / fallback / blocked / unsupported decisions | That adapter probes or evidence are truthful; the classifier consumes them |
| FLOAT16 fallback | Binary16 word decoding and promotion to binary32-valued Python floats | Cmajor arithmetic parity or preserved FLOAT16 rounding at every graph operator |
| INT8 fallback | Signed INT8 dequantization with scalar or per-axis parameters | Correct extraction of quantization annotations from an actual ONNX graph |
| Ternary fallback | Explicit two-bit decoding with codebook, bit order, reserved-symbol and padding checks | A universal ternary encoding or automatic discovery of a model's encoding |
| End-to-end adapter | Explicit integration contract and fail-closed policy | ONNX protobuf ingestion, graph rewriting, Cmajor emission, compilation, runtime parity, or codec-quality validation |

### Plain-English interpretation

Think of this repository as a safety-conscious bridge inspector, not yet the completed bridge. It can check whether the paperwork is internally consistent, perform several carefully specified number conversions, and decide whether the evidence presented is sufficient to permit a path. It does not yet drive a real ONNX model all the way through the Cmajor compiler and target runtime.

---

## 1. System architecture

The repository has three logical layers. The first checks claims. The second decides which implementation path is admissible. The third performs a small set of numeric transforms and compares them with separate reference calculations.

~~~mermaid
flowchart TD
    A["Technical proposal / review"] --> B["Review JSON"]
    B --> C["Draft-07 structural validation"]
    C --> D["Semantic consistency rules"]
    D --> E["Review verdict + diagnostics"]

    F["Adapter capability probes"] --> G["Feature/fallback report"]
    G --> H["Deterministic classifier"]
    H --> I{"Decision"}
    I -->|native| J["Native path is eligible"]
    I -->|fallback| K["Named fallback is eligible"]
    I -->|blocked| L["Collect missing evidence"]
    I -->|unsupported| M["Reject requested feature"]

    N["Decoded initializer payload"] --> O["Fallback transform"]
    O --> P["Independent standard-library reference"]
    P --> Q["Numerical + metadata report"]
    Q --> K

    J --> R["External adapter obligations"]
    K --> R
    R --> S["Cmajor generation, compilation, runtime parity"]
~~~

**Important boundary:** The diagram shows the intended adapter decision flow. The package implements the review validator, classifier, and decoded-payload transforms. ONNX inspection, initializer extraction, Cmajor generation, compilation, and target execution remain integration work; they are not silently supplied by the classifier.

### 1.1 The evidence pipeline

A review is not considered trustworthy merely because it is valid JSON. Validation occurs in layers:

~~~mermaid
flowchart TD
    A["Read review JSON"] --> B["Strict JSON parser"]
    B --> C{"Parse succeeded?"}
    C -->|No| X["Exit 2: input/schema error"]
    C -->|Yes| D["Structural schema checks"]
    D --> E{"Structure valid?"}
    E -->|No| Y["Exit 1: validation errors"]
    E -->|Yes| F["Semantic review checks"]
    F --> G{"Semantic checks pass?"}
    G -->|No| Y
    G -->|Yes| H["Exit 0: OK"]
~~~

The strict parser rejects duplicate object keys, non-standard NaN/Infinity values, and numeric values that overflow the supported finite range. Structural validation checks the documented subset of JSON Schema Draft 7 used by the bundled review schema. Semantic validation then checks project-specific rules that are awkward or impossible to express in that schema subset.

**Layman's version:** First make sure the file can be read. Then make sure it has the required shape. Finally, check that its claims do not contradict its own verdict or misuse the evidence labels.

---

## 2. Evidence-classed review system

### 2.1 Why evidence classes exist

Engineering reviews routinely mix statements with very different epistemic status:

- a fact observed directly in source code;
- a result produced by an actual test or measurement;
- a statement made by documentation;
- a proposed design that has not been implemented;
- an assumption;
- an item that could not be verified.

If these are written in one paragraph without labels, a reader can mistake a proposal for a result. This repository encodes evidence classes in a JSON schema so that the review's structure makes those distinctions visible and checkable.

| Class | Meaning | Example |
|---|---|---|
| known-source | Directly established by reading a primary source | A function contains a particular dtype whitelist |
| known-measured | Established by a run or measurement in the review session | A recorded test reports a measured error |
| known-documented | Explicitly stated by documentation | A versioned manual specifies a behavior |
| proposed-unmeasured | Intended or predicted behavior not yet measured | A proposed conversion may reduce source size |
| assumption | An unsupported premise being used provisionally | A workload is assumed to use a particular tensor layout |
| unverified | Evidence is missing, inaccessible, or conflicting | Compiler support could not be established |

The schema also separates **prompt-premise corrections** from **evidence corrections**. This matters because a flawed instruction and a flawed source interpretation are different failure modes and should not be merged into one vague correction list.

### 2.2 What the review schema enforces

The bundled review schema and semantic validator enforce, among other rules:

- known claims must use a known-* class and carry a source reference;
- proposed claims must be classified as proposed-unmeasured;
- the no-implementation guard must remain true;
- the primary blocker must use an allowed evidence class;
- a near-term-in-scope verdict requires a known-evidence blocker assessment;
- selected scope contradictions are rejected;
- duplicate entries and blank strings are rejected;
- the bottom line must contain two to four sentences;
- verification limits must be present and non-empty;
- extensibility_verified = true produces a warning for human review.

The semantic scope and sentence checks are deliberately described as heuristics. They are not a formal proof that arbitrary prose is logically consistent.

### 2.3 Validator implementation boundary

The runtime validator has no third-party dependencies. Its structural checker supports the keywords used by the bundled schema: type, required, properties, additionalProperties, items, enum, const, minItems, minLength, pattern, and local $ref references.

It is **not** a general-purpose JSON Schema implementation. Unsupported schema keywords are reported rather than silently ignored. The optional test dependency jsonschema supplies an independent Draft-07 implementation for differential checking.

---

## 3. ONNX-to-Cmajor capability and fallback model

### 3.1 The key distinction: recognition is not execution

A converter recognizing an ONNX data type proves only that it can identify the type. It does not prove that it can decode the initializer, represent the values, lower the operations, generate valid Cmajor, compile the generated program, or reproduce the reference runtime's results.

The classifier therefore treats these as separate gates.

~~~mermaid
flowchart LR
    A["ONNX dtype recognized"] --> B["Initializer decoded"]
    B --> C["Representation supported"]
    C --> D["Emitter lowering supported"]
    D --> E["Operator coverage verified"]
    E --> F["Generated Cmajor compiles"]
    F --> G["Runtime parity verified"]
    G --> H["Native path eligible"]
~~~

**Plain English:** Knowing what kind of number is in a file is like recognizing a part on a workbench. It does not prove the part fits, that the machine can assemble it, or that the finished machine works.

### 3.2 Four decisions

Each requested feature is classified independently:

| Decision | Exact meaning | Required behavior |
|---|---|---|
| native | Every native capability gate is explicitly true and has evidence | Native emission may be selected |
| fallback | A named transform passes all fallback-specific guards and shared integration gates | Use only that validated transform |
| blocked | A required capability or its evidence is unknown or missing | Stop and collect the missing evidence |
| unsupported | A required capability is explicitly false and no eligible fallback exists | Reject the feature with a diagnostic |

Unknown is not the same as false. However, unknown is never permission to emit.

The model is ready only when **every** requested feature is either native or fallback. One blocked or unsupported feature makes the complete model decision blocked. This prevents a partially converted graph from being advertised as a complete conversion.

### 3.3 Native gates

The classifier requires all seven native gates:

1. dtype_recognized
2. initializer_decoded
3. representation_supported
4. emitter_lowering_supported
5. operator_coverage_verified
6. codegen_compiles
7. runtime_parity_verified

Every gate must be explicitly true, and every gate must have a non-empty evidence reference. A boolean without provenance is not sufficient.

### 3.4 Fallback gates

A fallback may be considered when the native representation or emitter-lowering path is unavailable. It does **not** bypass the rest of the integration contract. It still requires:

- dtype recognition and correct initializer decoding;
- verified operator coverage;
- successful code generation;
- runtime parity evidence;
- the exact expected transform for the requested feature;
- preserved metadata;
- a successful numerical validation result;
- non-empty fallback evidence;
- finite, non-negative error and tolerance values, with measured error no greater than tolerance;
- explicit ternary encoding verification when the feature is ternary.

The classifier evaluates the report supplied by the adapter. It does not run the probes itself, inspect source repositories, or independently prove that evidence identifiers are truthful.

---

## 4. Executable fallback transforms

The package exposes schema_review.execute_and_validate. It accepts **already-decoded initializer payloads**, performs a transform, compares the result with a separately implemented standard-library reference calculation, and returns an auditable result object.

~~~mermaid
flowchart TD
    A["Caller supplies decoded payload + metadata"] --> B{"Feature"}
    B -->|FLOAT16| C["Decode IEEE-754 binary16 words"]
    B -->|INT8| D["Apply signed quantization formula"]
    B -->|Ternary| E["Decode explicit two-bit symbols"]
    C --> F["Independent reference calculation"]
    D --> F
    E --> F
    F --> G["Compare output values"]
    G --> H["Measure absolute and relative error"]
    H --> I["Check supplied metadata snapshot"]
    I --> J["Return validation report"]
~~~

### 4.1 FLOAT16 to FLOAT32

The implementation accepts raw unsigned 16-bit words representing IEEE-754 binary16 values and promotes finite values to Python floats representing binary32-compatible values.

- Tensor shape must be a non-empty list of positive integer dimensions.
- The number of words must equal the product of the dimensions.
- Each word must be in the unsigned 16-bit range.
- NaN and infinity are rejected.

A second implementation decodes the sign, exponent, and fraction fields directly. The production path uses Python's binary16 struct conversion. Comparing these distinct calculations is useful for catching implementation mistakes in the decoding logic.

**Engineering caution:** Promotion preserves the represented finite value, but it does not automatically preserve every graph's operation-level rounding behavior. If the source model rounds intermediate operations to FLOAT16, promoting only its stored weights can change model semantics. That must be tested at the operator and graph levels.

### 4.2 INT8 dequantization

The signed INT8 transform uses the standard affine form:

\[
x_{\mathrm{real}} = (q - z)\,s
\]

where q is the stored signed integer, z is the zero point, and s is the positive scale.

The implementation supports scalar parameters and per-axis scale/zero-point arrays. For per-axis conversion it validates the axis, normalizes negative axes, checks parameter lengths, and maps each flattened value to its coordinate along the quantization axis.

It rejects values outside [-128, 127], non-integer zero points, invalid axes, non-finite parameters, and non-positive scales.

**Engineering caution:** The transform can only be as correct as the metadata supplied to it. The current package does not read ONNX protobufs to discover whether a tensor is per-tensor or per-axis quantized. The ONNX adapter must extract the exact scale, zero point, axis, shape, signedness, and broadcasting semantics.

### 4.3 Ternary two-bit decoding

Three logical symbols are stored using two-bit codes. Four code slots fit in one byte. Because two bits encode four possible values but ternary has only three, one code is reserved.

The decoder requires:

- an explicit mapping for codes 0, 1, and 2;
- three distinct finite codebook values;
- an explicit bit order (lsb or msb);
- an explicit tensor shape and value count;
- exactly the required number of packed bytes;
- the reserved code never to appear in active values;
- all unused padding bits to be zero.

It does **not** assume that ternary means {-1, 0, +1}. The caller must provide the codebook. It also does not infer the packing convention from a model name or guess from a byte sequence.

**Plain English:** The decoder will not guess what the bits mean. You must tell it how each code maps to a number and how the bits are arranged. If the data contradict those instructions, it rejects the payload.

### 4.4 What the numerical report means

The result includes:

- transformed output values and shape;
- a copy of the metadata supplied by the caller;
- a metadata-preservation flag;
- maximum absolute and relative error;
- the declared tolerance;
- a validation_passed boolean;
- evidence identifiers naming the transform and reference calculation.

The metadata check is a pass-through snapshot of caller-supplied metadata. It is **not** proof that metadata was correctly extracted from an ONNX file or that every ONNX annotation was preserved through graph rewriting.

The two calculations are independently written algorithms within the same package. They are not an external oracle: they share the same process and input payload. Production acceptance should additionally compare against ONNX Runtime or another separately versioned reference implementation, and should retain per-tensor and end-to-end metrics.

---

## 5. Numerical and decision acceptance

The fallback model uses a scalar minimum gate:

\[
\text{measured error} \leq \text{declared tolerance}
\]

This is a deterministic admission check, not a complete numerical-quality methodology. A production report should also preserve:

- the exact model and input corpus;
- reference runtime and version;
- per-tensor and per-operator error;
- maximum absolute and relative error;
- worst-case tensor and output location;
- the tolerance rationale and downstream task requirements;
- reproducible command, model hash, and artifact identifiers.

Tolerance must be selected from an acceptance requirement, not adjusted after a failing result merely to make the test pass.

For audio models, numerical parity is also not identical to perceptual equivalence. A production codec evaluation needs an explicit audio-quality protocol and appropriate objective and/or listening tests. This repository reports no new codec-quality measurement.

---

## 6. Engineering diagrams: end-to-end target and failure behavior

### 6.1 Intended adapter pipeline

The following diagram distinguishes implemented package components from future integration work.

~~~mermaid
flowchart TD
    A["ONNX model file"] --> B["ONNX protobuf inspection"]
    B --> C["Initializer + quantization metadata extraction"]
    C --> D["Operator, shape and graph-consumer inventory"]
    D --> E["Capability probes with evidence"]
    E --> F["Feature/fallback classifier"]
    F --> G{"All requested features ready?"}
    G -->|No| H["Reject model; report blocked/unsupported features"]
    G -->|Yes| I["Apply selected native/fallback transforms"]
    I --> J["Generate Cmajor source"]
    J --> K["Compile with pinned Cmajor toolchain"]
    K --> L["Execute target program"]
    L --> M["Compare against independent reference"]
    M --> N{"Parity and tolerances pass?"}
    N -->|No| O["Reject conversion and retain diagnostics"]
    N -->|Yes| P["Emit conversion manifest + verified result"]
~~~

**Implemented today:** the classifier, the three decoded-payload transforms, and their package-level reference comparisons.

**Not implemented by this repository:** the ONNX file reader/extractor, graph rewrite, Cmajor emitter, compiler invocation, target runtime, and full-model parity harness. Those boxes are the next integration milestones, not hidden features.

### 6.2 Fail-closed rules

| Condition | Required result |
|---|---|
| Capability is unknown | blocked; name the missing probe |
| Capability is explicitly false and no eligible fallback exists | unsupported |
| Fallback transform does not match the feature | Reject fallback |
| Ternary encoding is not verified | Reject fallback |
| Quantization metadata is missing or not preserved | Reject fallback |
| Numerical validation fails or error exceeds tolerance | Reject fallback |
| Operator coverage is not verified | Do not emit |
| Generated Cmajor fails compilation | Reject the model |
| Runtime parity fails | Reject the model |
| Only part of the requested model converts | Do not report complete success |

The system must not silently discard quantization metadata, invent a codebook, clamp invalid scales, substitute zero/random weights, or convert only the easy tensors while marking the model successful.

---

## 7. Test strategy and observed results

### 7.1 Test layers

The repository's tests cover four broad categories:

1. **Review validator:** schema structure, semantic rules, strict JSON parsing, CLI exit behavior, and regressions.
2. **Independent structural validation:** clean-room package execution, deterministic repeated CLI output, schema-derived mutations, and comparison against jsonschema.Draft7Validator.
3. **Feature classifier:** native/fallback gates, evidence completeness, unknown capabilities, incorrect transforms, metadata failures, tolerance failures, and all-or-nothing model readiness.
4. **Fallback runtime:** FLOAT16 decoding, INT8 scalar/per-axis dequantization, ternary LSB/MSB decoding, malformed inputs, reserved symbols, padding, and fail-closed behavior.

### 7.2 Hosted CI result

The integrated fallback branch has a recorded successful GitHub Actions run. The observed run completed all eight substantive workflow steps:

- checkout;
- Python setup;
- editable installation with the test extra;
- unit tests;
- bundled fixture validation;
- independent-validation artifact upload;
- non-editable installation;
- schema lookup from outside the repository working directory.

**Hosted CI: PASSED**

| Measurement | Observed result |
|---|---:|
| Unit tests | **90 passed** |
| Unit-test duration | **13.369 seconds** |
| Independent mutation cases | **482** |
| Cases invalid under Draft-07 oracle | **468** |
| Invalid cases rejected by project CLI | **468** |
| Oracle/CLI disagreements | **0** |
| Mandatory invalid cases accepted | **0** |
| Clean-room CLI execution | **Passed** |
| Repeated execution deterministic | **Yes** |
| Bundled fixture | **Passed** |
| Non-editable install/schema lookup | **Passed** |

The mutation report classifies 14 generated cases as valid optional-field/empty-array mutations. Those are not false negatives: the schema permits those forms. The remaining 468 cases are invalid according to the independent Draft-07 oracle, and the CLI rejected all 468. The report records no disagreements.

**Scope of this result:** this is strong evidence that the bundled structural checker agrees with jsonschema on the generated mutation set derived from the bundled schema and fixture. It is not exhaustive Draft-07 certification, does not independently validate every semantic heuristic, and does not prove that any cited engineering claim is true.

### 7.3 Historical paper result versus current CI

The paper in paper/PAPER_(2).md reports an earlier harness snapshot: 435 generated mutations, 421 mandatory-invalid mutations all rejected, and seven accepted empty-array cases that the schema deliberately permitted. That result belongs to the paper's recorded version.

The current integrated branch has a later harness that adds differential comparison with jsonschema.Draft7Validator; its hosted report is the 482-case result above. Do not combine the historical and current counts into one experiment.

The paper also records an earlier 24-test suite. That is a historical result, not the current hosted suite. The current CI log records 90 passing tests.

### 7.4 What the tests do not establish

- They do not test the ONNX parser or real initializer extraction.
- They do not compile generated Cmajor, because this repository does not generate it.
- They do not execute a converted ONNX graph in Cmajor.
- They do not compare end-to-end model outputs against ONNX Runtime.
- They do not measure codec latency, memory use, source size, or audio quality.
- They do not establish reviewer agreement or prove that the schema improves human judgment.

---

## 8. Reproducible use

### Requirements

- Python >=3.10
- Runtime: Python standard library only
- Test extra: jsonschema>=4.20,<5

### Install the runtime package

~~~bash
python -m pip install .
~~~

### Validate the worked review

~~~bash
python -m schema_review examples/review_example.json
~~~

CLI exit codes:

| Exit code | Meaning |
|---:|---|
| 0 | Review passed validation |
| 1 | Validation errors |
| 2 | Input or schema could not be read or parsed |

### Install test dependencies and run all tests

~~~bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
~~~

### Run the independent differential harness

~~~bash
python scripts/independent_check.py validation_report.json
~~~

The harness reads the bundled schema and fixture, checks the schema with jsonschema.Draft7Validator, copies the package into a temporary clean-room directory, invokes the public CLI from an unrelated working directory twice, and compares structural outcomes across generated mutations. It writes a JSON report containing environment details, file hashes, counts, disagreements, and survivors.

For a non-bundled schema, the CLI supports:

~~~bash
python -m schema_review review.json --schema path/to/schema.json --structural-only
~~~

The bundled semantic rules are specific to the bundled review schema and are intentionally skipped in structural-only mode.

### Python API

~~~python
from schema_review import (
    classify_feature,
    classify_model,
    execute_and_validate,
    validate_review,
)

# Run one already-decoded fallback transform.
report = execute_and_validate(
    "int8",
    {
        "values": [-4, 0, 8, 12],
        "shape": [2, 2],
        "scale": [0.25, 0.5],
        "zero_point": [0, 0],
        "axis": 0,
    },
    metadata={"quantization": "caller-supplied"},
    tolerance=0.0,
)

assert report["validation_passed"]
~~~

The example demonstrates the package-level API only. It is not an ONNX file conversion example.

---

## 9. Repository map

| Path | Responsibility |
|---|---|
| src/schema_review/__init__.py | Public package exports and version |
| src/schema_review/__main__.py | Review-validation CLI |
| src/schema_review/validate.py | Strict JSON loading, structural subset checker, semantic rules |
| src/schema_review/feature_fallback.py | Deterministic capability and fallback classifier |
| src/schema_review/fallback_runtime.py | Executable FLOAT16, INT8 and ternary transforms plus reference calculations |
| src/schema_review/data/quantized_emitter_review.schema.json | Draft-07 review-output contract |
| src/schema_review/data/feature_fallback.schema.json | Feature/fallback report contract |
| prompts/master_prompt.md | Evidence-classed review instructions and adapter policy |
| examples/review_example.json | Worked evidence-classed review |
| docs/FEATURE_FALLBACK_MODEL.md | Detailed fallback policy and integration requirements |
| scripts/independent_check.py | Clean-room, schema-derived differential harness |
| tests/ | Validator, classifier, transform, malformed-input and regression tests |
| tests/fixtures/review_example.json | Test copy of the worked review fixture |
| .github/workflows/ci.yml | Hosted CI workflow |
| INTEGRATION.md | Validator/harness execution notes and boundaries |
| Paper/Paper_(1).md | Earlier technical note on evidence-classed review |
| paper/PAPER_(2).md | Expanded research report on the schema, validator and independent harness |
| CHANGELOG.md | Historical package change notes |
| LICENSE | MIT license |

---

## 10. Research paper and quotation

The repository's expanded report is:

**“Evidence-Classed Review of Quantized-Type Proposals for the Cmajor ONNX Emitter: A Schema, a Validator, and an Independent Validation Harness”**

Authors: Lewis Seals and Angela La'Vette Seals  
DOI: https://zenodo.org/records/24267777  
Repository copy: paper/PAPER_(2).md

The earlier technical note is retained at Paper/Paper_(1).md. It documents the initial schema/validator rationale and explicitly reports no new measurements.

> “The harness establishes that the validator enforces the schema it ships with.”

— PAPER_(2).md, Section 7.5, “What the harness shows”

### Interpretation of the quotation

The statement is deliberately narrow. It means the independent harness tested whether the validator enforced the constraints encoded in the bundled schema. It does not mean the validator can prove the truth of a cited claim, establish that Cmajor supports a dtype, or certify an ONNX-to-Cmajor conversion.

The paper's central methodological contribution is not a claim that quantized audio conversion has been solved. It is a way to keep evidence, proposals, and unknowns separate—and to test the validator's structural behavior without importing the validator's own implementation into the harness.

---

## 11. Limitations and next engineering milestones

The current package is a useful foundation, but a complete adapter requires a further sequence of independently testable components.

| Milestone | Required implementation | Acceptance evidence |
|---|---|---|
| 1. ONNX ingestion | Parse real ONNX protobufs and inspect IR/opset metadata | Versioned model fixtures and parser tests |
| 2. Initializer extraction | Decode tensor payloads, shapes, dtypes and quantization annotations | Byte-for-byte/source metadata checks |
| 3. Operator inventory | Resolve graph nodes, domains, opsets, shapes and required operators | Coverage manifest with explicit unknowns |
| 4. Capability probes | Inspect the pinned Cmajor emitter/compiler/runtime | Source references and captured probe results |
| 5. Fallback integration | Connect extracted payloads to the existing transforms | ONNX Runtime differential tests |
| 6. Graph lowering | Generate Cmajor for supported operators and layouts | Golden generated source and compiler results |
| 7. Runtime parity | Execute generated programs and compare outputs | Per-tensor and end-to-end error reports |
| 8. Audio validation | Evaluate decoded audio against a declared reference protocol | Reproducible quality, latency and memory measurements |
| 9. Conversion manifest | Record selected paths, transforms, hashes, compiler version and evidence | Machine-readable, reproducible conversion record |

### Final engineering rule

A successful tensor transform is not a successful model conversion. A successful source-generation step is not a successful compilation. A successful compilation is not runtime parity. And numerical parity alone is not proof of perceptual audio quality.

Each boundary needs its own test, evidence artifact, and explicit pass/fail result.

---

## License

MIT. See LICENSE.
