# Independent verification record — Realm-Quant / Cmajor 1.0.3209

## Result

The review JSON was checked against the cited primary artifacts in this session. Every known-source claim has a direct source URL and is marked `second_hand:false` / `source_verified_by:"self"`. The one historical-status claim is explicitly second-hand from the Realm-Quant research note and is not presented as a measurement made in this session.

## Revision pin

- Repository: `cmajor-lang/cmajor`
- Tag: `1.0.3209`
- Tag target commit: `2d6ceb641b6f121b44d64e6f0cd134dbabfaa122`
- Research repository: `ljrseals-dev/realm-quant`, branch `main`

## Claim-by-claim source checks

| Review claim | Primary source inspected | Check |
|---|---|---|
| ONNX datatype whitelist contains FLOAT, DOUBLE, INT64, INT32, BOOL | `tools/onnx/onnxToCmajor.py`, lines 75–86 | Confirmed; unsupported values fall through to `assert(False)`. |
| Cmajor primitive type enum omits INT8, FLOAT16, and ternary | `modules/compiler/src/AST/cmaj_AST_Properties.h`, lines 369–370 | Confirmed by the complete enum declaration. |
| Primitive parser omits INT8 and FLOAT16 | `modules/compiler/src/AST/cmaj_Parser.h`, lines 2165–2175 | Confirmed by the inspected primitive-type dispatch list. |
| Native primitive packed storage sizes | `modules/compiler/src/AST/cmaj_AST_Classes_Types.h`, lines 207–221 | Confirmed: int32/float32 are 4 bytes; int64/float64 are 8 bytes; no int8/float16 primitive cases. |
| Generic ONNX operator name emission | `tools/onnx/onnxToCmajor.py`, lines 249–260 | Confirmed: generated processor reference uses the ONNX `op_type` string. |
| Conv lacks pads/strides parameters and fixes widthPadding to zero | `tools/onnx/patchTemplate/onnx.cmajor`, lines 187–245 | Confirmed in the shipped processor signature and implementation. |

## Second-hand material retained as second-hand

`notes/cmajor-research.md` reports historical conversion attempts, a Cmajor launch failure, and the absence of compile-checked shims/WASM. Those historical run claims were not repeated here. The review records only the note's documented status and does not relabel it as a current-session measurement.

## Downgrades and scope controls

- No claim of INT8, FLOAT16, or ternary implementation was accepted.
- No benchmark or audio-quality claim was accepted as a current-session measurement.
- The prompt's language-level “unverified” qualifier was corrected narrowly for the inspected 1.0.3209 primitive enum/parser: native INT8 and FLOAT16 primitive types are absent. User-defined/packed emulation remains unverified.
- All `required_changes` entries retain `extensibility_verified:false`; inspected source shows missing extension points/types, not a demonstrated working extension mechanism.

## Validation limits

- JSON syntax was parsed after retrieval/assembly.
- A dedicated Draft-07 schema validator was not available/run in this verification pass.
- No compiler, generated patch, runtime, ONNX model, or audio codec was executed.
- This is a source-verification result, not an implementation or performance result.
