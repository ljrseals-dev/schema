# ONNX-to-Cmajor feature and fallback model

## Purpose

The adapter makes a per-feature decision before code generation. It must not
confuse ONNX parsing with Cmajor support: recognizing a tensor dtype is only
the first gate. The model classifies each requested feature independently as
`native`, `fallback`, `blocked`, or `unsupported`.

This is a capability policy and decision function. It is not an ONNX parser,
a Cmajor compiler, a dequantizer, or proof that any path is implemented. The
adapter must populate the report from actual probes and validation artifacts.

## Decision contract

1. **Native** only when every native gate is explicitly true and each gate has
   a non-empty evidence reference:
   - dtype recognized;
   - initializer decoded correctly;
   - representation supported;
   - emitter lowering supported;
   - all required operators covered;
   - generated Cmajor compiles;
   - runtime parity is verified.
2. **Fallback** only when the native path is unavailable and the exact
   feature-specific transform is enabled, evidence-backed, metadata-preserving,
   and passes numerical validation. The fallback still requires initializer
   decoding, operator coverage, successful code generation, and runtime parity.
3. **Blocked** when a required capability is unknown, missing, or lacks
   evidence. Unknown is not false, but it is never permission to emit.
4. **Unsupported** when a required capability is explicitly false and no
   eligible fallback exists. Reject conversion with a diagnostic; do not
   silently change the model.

A model is ready only when every requested feature resolves to native or
fallback. The adapter must fail closed if any requested feature is blocked or
unsupported. A partial result must be explicit and must not be presented as a
complete converted model.

## Per-type fallback policy

| ONNX weight type | Candidate fallback | Mandatory safeguards |
|---|---|---|
| FLOAT16 | Promote stored values to FLOAT32 | Confirm initializer decoding; test whether promotion changes operation-level rounding; verify operator behavior and output tolerances |
| INT8 | Dequantize to FLOAT32 | Preserve scale, zero point, axis, broadcasting, signedness and quantization scheme; test per-tensor and per-axis cases independently |
| Ternary | Decode to FLOAT32 only with a verified encoding | Require a documented codebook, packing/bit order, padding rules and scale semantics; never assume {-1, 0, +1} or infer a packing layout |

These are candidate conversion paths, not claims that the current Cmajor emitter
supports them. No fallback is automatically safe merely because the target
representation is FLOAT32.

### FLOAT16

Every finite FLOAT16 value can be represented in FLOAT32, but that fact alone
does not establish equivalent model behavior. Promotion can change where
rounding occurs in the graph. Test weights and intermediate arithmetic
separately; document whether the source graph requires FLOAT16 rounding at
operator boundaries. Reject unsupported NaN/Infinity handling rather than
silently normalizing it.

### INT8

Dequantization must use the ONNX quantization parameters exactly. Preserve
signedness, scale tensor, zero-point tensor, quantization axis, tensor shape,
and broadcasting semantics. A per-axis quantized initializer must not be
treated as per-tensor. Check overflow and zero/invalid scales. Quantization
metadata loss is a hard failure.

### Ternary

Ternary has no universal ONNX packing convention that this adapter may safely
guess. Require explicit encoding and decoding evidence: symbol/codebook
mapping, bit packing, bit order, block boundaries, padding, scale and any
zero-point convention. Without these, mark the feature blocked or unsupported;
do not fabricate a decoder or silently reinterpret packed bytes.

## Numerical acceptance

For every fallback, record:
- the reference implementation and exact model/input set;
- absolute and relative error metrics, including the aggregation method;
- configured tolerance and measured error;
- worst-case tensor/operator and output location;
- whether the tolerance is suitable for the downstream audio task;
- test corpus and reproducible command/artifact identifiers.

The report currently uses a scalar `measured_error <= error_tolerance` guard as
a minimum deterministic gate. A production adapter should additionally retain
per-tensor and end-to-end metrics; one scalar must not hide localized failures.
Tolerance must be selected from an explicit acceptance requirement, not tuned
after seeing a failing result.

## Required implementation sequence

1. Build an ONNX inspection pass that inventories initializers, dtypes,
   quantization metadata, shapes, operator inputs and graph consumers.
2. Build a Cmajor capability manifest keyed by dtype, operator, layout and
   target/compiler version. Unknown entries remain unknown.
3. Implement native lowering only for entries with source-backed support.
4. Implement each fallback as an explicit transform with no implicit casts.
5. Add deterministic classifier/report generation using
   `schema_review.feature_fallback`.
6. Add golden numerical tests against ONNX Runtime (or another declared
   reference), including edge cases for scales, axes, packing and rounding.
7. Compile generated Cmajor with the pinned compiler and run parity tests.
8. Emit a conversion manifest containing chosen path, transforms, metadata,
   evidence references, compiler version, hashes, and numerical results.

## Failure behavior

- Unknown capability: stop with `blocked` and name the missing probe.
- Known unsupported path with no valid fallback: stop with `unsupported`.
- Fallback metadata missing, evidence absent, validation failing, or error
  above tolerance: reject conversion.
- Compiler or runtime parity failure: reject the model; never downgrade the
  failure to a warning.
- Never silently drop quantization metadata, substitute zero/random weights,
  clamp invalid scales, or emit a partially converted model as successful.

## Current evidence boundary

This repository contains the decision model and report contract only. The
actual ONNX inspection, Cmajor capability probing, tensor transformations,
compiler invocation and runtime parity integration remain to be implemented
against the target adapter/compiler. A `native` or `fallback` decision is
only as trustworthy as the adapter's evidence inputs; the classifier cannot
independently verify that those inputs are true.
