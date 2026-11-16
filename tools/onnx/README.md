# ONNX-to-Cmajor emitter fallback

This directory contains the Cmajor Toolkit emitter with explicit, fail-closed
initializer conversion for three cases. It retains the upstream Cmajor license
header. It is a repository-local replacement for the source file, not a claim
that the upstream Cmajor project has accepted these changes.

## Runtime

From the repository root:

```sh
python -m pip install -r tools/onnx/requirements.txt
python tools/onnx/onnxToCmajor.py --model path/to/model.onnx --patchDir path/to/output/patch
```

The emitter imports the audited numeric routines from
`src/schema_review/fallback_runtime.py`; run it from a complete checkout.
The Cmajor `patchTemplate/` directory must be present beside this script.

## Supported fallback contracts

### FLOAT16 initializers

Finite IEEE-754 binary16 initializers and FLOAT16 Constant tensor attributes are
promoted to FLOAT32 using the repository's `promote_float16` implementation.
FLOAT16 graph inputs and outputs are refused: changing an interface type without
an explicit runtime cast would be incorrect. NaN and infinity are refused.

### INT8 initializers

INT8 data must be an initializer consumed only by an ONNX
`DequantizeLinear` node. Its scale and optional zero-point must also be
initializers. Only signed INT8 data/zero-point tensors and scalar or
one-dimensional scales are accepted. Per-axis conversion uses the node's
`axis` attribute (ONNX default: 1). The emitter folds the dequantization into
the initializer, rewires consumers to the FLOAT32 tensor, and removes the
`DequantizeLinear` node.

Quantized graph inputs/outputs, dynamic scales, non-initializer data, mixed
direct INT8 consumers, UINT8 quantization, and unsupported parameter layouts
are refused. The emitter does not guess scale, zero point, or axis.

### Ternary packed initializers

A ternary initializer must be stored as UINT8 packed two-bit codes and carry a
model metadata property named:

```text
cmajor.ternary.<initializer-name>
```

Its value must be JSON with exactly these fields:

```json
{
  "encoding": "2bit-packed-v1",
  "shape": [4],
  "bit_order": "lsb",
  "codebook": {
    "0": -1.0,
    "1": 0.0,
    "2": 1.0
  }
}
```

The logical shape must contain positive dimensions. The codebook must define
three distinct finite values for symbols 0, 1, and 2. Symbol 3 is reserved.
The bit order must be explicitly `lsb` or `msb`; active reserved symbols,
incorrect byte counts, and non-zero padding are rejected. The example codebook
is illustrative only—the emitter does not assume those values.

No metadata means no ternary decoding. An unannotated UINT8 initializer is
refused.

## Tests

Run the focused synthetic ONNX integration tests with:

```sh
python -m pip install -r tools/onnx/requirements.txt
python tools/onnx/test_fallback.py
```

These tests exercise initializer conversion and refusal cases. They do not
prove that a generated Cmajor patch compiles, that every operator is supported,
or that target execution matches ONNX Runtime. Those remain separate acceptance
gates. Unsupported operators and Cmajor compiler/runtime failures are not
made safe merely by a successful initializer transform.
