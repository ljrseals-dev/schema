"""Executable, independently checked fallback transforms.

This module operates on decoded initializer payloads. It deliberately does not
parse ONNX files or claim Cmajor runtime parity; those remain adapter duties.
"""
from __future__ import annotations

import copy
import math
import struct
from typing import Any, Iterable


class FallbackError(ValueError):
    """Input or metadata is not safe for the requested fallback."""


def _shape_count(shape: Any) -> int:
    if not isinstance(shape, list) or not shape:
        raise FallbackError("shape must be a non-empty list of positive integers")
    if any(type(d) is not int or d <= 0 for d in shape):
        raise FallbackError("shape dimensions must be positive integers")
    count = math.prod(shape)
    return count


def _flat(values: Any, expected: int) -> list:
    if not isinstance(values, (list, tuple)) or len(values) != expected:
        raise FallbackError(f"expected {expected} flat initializer values")
    return list(values)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FallbackError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FallbackError(f"{name} must be finite")
    return result


def promote_float16(words: Iterable[int], shape: list[int]) -> list[float]:
    """Promote raw IEEE-754 binary16 words to binary32-valued Python floats."""
    count = _shape_count(shape)
    raw = _flat(list(words), count)
    output = []
    for word in raw:
        if type(word) is not int or not 0 <= word <= 0xFFFF:
            raise FallbackError("FLOAT16 payload must contain unsigned 16-bit words")
        value = struct.unpack("<e", struct.pack("<H", word))[0]
        if not math.isfinite(value):
            raise FallbackError("FLOAT16 NaN/Infinity is not accepted by this fallback")
        output.append(float(value))
    return output


def reference_float16(words: Iterable[int], shape: list[int]) -> list[float]:
    """Independent bit-field decoder used only as the validation oracle."""
    count = _shape_count(shape)
    raw = _flat(list(words), count)
    result = []
    for word in raw:
        if type(word) is not int or not 0 <= word <= 65535:
            raise FallbackError("FLOAT16 payload must contain unsigned 16-bit words")
        sign = -1.0 if word & 0x8000 else 1.0
        exponent = (word >> 10) & 0x1F
        fraction = word & 0x03FF
        if exponent == 0x1F:
            raise FallbackError("FLOAT16 NaN/Infinity is not accepted by this fallback")
        if exponent == 0:
            value = sign * math.ldexp(float(fraction), -24)
        else:
            value = sign * math.ldexp(1.0 + fraction / 1024.0, exponent - 15)
        result.append(value)
    return result


def _axis_parameters(param: Any, shape: list[int], axis: int, name: str) -> list[float]:
    axis_size = shape[axis]
    values = param if isinstance(param, (list, tuple)) else [param]
    if len(values) not in (1, axis_size):
        raise FallbackError(f"{name} length must be 1 or shape[axis]")
    parsed = [_finite_number(v, name) for v in values]
    if len(parsed) == 1:
        parsed *= axis_size
    return parsed


def dequantize_int8(values: Iterable[int], shape: list[int], *,
                    scale: Any, zero_point: Any = 0, axis: int | None = None) -> list[float]:
    """Dequantize signed INT8 values using scalar or ONNX per-axis parameters."""
    count = _shape_count(shape)
    raw = _flat(list(values), count)
    if any(type(v) is not int or not -128 <= v <= 127 for v in raw):
        raise FallbackError("INT8 payload values must be signed integers in [-128, 127]")
    if axis is None:
        scales = [_finite_number(scale, "scale")]
        zeros = [_finite_number(zero_point, "zero_point")]
        if scales[0] <= 0:
            raise FallbackError("scale must be positive")
        return [(v - zeros[0]) * scales[0] for v in raw]
    if type(axis) is not int or not -len(shape) <= axis < len(shape):
        raise FallbackError("axis is outside tensor rank")
    axis %= len(shape)
    scales = _axis_parameters(scale, shape, axis, "scale")
    zeros = _axis_parameters(zero_point, shape, axis, "zero_point")
    if any(s <= 0 for s in scales):
        raise FallbackError("all scales must be positive")
    stride = math.prod(shape[axis + 1:])
    return [
        (v - zeros[(i // stride) % shape[axis]]) * scales[(i // stride) % shape[axis]]
        for i, v in enumerate(raw)
    ]


def reference_dequantize_int8(values: Iterable[int], shape: list[int], *,
                              scale: Any, zero_point: Any = 0,
                              axis: int | None = None) -> list[float]:
    """Reference dequantizer computes coordinates independently of flat stride."""
    count = _shape_count(shape)
    raw = _flat(list(values), count)
    if any(type(v) is not int or v < -128 or v > 127 for v in raw):
        raise FallbackError("INT8 payload values must be signed integers in [-128, 127]")
    if axis is None:
        s = _finite_number(scale, "scale")
        z = _finite_number(zero_point, "zero_point")
        if s <= 0:
            raise FallbackError("scale must be positive")
        return [float(v - z) * s for v in raw]
    if type(axis) is not int or axis < -len(shape) or axis >= len(shape):
        raise FallbackError("axis is outside tensor rank")
    axis = axis % len(shape)
    scale_values = list(scale) if isinstance(scale, (list, tuple)) else [scale]
    zero_values = list(zero_point) if isinstance(zero_point, (list, tuple)) else [zero_point]
    if len(scale_values) not in (1, shape[axis]) or len(zero_values) not in (1, shape[axis]):
        raise FallbackError("per-axis parameter length mismatch")
    scale_values = [_finite_number(x, "scale") for x in scale_values]
    zero_values = [_finite_number(x, "zero_point") for x in zero_values]
    if len(scale_values) == 1:
        scale_values = scale_values * shape[axis]
    if len(zero_values) == 1:
        zero_values = zero_values * shape[axis]
    if any(x <= 0 for x in scale_values):
        raise FallbackError("all scales must be positive")
    result = []
    for flat_index, q in enumerate(raw):
        remainder = flat_index
        coordinates = [0] * len(shape)
        for dim in range(len(shape) - 1, -1, -1):
            coordinates[dim] = remainder % shape[dim]
            remainder //= shape[dim]
        channel = coordinates[axis]
        result.append(float(q - zero_values[channel]) * scale_values[channel])
    return result


def decode_ternary(packed: Iterable[int], *, values_count: int, shape: list[int],
                   codebook: dict[int, float], bit_order: str = "lsb") -> list[float]:
    """Decode explicit two-bit symbols; code 3 must be absent from the codebook."""
    count = _shape_count(shape)
    if count != values_count or type(values_count) is not int or values_count < 1:
        raise FallbackError("values_count must equal the positive shape element count")
    if bit_order not in ("lsb", "msb"):
        raise FallbackError("bit_order must be 'lsb' or 'msb'")
    if not isinstance(codebook, dict) or set(codebook) != {0, 1, 2}:
        raise FallbackError("ternary codebook must explicitly map codes 0, 1, and 2")
    symbols = {k: _finite_number(v, "codebook value") for k, v in codebook.items()}
    if len(set(symbols.values())) != 3:
        raise FallbackError("ternary codebook values must be distinct")
    data = list(packed)
    if any(type(b) is not int or not 0 <= b <= 255 for b in data):
        raise FallbackError("packed ternary payload must contain bytes")
    needed = (values_count + 3) // 4
    if len(data) != needed:
        raise FallbackError("packed byte count does not match values_count")
    output = []
    for index in range(values_count):
        byte = data[index // 4]
        shift = (index % 4) * 2 if bit_order == "lsb" else 6 - (index % 4) * 2
        code = (byte >> shift) & 0x03
        if code not in symbols:
            raise FallbackError("reserved ternary symbol encountered")
        output.append(symbols[code])
    unused = values_count % 4
    if unused:
        last = data[-1]
        mask = (0xFF << (unused * 2)) & 0xFF if bit_order == "lsb" else (1 << (8 - unused * 2)) - 1
        if last & mask:
            raise FallbackError("non-zero ternary padding bits")
    return output


def reference_decode_ternary(packed: Iterable[int], *, values_count: int, shape: list[int],
                             codebook: dict[int, float], bit_order: str = "lsb") -> list[float]:
    """Independent decoder extracts each two-bit symbol by explicit bit position."""
    count = _shape_count(shape)
    if count != values_count or type(values_count) is not int or values_count < 1:
        raise FallbackError("values_count must equal the positive shape element count")
    if bit_order not in ("lsb", "msb") or not isinstance(codebook, dict) or set(codebook) != {0, 1, 2}:
        raise FallbackError("invalid ternary encoding contract")
    data = list(packed)
    if len(data) != (values_count + 3) // 4 or any(type(x) is not int or x < 0 or x > 255 for x in data):
        raise FallbackError("invalid packed ternary byte sequence")
    values = {key: _finite_number(value, "codebook value") for key, value in codebook.items()}
    decoded = []
    for byte_index, byte in enumerate(data):
        for slot in range(4):
            index = byte_index * 4 + slot
            if index == values_count:
                break
            bit = 2 * slot if bit_order == "lsb" else 6 - 2 * slot
            symbol = (byte >> bit) & 3
            if symbol == 3 or symbol not in values:
                raise FallbackError("reserved ternary symbol encountered")
            decoded.append(values[symbol])
    if values_count % 4:
        valid_bits = (values_count % 4) * 2
        padding = (data[-1] >> valid_bits) if bit_order == "lsb" else (data[-1] & ((1 << (8 - valid_bits)) - 1))
        if padding:
            raise FallbackError("non-zero ternary padding bits")
    return decoded


def execute_and_validate(feature: str, payload: Any, metadata: dict[str, Any],
                         tolerance: float) -> dict[str, Any]:
    """Run production transform and independent oracle; return auditable metrics."""
    tolerance = _finite_number(tolerance, "tolerance")
    if tolerance < 0:
        raise FallbackError("tolerance must be non-negative")
    if not isinstance(metadata, dict):
        raise FallbackError("metadata must be an object")
    before = copy.deepcopy(metadata)
    if feature == "float16":
        if not isinstance(payload, dict):
            raise FallbackError("FLOAT16 payload must be an object")
        args = (payload.get("words"), payload.get("shape"))
        actual = promote_float16(*args)
        expected = reference_float16(*args)
    elif feature == "int8":
        if not isinstance(payload, dict):
            raise FallbackError("INT8 payload must be an object")
        kwargs = {key: payload[key] for key in ("scale", "zero_point", "axis") if key in payload}
        args = (payload.get("values"), payload.get("shape"))
        actual = dequantize_int8(*args, **kwargs)
        expected = reference_dequantize_int8(*args, **kwargs)
    elif feature == "ternary":
        if not isinstance(payload, dict):
            raise FallbackError("ternary payload must be an object")
        kwargs = {key: payload[key] for key in ("values_count", "shape", "codebook", "bit_order") if key in payload}
        actual = decode_ternary(payload.get("packed", []), **kwargs)
        expected = reference_decode_ternary(payload.get("packed", []), **kwargs)
    else:
        raise FallbackError(f"unsupported fallback feature: {feature!r}")
    if len(actual) != len(expected):
        raise FallbackError("reference output length mismatch")
    errors = [abs(a - b) for a, b in zip(actual, expected)]
    max_abs = max(errors, default=0.0)
    max_ref = max((abs(x) for x in expected), default=0.0)
    max_rel = max((e / max(abs(r), 1e-30) for e, r in zip(errors, expected)), default=0.0)
    preserved = metadata == before
    passed = preserved and max_abs <= tolerance
    return {
        "feature": feature,
        "output": actual,
        "shape": payload.get("shape") if isinstance(payload, dict) else None,
        "metadata_preserved": preserved,
        "preserved_metadata": before,
        "validation_passed": passed,
        "measured_error": max_abs,
        "error_tolerance": tolerance,
        "metrics": {"max_absolute_error": max_abs, "max_relative_error": max_rel,
                    "reference_max_absolute_value": max_ref, "tolerance": tolerance},
        "evidence": [
            "oracle:independent-stdlib-reference-v1",
            f"transform:{feature}-fallback-v1",
            f"metadata-preserved:{str(preserved).lower()}",
        ],
        "reason": None if passed else "metadata_changed_or_error_above_tolerance",
    }
