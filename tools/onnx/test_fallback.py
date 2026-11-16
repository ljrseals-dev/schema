"""Focused integration tests for the ONNX emitter's explicit fallback rewrites.

Run with: python -m pip install -r tools/onnx/requirements.txt
          python tools/onnx/test_fallback.py
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SCRIPT = Path(__file__).with_name("onnxToCmajor.py")
SPEC = importlib.util.spec_from_file_location("onnx_to_cmajor", SCRIPT)
EMITTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EMITTER)


def tensor_value_info(name, elem_type, shape):
    return helper.make_tensor_value_info(name, elem_type, shape)


class QuantizedEmitterTests(unittest.TestCase):
    def model(self, nodes, initializers, outputs, metadata=None, value_info=None):
        graph = helper.make_graph(
            nodes, "fallback-test", [], outputs,
            initializer=initializers, value_info=value_info or [],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)],
        )
        for key, value in (metadata or {}).items():
            item = model.metadata_props.add()
            item.key, item.value = key, value
        return model

    def test_float16_initializer_promoted(self):
        weight = numpy_helper.from_array(np.array([1.5, -2.0], dtype=np.float16), name="weight")
        model = self.model([], [weight], [])
        EMITTER.prepare_quantized_model(model)
        result = numpy_helper.to_array(model.graph.initializer[0])
        self.assertEqual(model.graph.initializer[0].data_type, TensorProto.FLOAT)
        np.testing.assert_array_equal(result, np.array([1.5, -2.0], dtype=np.float32))

    def test_int8_dequantize_linear_folded(self):
        q = numpy_helper.from_array(np.array([-4, 0, 8], dtype=np.int8), name="q")
        scale = numpy_helper.from_array(np.array(0.25, dtype=np.float32), name="scale")
        zero = numpy_helper.from_array(np.array(0, dtype=np.int8), name="zero")
        dq = helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["deq"], name="dq")
        output = tensor_value_info("deq", TensorProto.FLOAT, [3])
        model = self.model([dq], [q, scale, zero], [output])
        EMITTER.prepare_quantized_model(model)
        self.assertFalse(any(n.op_type == "DequantizeLinear" for n in model.graph.node))
        self.assertEqual(model.graph.output[0].name, "q")
        self.assertEqual(model.graph.initializer[0].data_type, TensorProto.FLOAT)
        np.testing.assert_allclose(
            numpy_helper.to_array(model.graph.initializer[0]), [-1.0, 0.0, 2.0],
        )

    def test_ternary_requires_and_uses_explicit_contract(self):
        packed = numpy_helper.from_array(np.array([0b00100100], dtype=np.uint8), name="weights")
        contract = (
            '{"encoding":"2bit-packed-v1","shape":[4],"bit_order":"lsb",'
            '"codebook":{"0":-1.0,"1":0.0,"2":1.0}}'
        )
        model = self.model([], [packed], [], {"cmajor.ternary.weights": contract})
        EMITTER.prepare_quantized_model(model)
        self.assertEqual(model.graph.initializer[0].data_type, TensorProto.FLOAT)
        np.testing.assert_array_equal(
            numpy_helper.to_array(model.graph.initializer[0]), [-1.0, 0.0, 1.0, -1.0],
        )

    def test_unannotated_uint8_is_rejected(self):
        packed = numpy_helper.from_array(np.array([0], dtype=np.uint8), name="weights")
        model = self.model([], [packed], [])
        with self.assertRaisesRegex(ValueError, "unsupported quantized tensors remain"):
            EMITTER.prepare_quantized_model(model)

    def test_float16_graph_boundary_is_rejected(self):
        model_input = tensor_value_info("x", TensorProto.FLOAT16, [1])
        graph = helper.make_graph([], "boundary", [model_input], [])
        model = helper.make_model(graph)
        with self.assertRaisesRegex(ValueError, "FLOAT16 graph I/O"):
            EMITTER.prepare_quantized_model(model)


if __name__ == "__main__":
    unittest.main()
