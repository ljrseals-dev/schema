#!/usr/bin/env python3

#
#     ,ad888ba,                              88
#    d8"'    "8b
#   d8            88,dba,,adba,   ,aPP8A.A8  88     The Cmajor Toolkit
#   Y8,           88    88    88  88     88  88
#    Y8a.   .a8P  88    88    88  88,   ,88  88     (C)2024 Cmajor Software Ltd
#     '"Y888Y"'   88    88    88  '"8bbP"Y8  88
#                                           ,88
#                                        888P"
#
#  The Cmajor project is subject to commercial or open-source licensing.
#  You may use it under the terms of the GPLv3 (see www.gnu.org/licenses), or
#  visit https://cmajor.dev to learn about our commercial licence options.
#
#  CMAJOR IS PROVIDED "AS IS" WITHOUT ANY WARRANTY, AND ALL WARRANTIES, WHETHER
#  EXPRESSED OR IMPLIED, INCLUDING MERCHANTABILITY AND FITNESS FOR PURPOSE, ARE
#  DISCLAIMED.

import sys, shutil, os, getopt, onnx, json
from pathlib import Path
import numpy as np
from onnx import numpy_helper, shape_inference

# Reuse the repository's audited numeric transforms. This script remains
# executable from any working directory inside a checkout.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
try:
    from schema_review.fallback_runtime import (
        FallbackError, promote_float16, dequantize_int8, decode_ternary,
    )
except ImportError as exc:
    raise RuntimeError(
        "schema_review fallback runtime is required; run from a complete schema checkout"
    ) from exc

patchTemplateDir = os.path.dirname(os.path.abspath(sys.argv[0])) + "/patchTemplate"

dimParamFrames = 1
endpointSuffix = "_Transformer"

ioBlock = []
nodes = []
connections = []
initialisers = []
endpoints = {}
connectionDataTypes = {}
outputNames = []

nodeEndpointNames = {
    "Tanh" :        [["in"],                                                                                    ["out"]],
    "Reshape" :     [["in", ""],                                                                                ["out"]],
    "MatMul" :      [["in", "mul"],                                                                             ["out"]],
    "Gemm" :        [["in", "mul", "add"],                                                                      ["out"]],
    "Add" :         [["inA", "inB"],                                                                            ["out"]],
    "Unsqueeze" :   [["in", ""],                                                                                ["out"]],
    "Squeeze" :     [["in", ""],                                                                                ["out"]],
    "Conv" :        [["in", "inWeights"],                                                                       ["out"]],
    "GRU" :         [["in", "inWVals", "inRVals", "inBVals", "seq", "inInitial"],                               ["out"]],
    "Pad" :         [["in", "pads"],                                                                            ["out"]],
    "Transpose" :   [["in"],                                                                                    ["out"]],
    "Constant":     [[],                                                                                        ["out"]],
    "LSTM":         [["inX", "inW", "inR", "inB", "inSequenceLens", "initialH", "initialC", "inP"],             ["outY", "outY_h", "outY_c"]],
    "Slice" :       [["data", "starts", "ends", "axes", "steps"],                                               ["out"]],
    "Gather" :      [["in", "indices"],                                                                         ["out"]],
    "Concat" :      [["in1", "in2", "in3"],                                                                     ["out"]],
}

def getConnectionDataType (name):

    if name in connectionDataTypes:
        return connectionDataTypes[name][0]

    assert (False)

def getConnectionDataTypeDimensions (name):
    if name in connectionDataTypes:
        return connectionDataTypes[name][1]

    assert (False)

def getConnectionDataTypeTotalElements (name):
    if name in connectionDataTypes:
        return connectionDataTypes[name][2]

    assert (False)

def getDataType (dataType: int):

    dataTypes = {
        onnx.TensorProto.FLOAT:     ["float32", "f"],
        onnx.TensorProto.DOUBLE:    ["float64", ""],
        onnx.TensorProto.INT64:     ["int64",   "i64"],
        onnx.TensorProto.INT32:     ["int32",   ""],
        onnx.TensorProto.BOOL:      ["bool",    ""]
    }

    if dataType in dataTypes:
        return dataTypes[dataType]

    raise ValueError(f"unsupported ONNX tensor data type: {dataType}")

def createCmajArray (dimensions, array, elementSuffix):

    if len(dimensions) == 0:
        return str (array) + elementSuffix

    if len(dimensions) == 1:
        items = []

        for i in array.tolist():
            items.append (str (i) + elementSuffix)

        return "(" + ", ".join (items) + ")"

    items = []
    innerDims = dimensions[1:]

    for i in range (dimensions[0]):
        items.append (createCmajArray (innerDims, array[i], elementSuffix))

    return "(" + ", ".join (items) + ")"

def getEndpointInputName (opType, id):
    if opType in nodeEndpointNames:
        nodeEndpointInputNameList = nodeEndpointNames[opType][0]
        if id < len (nodeEndpointInputNameList):
            return nodeEndpointInputNameList[id]

    return "in" + str (id)

def getEndpointOutputName (opType, id):
    if opType in nodeEndpointNames:
        nodeEndpointInputNameList = nodeEndpointNames[opType][1]
        if id < len (nodeEndpointInputNameList):
            return nodeEndpointInputNameList[id]

    return "out" + str (id)

def sanitiseName (s):
    return s.replace('/','_').replace(":", "_").replace(".", "_").lstrip ("_")

def populateNodeOutputs (model: onnx.ModelProto):
    for op in model.graph.node:
        nodeName = sanitiseName (op.name)
        for index, outputName in enumerate (op.output):
            endpoints[sanitiseName (outputName)] = nodeName + "." + getEndpointOutputName (op.op_type, index)

    initialisers = []

    for initialiser in model.graph.initializer:
        initialisers.append (initialiser.name)
        endpoints[sanitiseName (initialiser.name)] = sanitiseName (initialiser.name) + ".out"
        addInitialiser (initialiser)

    for input in model.graph.input:
        if input.name not in initialisers:
            endpoints[sanitiseName (input.name)] = sanitiseName (input.name) + endpointSuffix

    for output in model.graph.output:
        endpoints[sanitiseName (output.name)] = sanitiseName (output.name)

def addInitialiser (initialiser):
    nodeName = sanitiseName (initialiser.name)
    nodes.append ("        " + nodeName + " = onnx::operation::Constant (" + nodeName + "_.type, " + nodeName + "_);")

    dimensions = ','.join(map(str, initialiser.dims))
    dataType, elementSuffix = getDataType (initialiser.data_type)
    array = numpy_helper.to_array(initialiser)
    dataArray = str(createCmajArray (initialiser.dims, array, elementSuffix)).replace ('[','(').replace(']',')')
    initialisers.append ("    let " + nodeName + "_ = " + dataType + "[" + dimensions + "] " + dataArray + ";")

def createConstantValue (constantName, attribute):

    dataType, elementSuffix = getDataType (attribute.t.data_type)
    dimensions = ','.join(map(str, attribute.t.dims))
    array = numpy_helper.to_array(attribute.t)
    dataArray = str(createCmajArray (attribute.t.dims, array, elementSuffix)).replace ('[','(').replace(']',')')

    if len (dimensions) == 0:
        initialisers.append ("    let " + constantName + "_ = " + dataType + " (" + dataArray + ");")
    else:
        initialisers.append ("    let " + constantName + "_ = " + dataType + "[" + dimensions + "] " + dataArray + ";")

    return constantName + "_"

def getNodeAttribute (op: onnx.NodeProto, name):
    for attribute in op.attribute:
        if attribute.name == name:
            return attribute

def getNodeAttributeValue (op: onnx.NodeProto, name):

    for attribute in op.attribute:
        if attribute.name == name:
            if attribute.type == onnx.AttributeProto.INT:
                return str (attribute.i)

            if attribute.type == onnx.AttributeProto.INTS:

                intList = []
                for i in attribute.ints:
                    intList.append (i)

                return "int[" + str (len (intList)) + "] (" + ", ".join(map(str, intList)) + ")"

def getOperatorParameters (op: onnx.NodeProto):
    outputDataType = getConnectionDataType (op.output[0])

    if op.op_type == "Constant":
        constantName = createConstantValue (sanitiseName (op.name), getNodeAttribute (op, "value"))
        return " (" + outputDataType + ", " + constantName + ")"

    inputDataType = getConnectionDataType (op.input[0])

    if op.op_type == "Add":
        return " (" + inputDataType + ")"

    if op.op_type == "Tanh":
        return " (" + inputDataType + ")"

    if op.op_type == "Conv":
        dilations = getNodeAttributeValue (op, "dilations")
        kernelShape = getNodeAttributeValue (op, "kernel_shape")
        return " (" + inputDataType + ", " + outputDataType + ", " + str (kernelShape) + ", " + str (dilations) + ")"

    if op.op_type == "LSTM":
        hiddenSize = getNodeAttributeValue (op, "hidden_size")
        return " (" + inputDataType + ", " + outputDataType + ", " + str (hiddenSize) + ")"

    if op.op_type == "Transpose":
        perm = getNodeAttributeValue (op, "perm")
        totalElements = getConnectionDataTypeTotalElements (op.input[0])
        return " (" + inputDataType + ", " + outputDataType + ", " + str (totalElements) + ", " + perm + ")"

    if op.op_type == "MatMul":
        mulType = getConnectionDataType (op.input[1])
        return " (" + inputDataType + ", " + outputDataType + ", " + mulType + ")"

    if op.op_type == "Gemm":
        mulType = getConnectionDataType (op.input[1])
        return " (" + inputDataType + ", " + outputDataType + ", " + mulType + ")"

    if op.op_type == "Slice":
        axesType = "int[" + str (getConnectionDataTypeDimensions (op.input[0])) + "]"

        if (op.input[3]):
            axesType = getConnectionDataType (op.input[3])

        return " (" + inputDataType + ", " + outputDataType + ", " + axesType + ")"

    if op.op_type == "Gather":
        indicesType = getConnectionDataType (op.input[1])
        return " (" + inputDataType + ", " + outputDataType + ", " + indicesType + ")"

    if op.op_type == "Concat":
        axisValue = getNodeAttributeValue (op, "axis")
        return " (" + inputDataType + ", " + outputDataType + ", " + "3" + ", " + axisValue + ")"

    return " (" + inputDataType + ", " + outputDataType + ")"

def generateNode (op: onnx.NodeProto):
    global nodes
    global connections

    nodeName = sanitiseName (op.name)
    opType   = op.op_type
    nodes.append ("        " + nodeName + " = onnx::operation::" + opType + getOperatorParameters (op) + ";")

    for index, input in enumerate (op.input):
        if input != "":
            inputEndpoint = getEndpointInputName (opType, index)
            if inputEndpoint != "":
                connections.append ("        " + endpoints[sanitiseName (input)] + " -> " + nodeName + "." + inputEndpoint + ";")

    for index, output in enumerate (op.output):
        if output in outputNames:
            outputEndpoint = getEndpointOutputName (opType, index)
            connections.append ("        " + nodeName + "." + outputEndpoint + " -> " + output + endpointSuffix + ";")

def parseInitialiserType (initialiser):
        dataType, elementSuffix = getDataType (initialiser.data_type)
        dimensions = []
        totalElements = 1
        for dim in initialiser.dims:
            dimensions.append (dim)
            totalElements *= dim

        if len (dimensions) == 0:
            return dataType, 0, 1

        dimensionList = ','.join(map(str, dimensions))

        return dataType + "[" + dimensionList + "]", len (dimensions), totalElements

def parseTensorType (tensorType):
        dataType, elementSuffix = getDataType (tensorType.elem_type)
        dimensions = []
        totalElements = 1

        for dim in tensorType.shape.dim:
            if dim.dim_value:
                dimensions.append (dim.dim_value)
                totalElements *= dim.dim_value
            elif dim.dim_param:
                dimensions.append (dimParamFrames)
                totalElements *= dimParamFrames

        if len (dimensions) == 0:
            return dataType, 0, 1

        dimensionList = ','.join(map(str, dimensions))

        return dataType + "[" + dimensionList + "]", len (dimensions), totalElements

def parseTensorTypeToStreamType (tensorType):
        dataType, elementSuffix = getDataType (tensorType.elem_type)
        elements = 1

        for dim in tensorType.shape.dim:
            if dim.dim_value:
                elements *= dim.dim_value
            elif dim.dim_param:
                elements *= dimParamFrames

        return dataType + "<" + str (elements) + ">"

def populateDataTypes (model: onnx.ModelProto):

    for input in model.graph.input:
        connectionDataTypes[input.name] = parseTensorType (input.type.tensor_type)

    for output in model.graph.output:
        connectionDataTypes[output.name] = parseTensorType (output.type.tensor_type)

    for valueInfo in model.graph.value_info:
        name = valueInfo.name
        connectionDataTypes[name] = parseTensorType (valueInfo.type.tensor_type)

    for initialiser in model.graph.initializer:
        name = initialiser.name
        connectionDataTypes[name] = parseInitialiserType (initialiser)


def populateIO (model: onnx.ModelProto):
    global ioBlock
    global nodes
    global connections
    global outputNames

    initialisers = []

    for initialiser in model.graph.initializer:
        initialisers.append (initialiser.name)

    for input in model.graph.input:
        if input.name not in initialisers:
            streamType = parseTensorTypeToStreamType (input.type.tensor_type)
            tensorType = getConnectionDataType (input.name)

            ioBlock.append ("    input stream " + streamType + " " + input.name + ";")
            nodes.append ("        " + input.name + endpointSuffix + " = onnx::graphIn (" + streamType + ", " + tensorType + ");")
            connections.append ("        " + input.name + " -> " + input.name + endpointSuffix + ";")

    for output in model.graph.output:
        outputNames.append (output.name)
        streamType = parseTensorTypeToStreamType (output.type.tensor_type)
        tensorType = getConnectionDataType (output.name)

        ioBlock.append ("    output stream " + streamType + " " + output.name + ";")
        nodes.append ("        " + output.name + endpointSuffix + " = onnx::graphOut (" + tensorType + ", " + streamType + ");")
        connections.append ("        " + output.name + endpointSuffix + " -> " + output.name + ";")

    connections.append ("")

def populateMissingNodeNames (model: onnx.ModelProto):
    nextNode = 1

    for node in model.graph.node:
        if node.name == "":
            node.name = "node_" + str (nextNode)
            nextNode = nextNode + 1

def generateNodesAndConnections (model: onnx.ModelProto):
    populateMissingNodeNames (model)
    populateDataTypes (model)
    populateIO (model)
    populateNodeOutputs (model)

    for op in enumerate (model.graph.node):
        generateNode (op[1])


def printCmajor (patchDir):

    with open (patchTemplateDir + "/model.cmajor", "r") as f:
        mainGraphBody = f.read()

    cmaj = mainGraphBody.replace("NODES", "\n".join (nodes)) \
                        .replace("IOBLOCK", "\n".join (ioBlock)) \
                        .replace("CONNECTIONS", "\n".join (connections)) \
                        .replace("INITIALISERS", "\n".join (initialisers))

    if patchDir == None:
        print (cmaj)
    else:

        if not os.path.exists (patchDir):
            shutil.copytree (patchTemplateDir, patchDir)
        else:
            with open (patchTemplateDir + "/onnx.cmajor", "r") as f:
                f2 = open (patchDir + "/onnx.cmajor", "w")
                f2.write (f.read())
            with open (patchTemplateDir + "/model.cmajorpatch", "r") as f:
                f2 = open (patchDir + "/model.cmajorpatch", "w")
                f2.write (f.read())

        f = open (patchDir + "/model.cmajor", "w")
        f.write (cmaj)
        print ("Cmajor patch created: " + patchDir + "/model.cmajor")
        f.close()

def usage():
    print (
"""

  This script generates Cmajor code from an ONNX model.

  Options:
    -h                   Display this help
    --model <file>       Specifies the ONNX model file to convert
    --patchDir <folder>  Specifies a folder into which the generated patch will be written

  If no patchDir argument is supplied, the output will be a single Cmajor file.

""")


def _tensor_shape(tensor):
    return [int(d) for d in tensor.dims]


def _replace_initializer(graph, name, values, shape):
    """Replace one initializer with finite FLOAT32 values and its logical shape."""
    array = np.asarray(values, dtype=np.float32).reshape(tuple(shape))
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: decoded initializer contains NaN or infinity")
    replacement = numpy_helper.from_array(array, name=name)
    for index, item in enumerate(graph.initializer):
        if item.name == name:
            graph.initializer[index].CopyFrom(replacement)
            return
    raise ValueError(f"{name}: initializer disappeared during conversion")


def _model_metadata(model):
    result = {}
    for item in model.metadata_props:
        if item.key in result:
            raise ValueError(f"duplicate model metadata key: {item.key}")
        result[item.key] = item.value
    return result


def _decode_ternary_initializers(model):
    """Decode only explicitly annotated UINT8 two-bit packed initializers.

    Metadata key: cmajor.ternary.<initializer-name>
    Value JSON: {"encoding":"2bit-packed-v1","shape":[...],
                 "bit_order":"lsb|msb","codebook":{"0":...,"1":...,"2":...}}
    """
    metadata = _model_metadata(model)
    prefix = "cmajor.ternary."
    annotated = {key[len(prefix):]: value for key, value in metadata.items()
                 if key.startswith(prefix)}
    initializer_by_name = {item.name: item for item in model.graph.initializer}
    for name, raw_contract in annotated.items():
        if name not in initializer_by_name:
            raise ValueError(f"ternary metadata references missing initializer: {name}")
        tensor = initializer_by_name[name]
        if tensor.data_type != onnx.TensorProto.UINT8:
            raise ValueError(f"{name}: ternary packed initializer must be UINT8")
        try:
            contract = json.loads(raw_contract)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name}: malformed ternary metadata JSON") from exc
        required = {"encoding", "shape", "bit_order", "codebook"}
        if not isinstance(contract, dict) or set(contract) != required:
            raise ValueError(f"{name}: ternary metadata must contain exactly {sorted(required)}")
        if contract["encoding"] != "2bit-packed-v1":
            raise ValueError(f"{name}: unsupported ternary encoding")
        shape = contract["shape"]
        if (not isinstance(shape, list) or not shape
                or any(type(d) is not int or d <= 0 for d in shape)):
            raise ValueError(f"{name}: ternary logical shape must contain positive integers")
        codebook_raw = contract["codebook"]
        if not isinstance(codebook_raw, dict) or set(codebook_raw) != {"0", "1", "2"}:
            raise ValueError(f"{name}: ternary codebook must explicitly define codes 0, 1, and 2")
        codebook = {int(key): value for key, value in codebook_raw.items()}
        packed = numpy_helper.to_array(tensor).reshape(-1).tolist()
        try:
            decoded = decode_ternary(
                packed, values_count=int(np.prod(shape)), shape=shape,
                codebook=codebook, bit_order=contract["bit_order"],
            )
        except (FallbackError, TypeError, ValueError) as exc:
            raise ValueError(f"{name}: ternary decode rejected: {exc}") from exc
        _replace_initializer(model.graph, name, decoded, shape)


def _dequantize_int8_initializers(model):
    """Fold initializer-backed ONNX DequantizeLinear nodes into FLOAT32 weights."""
    graph = model.graph
    initializer_by_name = {item.name: item for item in graph.initializer}
    dq_nodes = [node for node in graph.node if node.op_type == "DequantizeLinear"]
    if not dq_nodes:
        # INT8 tensors without a declared DequantizeLinear contract are ambiguous.
        int8_names = [t.name for t in graph.initializer
                      if t.data_type == onnx.TensorProto.INT8]
        if int8_names:
            raise ValueError("INT8 initializers require an explicit DequantizeLinear node: "
                             + ", ".join(int8_names))
        return

    parameter_names = set()
    plans = []
    for node in dq_nodes:
        if len(node.input) < 2 or len(node.input) > 3 or len(node.output) != 1:
            raise ValueError(f"{node.name or 'DequantizeLinear'}: unsupported input/output arity")
        data_name, scale_name = node.input[0], node.input[1]
        zero_name = node.input[2] if len(node.input) == 3 and node.input[2] else None
        if data_name not in initializer_by_name:
            raise ValueError(f"{node.name or 'DequantizeLinear'}: data input must be an initializer")
        if scale_name not in initializer_by_name:
            raise ValueError(f"{node.name or 'DequantizeLinear'}: scale must be an initializer")
        data_tensor = initializer_by_name[data_name]
        if data_tensor.data_type != onnx.TensorProto.INT8:
            raise ValueError(f"{data_name}: this fallback supports signed INT8 data only")
        scale_array = numpy_helper.to_array(initializer_by_name[scale_name])
        if scale_array.ndim > 1:
            raise ValueError(f"{scale_name}: scale must be scalar or one-dimensional")
        if zero_name is None:
            zero_array = np.asarray(0, dtype=np.int8)
        else:
            if zero_name not in initializer_by_name:
                raise ValueError(f"{node.name or 'DequantizeLinear'}: zero point must be an initializer")
            zero_tensor = initializer_by_name[zero_name]
            if zero_tensor.data_type != onnx.TensorProto.INT8:
                raise ValueError(f"{zero_name}: zero point must be signed INT8")
            zero_array = numpy_helper.to_array(zero_tensor)
            if zero_array.ndim > 1:
                raise ValueError(f"{zero_name}: zero point must be scalar or one-dimensional")
        axis = None
        if scale_array.ndim == 1 and scale_array.size > 1:
            axis_attr = next((a for a in node.attribute if a.name == "axis"), None)
            axis = int(axis_attr.i) if axis_attr is not None else 1
        shape = _tensor_shape(data_tensor)
        transform_shape = shape if shape else [1]
        try:
            values = dequantize_int8(
                numpy_helper.to_array(data_tensor).reshape(-1).tolist(),
                transform_shape, scale=scale_array.reshape(-1).tolist()[0]
                    if scale_array.size == 1 else scale_array.reshape(-1).tolist(),
                zero_point=zero_array.reshape(-1).tolist()[0]
                    if zero_array.size == 1 else zero_array.reshape(-1).tolist(),
                axis=axis,
            )
        except (FallbackError, TypeError, ValueError) as exc:
            raise ValueError(f"{data_name}: INT8 dequantization rejected: {exc}") from exc
        plans.append((node, data_name, node.output[0], values, shape))
        parameter_names.add(scale_name)
        if zero_name:
            parameter_names.add(zero_name)

    # A quantized initializer may not also feed an ordinary operator directly.
    dq_data_names = {plan[1] for plan in plans}
    if len(dq_data_names) != len(plans):
        raise ValueError("an INT8 initializer may have only one DequantizeLinear consumer")
    graph_input_names = {item.name for item in graph.input}
    graph_output_names = {item.name for item in graph.output}
    for data_name in dq_data_names:
        if data_name in graph_input_names or data_name in graph_output_names:
            raise ValueError(f"{data_name}: quantized graph inputs/outputs require an explicit runtime adapter")
    for node in graph.node:
        if node.op_type == "DequantizeLinear":
            continue
        for input_name in node.input:
            if input_name in dq_data_names:
                raise ValueError(f"{input_name}: INT8 initializer has a non-DequantizeLinear consumer")

    # Replace every DQ output use with the decoded FLOAT32 initializer name.
    aliases = {output_name: data_name for _, data_name, output_name, _, _ in plans}
    for node in graph.node:
        if node.op_type == "DequantizeLinear":
            continue
        for index, input_name in enumerate(node.input):
            if input_name in aliases:
                node.input[index] = aliases[input_name]
    for output in graph.output:
        if output.name in aliases:
            output.name = aliases[output.name]
            output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    for value_info in list(graph.value_info):
        if value_info.name in aliases:
            value_info.name = aliases[value_info.name]
        if value_info.name in dq_data_names:
            value_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    for _, data_name, _, values, shape in plans:
        _replace_initializer(graph, data_name, values, shape)

    kept_nodes = [node for node in graph.node if node.op_type != "DequantizeLinear"]
    del graph.node[:]
    graph.node.extend(kept_nodes)

    # Remove quantization parameter initializers only when no remaining node uses them.
    used = {name for node in graph.node for name in node.input if name}
    kept_initializers = [
        tensor for tensor in graph.initializer
        if tensor.name not in parameter_names or tensor.name in used
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)


def _promote_float16_tensors(model):
    """Promote FLOAT16 initializers and tensor value declarations to FLOAT32."""
    for tensor in model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT16:
            array = numpy_helper.to_array(tensor)
            words = array.view(np.uint16).reshape(-1).tolist()
            shape = _tensor_shape(tensor)
            transform_shape = shape if shape else [1]
            try:
                values = promote_float16(words, transform_shape)
            except (FallbackError, TypeError, ValueError) as exc:
                raise ValueError(f"{tensor.name}: FLOAT16 promotion rejected: {exc}") from exc
            _replace_initializer(model.graph, tensor.name, values, shape)

    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type == onnx.TensorProto.FLOAT16:
                array = numpy_helper.to_array(attribute.t)
                words = array.view(np.uint16).reshape(-1).tolist()
                shape = _tensor_shape(attribute.t)
                transform_shape = shape if shape else [1]
                try:
                    values = promote_float16(words, transform_shape)
                except (FallbackError, TypeError, ValueError) as exc:
                    raise ValueError(f"{node.name}: FLOAT16 Constant promotion rejected: {exc}") from exc
                replacement = numpy_helper.from_array(
                    np.asarray(values, dtype=np.float32).reshape(tuple(shape)),
                    name=attribute.t.name,
                )
                attribute.t.CopyFrom(replacement)

    # A graph boundary needs an explicit input/output cast, not just a changed
    # type declaration. This emitter has no such adapter, so reject it.
    for value_info in list(model.graph.input) + list(model.graph.output):
        if value_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            raise ValueError(
                f"{value_info.name}: FLOAT16 graph I/O requires an explicit runtime cast adapter"
            )
    for value_info in model.graph.value_info:
        if value_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            value_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT


def prepare_quantized_model(model):
    """Apply only explicit, supported fallback rewrites; reject ambiguous tensors."""
    # DequantizeLinear is admitted only as an initializer-folding pattern;
    # _dequantize_int8_initializers removes it before ordinary code generation.
    allowed_ops = set(nodeEndpointNames) | {"DequantizeLinear"}
    unsupported_nodes = [
        f"{node.name or '<unnamed>'}:{node.domain or 'ai.onnx'}::{node.op_type}"
        for node in model.graph.node
        if node.domain not in ("", "ai.onnx") or node.op_type not in allowed_ops
    ]
    if unsupported_nodes:
        raise ValueError("unsupported ONNX operator(s): " + ", ".join(unsupported_nodes))
    _decode_ternary_initializers(model)
    _dequantize_int8_initializers(model)
    _promote_float16_tensors(model)

    # No unsupported quantized tensor type may reach the ordinary emitter.
    unsupported = []
    for tensor in model.graph.initializer:
        if tensor.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8,
                                onnx.TensorProto.FLOAT16):
            unsupported.append(f"{tensor.name}: ONNX tensor type {tensor.data_type}")
    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        elem_type = value_info.type.tensor_type.elem_type
        if elem_type in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8,
                         onnx.TensorProto.FLOAT16):
            unsupported.append(f"{value_info.name}: graph value type {elem_type}")
    if unsupported:
        raise ValueError("unsupported quantized tensors remain after fallback: "
                         + "; ".join(unsupported))



def main(argv):
    if len(argv) == 0:
        usage()
        exit(1)

    try:
        opts, args = getopt.getopt (argv, "h", ["patchDir=", "model="])
    except getopt.GetoptError as e:
        print (f"{e}")
        usage()
        exit (1)

    patchDir = None
    model = None

    for opt, arg in opts:
        if opt in ("-h"):
            usage()
            exit (1)

        if opt in ("--patchDir"):
            patchDir = arg

        if opt in ("--model"):
            model = arg

    if model == None:
        model = args[0]

    onnxModel = onnx.load (model)
    try:
        onnx.checker.check_model (onnxModel)
        prepare_quantized_model(onnxModel)
        inferredModel = shape_inference.infer_shapes(onnxModel)
        onnx.checker.check_model(inferredModel)
        generateNodesAndConnections (inferredModel)
    except (onnx.checker.ValidationError, ValueError, RuntimeError) as e:
        print (f"Model conversion refused: {e}", file=sys.stderr)
        exit(2)

    printCmajor (patchDir)

if __name__== "__main__" :
    main (sys.argv[1:])
    exit()
