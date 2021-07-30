#  Copyright (c) 2019 by LatentAI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "LatentAI Commercial Software License".
#  Please see the LICENSE file that should have been included as part of
#  this package.
#
# @file    common_utils.py
#
# @author  Videet Parekh
#
# @date    Wed, 16 Dec 20
from .preprocessors import ImagePreprocessor

import json
import glob
import os
from collections import OrderedDict
import numpy as np
from PIL import Image
import logging
import tvm
from tvm import relay
from tvm.contrib import graph_executor


import tensorflow as tf  # noqa: E402


def collect_image(test_path, data_type, preprocessor, shape):
    im = Image.open(test_path)
    rgb_im = im.convert('RGB')
    rgb_im = rgb_im.resize(shape[1:3])
    data = np.array(rgb_im)[np.newaxis, :].astype(data_type)
    return preprocessor(data)


def set_json_field(json_obj, field, value):
    if json_obj is None:
        json_obj = {}
    segments = field.split(".")
    assert(len(segments) > 0), \
        "field cannot be empty: {}".format(field)
    field = segments.pop()
    ptr = json_obj
    for i in segments:
        if (not i) in ptr:
            ptr[i] = {}
        if type(ptr[i]).__name__ != "dict":
            ptr[i] = {}
        ptr = ptr[i]
    ptr[field] = value
    return json_obj


def load_json(path, ordered=False):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f, object_pairs_hook=OrderedDict) if ordered else json.load(f)
    else:
        return {}


def write_json(path, data):
    with open(path, 'w') as f:
        return json.dump(data, f, indent=4)


def collect_preprocessor(preprocessor):
    imgPreprocessor = ImagePreprocessor()
    return getattr(imgPreprocessor, preprocessor.lower())


def parse_input_shapes(shape_str, batch_size=1):
    shape = [part.strip() for part in shape_str.split(",")]
    shape[0] = batch_size
    return tuple(map(int, shape))


def load_index(path):
    """
    Load a testset file with newline separated tests
    where each line is a input image path,
    followed by a space, followed by the class number.

    Return N random items from that set as a test subset.
    """
    base = '/'.join(path.split("/")[0:-1])
    parsed_index = []
    with open(path, "r") as test_set_file:
        testset = test_set_file.read()
    testset = testset.strip()
    testset = testset.split("\n")
    for line in testset:
        line = line.split(" ")
        if len(line) == 2:
            line[0] = os.path.join(base, line[0])
            line[1] = int(line[1])
            parsed_index.append(line)
    return parsed_index


def get_numeric_loglevel(loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    return numeric_level


def load_keras_model(path):
    return tf.keras.models.load_model(path)


def create_leip_runtime_module(base, context):
    # Collect TVM Context Definitions for execution engine
    ctx = get_tvm_context(context)

    # Check for legacy runtime artifacts
    if os.path.isfile(os.path.join(base, "modelDescription.json")) and os.path.isfile(os.path.join(base, "modelParams.params")):
        # Load the graph, functions and weights of the function into memory
        return create_multi_artifact_runtime(base, ctx)
    return create_single_artifact_runtime(base, ctx)
        

def create_multi_artifact_runtime(base, ctx):
    graph = load_json(os.path.join(base, "modelDescription.json"))
    lib = tvm.runtime.load_module(os.path.join(base, "modelLibrary.so"))
    params = read_binary_file(os.path.join(base, "modelParams.params"))

    # Reimann
    if "leip" in graph:
        del graph["leip"]
    cast_params = get_cast_params(params, base, os.path.isfile(os.path.join(base, "quantParams.params")))
    graph = json.dumps(graph)

    # Create TVM runtime module and load weights
    module = graph_executor.create(graph, lib, ctx)
    module.load_params(cast_params)
    return module


def create_single_artifact_runtime(base, ctx):
    lib = tvm.runtime.load_module(os.path.join(base, "modelLibrary.so"))
    return graph_executor.GraphModule(lib['default'](ctx))


def get_tvm_context(context):
    return tvm.device(context) if context in ['cuda', 'cpu', 'gpu'] else tvm.cpu(0)


def get_cast_params(loaded_params, base, quant_params_exist):
    if quant_params_exist:
        quantization_file = glob.glob(os.path.join(base, "quantParams.params"))[0]
        loaded_params_qparams = read_binary_file(quantization_file)
        return dequantize(loaded_params, loaded_params_qparams)
    return loaded_params


def read_binary_file(path):
    with open(path, "rb") as f:
        bindata = bytearray(f.read())
    return bindata


def dequantize(params, q_params):
    dequantized_dict = {}
    q_params_dict = relay.load_param_dict(q_params)
    params_dict = relay.load_param_dict(params)

    for k, v in params_dict.items():
        quant_arr = params_dict[k].asnumpy()
        q_params = q_params_dict[k].asnumpy()
        scale = q_params[3]
        zpoint = q_params[4]
        dequant_array = np.multiply(scale, (quant_arr - zpoint)).astype(np.float32)

        dequantized_dict[k] = tvm.runtime.ndarray.array(dequant_array)

    return relay.save_param_dict(dequantized_dict)


