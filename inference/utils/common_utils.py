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
from tvm.contrib import graph_runtime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
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


def create_runtime_module(base, context):
    # Load the graph, functions and weights of the function into memory
    graph = load_json(os.path.join(base, "modelDescription.json"))
    lib = tvm.runtime.load_module(os.path.join(base, "modelLibrary.so"))
    params = read_binary_file(os.path.join(base, "modelParams.params"))

    # Collect TVM Context Definitions for execution engine
    ctx = get_tvm_context(context)

    # Reimann
    precision = 'float32'
    if "leip" in graph:
        precision = graph["leip"].get("precision")
        del graph["leip"]
    graph = json.dumps(graph)

    cast_params = get_cast_params(precision, params, base)

    # Create TVM runtime module and load weights
    module = graph_runtime.create(graph, lib, ctx)
    module.load_params(cast_params)
    return module


def get_tvm_context(context):
    return tvm.context(context) if context in ['cuda', 'cpu', 'gpu'] else tvm.cpu(0)


def get_cast_params(precision, loaded_params, base):
    if precision == 'int8' or precision == 'uint8':
        quantization_file = glob.glob(os.path.join(base, "quant*"))[0]
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

    for i in params_dict:
        quant_arr = params_dict[i].asnumpy()
        q_params = q_params_dict[i].asnumpy()
        scale = q_params[3]
        zpoint = q_params[4]
        dequant_array = np.multiply(scale, (quant_arr - zpoint)).astype(np.float32)

        dequantized_dict[i] = tvm.runtime.ndarray.array(dequant_array)

    return relay.save_param_dict(dequantized_dict)
