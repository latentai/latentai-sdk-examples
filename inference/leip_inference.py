    #  Copyright (c) 2019 by LatentAI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "LatentAI Commercial Software License".
#  Please see the LICENSE file that should have been included as part of
#  this package.
#
# @file    leip_inference.py
#
# @author  Videet Parekh
#
# @date    Wed, 16 Dec 20
'''
This script is designed to run a model compiled with LEIP compile. By default, LEIP compile generates 4 artifacts:
             - modelDescription.json - Model graph
             - modelParams.params    - Model weights
             - modelLibrary.so       - Model layers as callable functions
             - modelSchema.json      - Model descriptor file containing model metadata

A fifth artifact - quantParams.params - is generated when compiled with float32 inputs and to data-type (u)int8

This script accepts a path to a directory containing ALL the above files, a path to a database index file in
LEIP Dataset format, a classnames filepath in a LEIP Dataset format, and several other arguments to effectively
execute the model.
'''

# from time import time
import logging
import utils.common_utils as utils
import argparse
import numpy as np


class LEIPModel():
    def __init__(self, base_path, context, config):
        self.context = context
        self.config = config
        self.load(base_path, self.context)

    def load(self, base, context):
        # Build runtime
        logging.debug("\nBuilding runtime\n")
        self.module = utils.create_leip_runtime_module(base, context)

        # Dry run to load model completely into memory
        self.module.run()

    def infer(self, data):
        self.module.set_input(self.config['input_names'], data)

        # Execute runtime

        # Here's how you may time different parts of the runtime
        # start = time()
        self.module.run()
        output = self.module.get_output(0).asnumpy()
        # end = time()

        pred = {'label': output}
        return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None, required=True, help='Path to model directory.')
    parser.add_argument('--test_path', type=str, default=None, required=True, help='Path to output test file')
    parser.add_argument('--class_names', type=str, default=None, required=True, help='Path to class names list.')
    parser.add_argument('--data_type', type=str, default="float32", required=False, help='Data Type.')
    parser.add_argument('--preprocessor', type=str, default="none", required=False, help='Preprocessor function')
    parser.add_argument('--inference_context', type=str, default="none", required=False, help='cpu/gpu/cuda.')
    parser.add_argument('--loglevel', type=str, default="WARNING", required=False, help='Logging verbosity.')

    args = parser.parse_args()

    base_path = args.input_path
    test_path = args.test_path
    class_names = args.class_names
    data_type = args.data_type
    context = args.inference_context
    preprocessor = args.preprocessor
    loglevel = args.loglevel

    # Set Logger Parameters
    logging.basicConfig(level=utils.get_numeric_loglevel(loglevel))

    # Get class_names for model
    with open(class_names) as f:
        synset = f.readlines()

    # Load dataset and collect preprocessor function
    data_index = utils.load_index(test_path)
    preprocessor = utils.collect_preprocessor(preprocessor)

    # Get model schema for configuration of runtime
    config = utils.load_json(base_path + 'model_schema.json')
    config['input_shapes'] = utils.parse_input_shapes(config['input_shapes'])

    # Create model object for inference
    model = LEIPModel(base_path, context, config)

    acc = 0
    # Loop over data and call infer()
    for data in data_index:
        # Load and preprocess image
        img = utils.collect_image(data[0], data_type, preprocessor, config['input_shapes'])

        # Infer
        pred = model.infer(img)
        pred_label = np.argmax(pred['label'])
        acc += 1 if pred_label == data[1] else 0

    print("Accuracy: ", acc*100/len(data_index))
