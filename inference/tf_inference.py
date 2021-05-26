#  Copyright (c) 2019 by LatentAI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "LatentAI Commercial Software License".
#  Please see the LICENSE file that should have been included as part of
#  this package.
#
# @file    tf_inference.py
#
# @author  Videet Parekh
#
# @date    Wed 16 Dec 20
#
# @brief   TF inference engine designed with the same interface as leip_inference for parallel comparison


# from time import time
# import tensorflow as tf
import glob
import os
import logging
import utils.common_utils as utils
import argparse
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)


class TFModel():
    def __init__(self, base_path, context, config):
        self.context = context
        self.config = config
        self.load(base_path)

    def load(self, base):
        h5_path = glob.glob(os.path.join(base, '*.h5'))[0]
        self.model = utils.load_keras_model(h5_path)

    def infer(self, data):
        # Here's how you may measure runtime speed
        # start = time()
        output_data = self.model.predict(data)
        # end = time()
        pred = {'label': output_data}
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
    base = args.input_path
    test_path = args.test_path
    class_names = args.class_names
    data_type = args.data_type
    preprocessor = args.preprocessor
    context = args.inference_context
    loglevel = args.loglevel

    # Set Logger Parameters
    logging.basicConfig(level=utils.get_numeric_loglevel(loglevel))

    # Get class_names for model
    with open(class_names) as f:
        synset = f.readlines()

    config = utils.load_json(os.path.join(base, 'model_schema.json'))
    config['input_shapes'] = utils.parse_input_shapes(config['input_shapes'])

    # Load dataset and collect preprocessor function
    data_index = utils.load_index(test_path)
    preprocessor = utils.collect_preprocessor(preprocessor)

    # Create model object for inference
    model = TFModel(base, context, config)

    acc = 0

    # Loop over data and call infer()
    for data in data_index:
        # Load and preprocess image
        img = utils.collect_image(data[0], data_type, preprocessor, config['input_shapes'])

        # Infer
        pred = model.infer(img)
        pred_label = np.argmax(pred['label'])
        acc += 1 if pred_label == data[1] else 0

    print(acc*100/len(data_index))
