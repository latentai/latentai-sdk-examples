#!/bin/bash

#  Copyright (c) 2021 by Latent AI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "Latent AI Commercial Software License". Please see the LICENSE
#  file that should have been included as part of this package.


########################################################################################################
# Flow script for Compress->Compile->Eval path on inceptionv3 trained on Latent AI Open Images dataset
########################################################################################################

set -e  # quit on errors
set -x  # print all commands before executing

# Define the model name and variant. To get a list of supported models on the zoo, run "leip zoo list".
MODEL_ID="inceptionv3"
MODEL_VARIANT_ID="keras-open-images-10-classes"

# Define the corresponding dataset name and variant. To get a list of supported models on the zoo, run "leip zoo list".
DATASET_ID="open-images-10-classes"
DATASET_VARIANT_ID="eval"

# Download model and dataset
leip zoo download --model_id $MODEL_ID --variant_id $MODEL_VARIANT_ID
leip zoo download --dataset_id $DATASET_ID --variant_id $DATASET_VARIANT_ID

# Define paths for dataset, model, output.
MODEL_DIR=workspace/models/$MODEL_ID/$MODEL_VARIANT_ID
DATASET_DIR=workspace/datasets/$DATASET_ID/$DATASET_VARIANT_ID
BASE_OUTPUT_PATH=./example-1/$MODEL_ID

# Clean and create fresh folders
rm -rf $BASE_OUTPUT_PATH

mkdir -p $BASE_OUTPUT_PATH/compress
mkdir -p $BASE_OUTPUT_PATH/compile
mkdir -p $BASE_OUTPUT_PATH/evaluate

# Compress the model with an Asymmetric quantizer to an 8 bit range.
leip compress --input_path $MODEL_DIR \
              --output_path $BASE_OUTPUT_PATH/compress \
              --bits 8 \
              --quantizer asymmetric

# Compile the model to a represent weights as int8.
leip compile --input_path $BASE_OUTPUT_PATH/compress \
             --output_path $BASE_OUTPUT_PATH/compile  \
             --input_types=float32 \
             --data_type=int8

# Evaluate the accuracy of the model. Weights are rescaled to float32 for inference.
leip evaluate --input_path $BASE_OUTPUT_PATH/compile \
              --output_path $BASE_OUTPUT_PATH/evaluate \
              --test_path $DATASET_DIR/index.txt \
              --class_names $DATASET_DIR/class_names.txt \
              --task=classifier \
              --dataset=custom \
              --input_types=float32
