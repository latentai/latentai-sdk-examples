#!/bin/bash

#  Copyright (c) 2021 by Latent AI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "Latent AI Commercial Software License". Please see the LICENSE
#  file that should have been included as part of this package.


#####################################################################################################################################
# Flow script for TFLite Convert->Eval and TFLite Convert->Compile->Eval path on inceptionv3 trained on Latent AI Open Images dataset
#####################################################################################################################################

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
BASE_OUTPUT_PATH=./example-2/$MODEL_ID
REP_DATASET_IMAGE=`du -a $DATASET_DIR | grep jpg | head -1 | awk '{print $2}'`

# Clean and create fresh folders
rm -rf $BASE_OUTPUT_PATH

mkdir -p $BASE_OUTPUT_PATH/convert
mkdir -p $BASE_OUTPUT_PATH/compile
mkdir -p $BASE_OUTPUT_PATH/evaluate

# make a representative dataset file
echo $REP_DATASET_IMAGE > rep_dataset.txt

# compress and convert Model to tflite.
leip compress --input_path=$MODEL_DIR \
              --output_path=$BASE_OUTPUT_PATH/convert \
              --data_type=uint8 \
              --rep_dataset=rep_dataset.txt

rm rep_dataset.txt

# evaluate uint in tflite
leip evaluate --input_path=$BASE_OUTPUT_PATH/convert \
              --test_path=$DATASET_DIR/index.txt \
              --class_names=$DATASET_DIR/class_names.txt \
              --task=classifier \
              --dataset=custom \
              --preprocessor=uint8
##
## If you'd like, you can now compile that int8 *.cast.tflite file in leip compile (TVM)
## this part of the process is no different than normal models, modulo --input_types flags
##

# compile
leip compile --input_path=$BASE_OUTPUT_PATH/convert \
             --output_path=$BASE_OUTPUT_PATH/compile \
             --input_types=uint8 \
             --storage_int8 false

# evaluate uint8 compiled
leip evaluate --input_path=$BASE_OUTPUT_PATH/compile \
              --test_path=$DATASET_DIR/index.txt \
              --class_names=$DATASET_DIR/class_names.txt \
              --task=classifier \
              --dataset=custom \
              --input_types=uint8 \
              --preprocessor uint8
