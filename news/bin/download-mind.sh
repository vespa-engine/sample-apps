#! /bin/bash

DATASET=$1
DIR="mind"
TRAIN_DIR="$DIR/train"
DEV_DIR="$DIR/dev"

if [ -z "$HF_TOKEN" ]; then
  echo "HF_TOKEN not set. Get a READ token from https://huggingface.co/settings/tokens"
  echo "Then: export HF_TOKEN=hf_..."
  exit 1
fi

HF_HEADER="Authorization: Bearer $HF_TOKEN"

mkdir -p $TRAIN_DIR
mkdir -p $DEV_DIR

if [ "$DATASET" = "demo" ]; then
  curl -L -H "$HF_HEADER" -o $DIR/train.zip "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_train.zip?download=true"
  curl -L -H "$HF_HEADER" -o $DIR/dev.zip "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_dev.zip?download=true"

elif [ "$DATASET" = "small" ]; then
  curl -L -H "$HF_HEADER" -o $DIR/train.zip "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip?download=true"
  curl -L -H "$HF_HEADER" -o $DIR/dev.zip "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip?download=true"

elif [ "$DATASET" = "large" ]; then
  curl -L -H "$HF_HEADER" -o $DIR/train.zip "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_train.zip?download=true"
  curl -L -H "$HF_HEADER" -o $DIR/dev.zip "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_dev.zip?download=true"

else
  echo "No dataset specified. Use demo|small|large."
  exit 1
fi

# -j: extract flat (ignore paths in zip) since zips contain a nested folder
unzip -j -o $DIR/train.zip -d $TRAIN_DIR
unzip -j -o $DIR/dev.zip -d $DEV_DIR
