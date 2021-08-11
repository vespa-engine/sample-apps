#! /bin/bash

DATASET=$1
DIR="mind"
TRAIN_DIR="$DIR/train"
DEV_DIR="$DIR/dev"

mkdir -p $TRAIN_DIR
mkdir -p $DEV_DIR

if [ $DATASET = "demo" ]
then
  curl -L -o $DIR/train.zip https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_train.zip
  curl -L -o $DIR/dev.zip https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_dev.zip

elif [ $DATASET = "small" ]
then
  curl -L -o $DIR/train.zip -nd https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip
  curl -L -o $DIR/dev.zip   -nd https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip

elif [ $DATASET = "large" ]
then
  curl -L -o $DIR/train.zip -nd https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip
  curl -L -o $DIR/dev.zip   -nd https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip

else
  echo "No dataset specified. Use demo|small|large."
  exit 1
fi

unzip -o $DIR/train.zip -d $TRAIN_DIR
unzip -o $DIR/dev.zip -d $DEV_DIR

