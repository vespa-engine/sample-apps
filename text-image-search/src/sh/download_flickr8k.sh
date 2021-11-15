#!/usr/bin/env sh

mkdir -p data
cd data
curl -L -o data.zip https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
unzip data.zip
cd ..
