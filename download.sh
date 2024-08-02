#!/bin/bash

# Define the URL for the download
URL="https://github.com/yakhyo/yolov5-crowdhuman-onnx/releases/download/v0.0.1/crowdhuman.onnx"

# Create the weights directory if it does not exist
mkdir -p weights

# Download the file
wget -O weights/crowdhuman.onnx $URL

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Downloaded crowdhuman.onnx to weights/"
else
    echo "Failed to download crowdhuman.onnx"
fi
