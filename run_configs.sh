#!/bin/bash

FILES="./configs/*"
for config in $FILES
do
  echo "Processing $config file..."
  cp $config ./config.py
  python ./main.py
done


