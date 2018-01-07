#!/bin/bash

fnames=""
for image in $(ls resources)
do
    fnames="$fnames resources/$image"
done

echo "processing files: $fnames"

python nn.py train nn resources/
#python nn.py train nn
