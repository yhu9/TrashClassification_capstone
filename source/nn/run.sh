#!/bin/bash

fnames=""
for image in $(ls resources)
do
    fnames="$fnames resources/$image"
done

echo "processing files: $fnames"

python nn.py train nn $fnames
#python nn.py train nn
