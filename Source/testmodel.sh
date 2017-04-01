#!/bin/bash

if [ "$#" -eq 2 ]
then
  validation_set=$1
  model=$2

  ./svm_classify $validation_set $model classification.txt >> classification_accuracy.txt
  mv classification.txt classification/
  mv classification_accuracy.txt classification/

else
  echo "wrong number of arguments. Expecting 2:"
  echo "arg1 = validation_set"
  echo "arg2 = model"
fi
