#!/bin/bash

if [ "$#" -eq 2 ]
then
  validationSet=$1
  model=$2

  ./svm_classify $validationSet $model classification.txt >> classification_accuracy.txt
  cat classification_accuracy.txt
  mv classification.txt classification/
  mv classification_accuracy.txt classification/

else
  echo "wrong number of arguments. Expecting 2:"
  echo "arg1 = validationSet"
  echo "arg2 = model"
fi
