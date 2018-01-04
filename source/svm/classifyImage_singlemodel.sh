#!/bin/bash

CLASS_DIR="classification"

if [ "$#" -ge 2 ]
then
  imageFileIn=$1
  modelIn=$2
  mode=$3

  python execute_classification.py $imageFileIn tmp.txt noshow $mode
  python parseForSVM.py tmp.txt tmp2.txt +
  ./svm_classify tmp2.txt $modelIn classification.txt > classification_accuracy.txt
  rm tmp.txt
  rm tmp2.txt
  cat classification_accuracy.txt
  mv classification.txt $CLASS_DIR/classification.txt
  mv classification_accuracy.txt $CLASS_DIR/classification_accuracy.txt

  python stichit.py $imageFileIn $CLASS_DIR/classification.txt

else
  echo "wrong number of arguments. Expecting 2"
  echo "arg1 = imageFileIn"
  echo "arg2 = modelIn"
fi
