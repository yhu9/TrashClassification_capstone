#!/bin/bash

if [ "$#" -eq 2 ]
then
  imageFileIn=$1
  modelIn=$2
  python execute.py $imageFileIn tmp.txt noshow
  python parseForSVM.py tmp.txt tmp2.txt +
  ./svm_classify tmp2.txt $modelIn classification.txt >> classification_accuracy.txt
  rm tmp.txt
  rm tmp2.txt
  cat classification_accuracy.txt
  mv classification.txt classification/classification.txt
  mv classification_accuracy.txt classification/classification_accuracy.txt

  python stichit.py $imageFileIn classification/classification.txt

else
  echo "wrong number of arguments. Expecting 2"
  echo "arg1 = imageFileIn"
  echo "arg2 = modelIn"
fi
