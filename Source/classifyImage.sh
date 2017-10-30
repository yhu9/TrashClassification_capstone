#!/bin/bash

####################################################################
#you can define your models you want to use here only takes 4
CLASS_DIR="classification"
####################################################################
#Main program


if [ "$#" -eq 2 ]
then
  imageFileIn=$1
  mode=$2
model1="models/construction_waste_$mode.model"
model2="models/treematter_$mode.model"
model3="models/plywood_$mode.model"
model4="models/cardboard_$mode.model"
model5="models/trashbags_$mode.model"

  python execute.py $imageFileIn tmp.txt noshow $mode
  python parseForSVM.py tmp.txt tmp2.txt +
  ./svm_classify tmp2.txt $model1 classification1.txt > classification_accuracy1.txt
  ./svm_classify tmp2.txt $model2 classification2.txt > classification_accuracy2.txt
  ./svm_classify tmp2.txt $model3 classification3.txt > classification_accuracy3.txt
  ./svm_classify tmp2.txt $model4 classification4.txt > classification_accuracy4.txt
  ./svm_classify tmp2.txt $model5 classification5.txt > classification_accuracy5.txt
  rm tmp.txt
  rm tmp2.txt

  python stichit.py $imageFileIn classification1.txt classification2.txt classification3.txt classification4.txt classification5.txt
  mv classification1.txt $CLASS_DIR/classification1.txt
  mv classification2.txt $CLASS_DIR/classification2.txt
  mv classification3.txt $CLASS_DIR/classification3.txt
  mv classification4.txt $CLASS_DIR/classification4.txt
  mv classification5.txt $CLASS_DIR/classification5.txt
  mv classification_accuracy1.txt $CLASS_DIR/classification_accuracy1.txt
  mv classification_accuracy2.txt $CLASS_DIR/classification_accuracy2.txt
  mv classification_accuracy3.txt $CLASS_DIR/classification_accuracy3.txt
  mv classification_accuracy4.txt $CLASS_DIR/classification_accuracy4.txt
  mv classification_accuracy5.txt $CLASS_DIR/classification_accuracy5.txt

else
  echo "wrong number of arguments. Expecting 1"
  echo "arg1 = imageFileIn"
fi
