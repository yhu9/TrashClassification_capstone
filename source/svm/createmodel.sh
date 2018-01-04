#!/bin/bash
CAT_DIR="categories"
TDATA_DIR="trainingdata"
VDATA_DIR="validationdata"
MODEL_DIR="models"
SVM_MODE=2
#http://svmlight.joachims.org/
#linear model = 0
#polynomial = 1
#radial = 2

if [ "$#" -ge 4 ]
then
  groupname=$1
  ingroup_dir=$2
  outgroup_dir=$3
  showFlag=$4
  mode=$5
  out_name="$1_$5"

  ##################################################################################################
  #create ingroup
  ##################################################################################################
  for inImage in $(ls ./$ingroup_dir)
  do
    echo $inImage
    python execute.py ./$ingroup_dir/$inImage tmp1.txt $showFlag $mode
    python parseForSVM.py tmp1.txt tmp2.txt +
    cat tmp2.txt >> ingroup.txt
    rm tmp1.txt
    rm tmp2.txt
  done

  ##################################################################################################
  #create outgroup
  ##################################################################################################
  for outImage in $(ls ./$outgroup_dir)
  do
    echo $outImage
    python execute.py ./$outgroup_dir/$outImage tmp1.txt $showFlag $mode
    python parseForSVM.py tmp1.txt tmp2.txt -
    cat tmp2.txt >> outgroup.txt
    rm tmp1.txt
    rm tmp2.txt
  done

  ##################################################################################################
  #Shuffle instances so that we get a little of each image in both the training and validation set
  ##################################################################################################
  shuf ingroup.txt > tmp1
  rm ingroup.txt
  mv tmp1 ingroup.txt
  shuf outgroup.txt > tmp2
  rm outgroup.txt
  mv tmp2 outgroup.txt

  ##################################################################################################
  #find out which group is shortest
  ##################################################################################################
  len_ingroup=$(wc -l ingroup.txt | grep -o "[0-9]*")
  len_outgroup=$(wc -l outgroup.txt | grep -o "[0-9]*")

  shortest=0
  if [ $len_ingroup -gt $len_outgroup ]
  then
    shortest=$len_outgroup
  else
    shortest=$len_ingroup
  fi
  #half=$(($shortest / 2))
  #half1=$(($half + 1))

  ##################################################################################################
  #create validation data and training data
  ##################################################################################################

  head -n $shortest ingroup.txt >> "$out_name"_training.txt
  head -n $shortest outgroup.txt >> "$out_name"_training.txt
  #head -n $half ingroup.txt >> "$groupname"_training.txt
  #head -n $half outgroup.txt >> "$groupname"_training.txt
  #tail -n +$half1 ingroup.txt >> "$groupname"_validation.txt
  #tail -n +$half1 outgroup.txt >> "$groupname"_validation.txt
  rm ingroup.txt
  rm outgroup.txt

  ##################################################################################################
  #create model
  ##################################################################################################
  ./svm_learn -t $SVM_MODE "$out_name"_training.txt "$out_name".model

  ##################################################################################################
  #organize the files
  ##################################################################################################
  mkdir ./$CAT_DIR/$groupname/$TDATA_DIR
  #mkdir ./$CAT_DIR/$groupname/$VDATA_DIR
  mv "$out_name"_training.txt ./$CAT_DIR/$groupname/$TDATA_DIR
  #mv "$groupname"_validation.txt ./$CAT_DIR/$groupname/$VDATA_DIR
  mv "$out_name".model ./$MODEL_DIR

else
  echo "wrong number of arguments passed. Expecting:"
  echo "arg1 = group_name"
  echo "arg2 = inGroupDirectory"
  echo "arg3 = outGroupDirectory"
  echo "arg4 = showFlag (show/noshow)"
fi

