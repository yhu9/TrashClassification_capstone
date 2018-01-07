#!/bin/bash

CATEGORY1="cardboard"
CATEGORY2="construction_waste"
CATEGORY3="plywood"
CATEGORY4="treematter"
INDIR="/home/masa/Projects/TrashClassification_capstone/source/categories"
OUTDIR="./resources"
MODE="both"

#create the feature files
#################################################################################################
count=1
cat_dir="$INDIR/$CATEGORY1/ingroup"
for image in $(ls $cat_dir)
do
    out_name="$CATEGORY1""$count".txt
    echo "processing: $image >> $out_name"
    count=$(($count + 1))
    python ../execute.py $cat_dir/$image $OUTDIR/$out_name noshow $MODE
done

count=1
cat_dir="$INDIR/$CATEGORY1/ingroup"
for image in $(ls $cat_dir)
do
    out_name="$CATEGORY2""$count".txt
    echo "processing: $image >> $out_name"
    count=$(($count + 1))
    python ../execute.py $cat_dir/$image $OUTDIR/$out_name noshow $MODE
done

count=1
cat_dir="$INDIR/$CATEGORY1/ingroup"
for image in $(ls $cat_dir)
do
    out_name="$CATEGORY3""$count".txt
    echo "processing: $image >> $out_name"
    count=$(($count + 1))
    python ../execute.py $cat_dir/$image $OUTDIR/$out_name noshow $MODE
done

count=1
cat_dir="$INDIR/$CATEGORY1/ingroup"
for image in $(ls $cat_dir)
do
    out_name="$CATEGORY4""$count".txt
    echo "processing: $image >> $out_name"
    count=$(($count + 1))
    python ../execute.py $cat_dir/$image $OUTDIR/$out_name noshow $MODE
done

#################################################################################################


