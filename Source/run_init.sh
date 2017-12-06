#!/bin/bash

#create color models
./createmodel.sh cardboard categories/cardboard/ingroup categories/cardboard/outgroup noshow color &&
./createmodel.sh construction_waste categories/construction_waste/ingroup categories/construction_waste/outgroup noshow color &&
./createmodel.sh plywood categories/plywood/ingroup categories/plywood/outgroup noshow color &&
./createmodel.sh treematter categories/treematter/ingroup categories/treematter/outgroup noshow color &&


#create edge models
./createmodel.sh cardboard categories/cardboard/ingroup categories/cardboard/outgroup noshow edge &&
./createmodel.sh construction_waste categories/construction_waste/ingroup categories/construction_waste/outgroup noshow edge &&
./createmodel.sh plywood categories/plywood/ingroup categories/plywood/outgroup noshow edge &&
./createmodel.sh treematter categories/treematter/ingroup categories/treematter/outgroup noshow edge &&


#create model using both
./createmodel.sh cardboard categories/cardboard/ingroup categories/cardboard/outgroup noshow both &&
./createmodel.sh construction_waste categories/construction_waste/ingroup categories/construction_waste/outgroup noshow both &&
./createmodel.sh plywood categories/plywood/ingroup categories/plywood/outgroup noshow both &&
./createmodel.sh treematter categories/treematter/ingroup categories/treematter/outgroup noshow both
