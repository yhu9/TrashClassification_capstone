#!/bin/bash

#create color models
#./createmodel.sh cardboard_color categories/cardboard/ingroup categories/cardboard/outgroup noshow color &&
#./createmodel.sh construction_waste_color categories/construction_waste/ingroup categories/construction_waste/outgroup noshow color &&
#./createmodel.sh plywood_color categories/plywood/ingroup categories/plywood/outgroup noshow color &&
#./createmodel.sh trashbags_color categories/trashbags/ingroup categories/trashbags/outgroup noshow color &&
#./createmodel.sh treematter_color categories/treematter/ingroup categories/treematter/outgroup noshow color &&


#create edge models
./createmodel.sh cardboard_edge categories/cardboard/ingroup categories/cardboard/outgroup noshow edge &&
./createmodel.sh construction_waste_edge categories/construction_waste/ingroup categories/construction_waste/outgroup noshow edge &&
./createmodel.sh plywood_edge categories/plywood/ingroup categories/plywood/outgroup noshow edge &&
./createmodel.sh trashbags_edge categories/trashbags/ingroup categories/trashbags/outgroup noshow edge &&
./createmodel.sh treematter_edge categories/treematter/ingroup categories/treematter/outgroup noshow edge


#create model using both
#./createmodel.sh cardboard_both categories/cardboard/ingroup categories/cardboard/outgroup noshow both &&
#./createmodel.sh construction_waste_both categories/construction_waste/ingroup categories/construction_waste/outgroup noshow both &&
#./createmodel.sh plywood_both categories/plywood/ingroup categories/plywood/outgroup noshow both &&
#./createmodel.sh trashbags_both categories/trashbags/ingroup categories/trashbags/outgroup noshow both &&
#./createmodel.sh treematter_both categories/treematter/ingroup categories/treematter/outgroup noshow both
