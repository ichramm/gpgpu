#!/bin/bash

# change this and only this
FILES=(launch_single.sh Makefile practico2.cu secreto.txt);


DIR=$(basename "$(dirname "$(readlink -f "$0")")");
scp "${FILES[@]}" clusteruy:~/"$DIR"
