#!/bin/bash

# change this and only this
FILES=(sbatch-wrapper.sh launch_single.sh Makefile src);

DIR=$(basename "$(dirname "$(readlink -f "$0")")");
scp -r "${FILES[@]}" clusteruy:~/"$DIR"
