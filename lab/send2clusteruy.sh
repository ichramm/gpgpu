#!/bin/bash

# change this and only this
FILES=(sbatch-wrapper.sh launch_single.sh Makefile include src);

DIR=$(basename "$(dirname "$(readlink -f "$0")")");

ssh clusteruy mkdir -p "$DIR"

scp -r "${FILES[@]}" clusteruy:~/"$DIR"
