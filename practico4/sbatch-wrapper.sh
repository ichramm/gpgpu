#!/bin/bash

set -e

rm -f salida.out
touch salida.out

#sbatch launch_single.sh "$@"
JOB=$(sbatch launch_single.sh "$@" | awk '{print $4}')

echo "JOB: $JOB"

#sleep 1
counter=0
while true; do
    status=$(scontrol show job "$JOB" | grep JobState | awk '{print $1}' | cut -d= -f2)
    if [[ "$status" =~ ^(COMPLETED|FAILED|CANCELLED)$ ]]; then
        echo -e "\n$status"
        break
    fi
    if (( counter == 10 )); then
        echo -e "\n$status"
        counter=0
    else
        echo -n ". "
        ((counter+=1))
    fi
    sleep 1
done
echo

cat salida.out
