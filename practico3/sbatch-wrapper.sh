#!/bin/bash

rm -f salida.out
touch salida.out

BEF=$(stat -c '%Y' salida.out)

#sbatch launch_single.sh "$@"
JOB=$(sbatch launch_single.sh "$@" | awk '{print $4}')

#sleep 1
while true; do
    status=$(scontrol show job "$JOB" | grep JobState | awk '{print $1}' | cut -d= -f2)
    if [ "$status" == "COMPLETED" ] || [ "$status" == "FAILED" ]; then
        break
    fi
    echo -n ". "
    sleep 1
done
echo

cat salida.out
