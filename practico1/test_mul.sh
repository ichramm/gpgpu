#!/bin/bash

function run() {
    local test="$1"
    local N="$2"
    local BL="$3"
    echo "Running $test with N=$N, BL=$BL"
    CFLAGS="-D$test" make clean run ARGS="$N $BL" | grep average_duration_ms
}

for N in 32 64 512 1024 2048; do
    run "MULT_SIMPLE" $N "NaN"
    run "MULT_FILA" $N "NaN"
    for BL in 16 32 64 128; do
        if [[ $BL -le $N ]]; then
            run "MULT_BL_SIMPLE" $N $BL
            run "MULT_BL_FILA" $N $BL
        fi
    done
done
