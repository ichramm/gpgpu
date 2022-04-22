#!/bin/bash

function run() {
    local test="$1"
    local N="$2"
    local BL="$3"
    echo -e "Running $test with N=$N, BL=$BL"
    CFLAGS="-D$test" make clean cachegrind ARGS="$N $BL"
}

N=1152

for BL in 16 18 32; do
    if [[ $BL -le $N ]]; then
        run "MULT_BL_FILA" $N $BL
    fi
done
