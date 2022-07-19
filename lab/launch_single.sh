#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=4096
#SBATCH --time=00:02:00

#SBATCH --partition=besteffort
# SBATCH --partition=normal

#SBATCH --qos=besteffort_gpu
# SBATCH --qos=gpu

#SBATCH --gres=gpu:1
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=mi@correo
#SBATCH -o salida.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

source /etc/profile.d/modules.sh

cd ~/lab || exit

# GCC 4.8.5 in clusteruy
#NVCCFLAGS=-std=c++11 make run ARGS="$*"

"$@"
