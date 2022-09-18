#!/usr/bin/env bash
#PBS -q SQUID
#PBS --group=hp220285
#PBS -l elapstim_req=2:00:00,cpunum_job=64,gpunum_job=1,memsz_job=4GB
module load BaseCPU

PATH="${HOME}/.local/bin:${HOME}/.ghcup/bin:${PATH}"
cd "${PBS_O_WORKDIR}" || exit
MODEL_PATH="$(pwd)/workspace/mnist/models"
time stack exec -- mnist train -b 500 -M "${MODEL_PATH}" +RTS -N64 -s 
seq 0 9 | while read -r i; do
  IMAGE="./data/mnist/my-digits/mnist-digit-${i}.png"
  time stack exec -- mnist recognise -M "${MODEL_PATH}" "${IMAGE}"
done
