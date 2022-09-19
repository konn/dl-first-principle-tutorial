#!/usr/bin/env bash
#PBS -q SQUID
#PBS --group=hp220285
#PBS -l elapstim_req=2:00:00,cpunum_job=32,gpunum_job=1,memsz_job=4GB
module load BaseCPU

PATH="${HOME}/.local/bin:${HOME}/.ghcup/bin:${PATH}"
cd "${PBS_O_WORKDIR}" || exit
MODEL_PATH="$(pwd)/workspace/mnist/models-adam"
time stack exec -- mnist train --adam -b 1000 --dt 0.01 -M "${MODEL_PATH}" +RTS -N32 -s 
seq 0 9 | while read -r i; do
  IMAGE="./data/mnist/my-digits/mnist-digit-${i}.png"
  time stack exec -- mnist recognise -M "${MODEL_PATH}" "${IMAGE}"
done
