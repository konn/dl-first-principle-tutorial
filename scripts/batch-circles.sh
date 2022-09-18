#!/usr/bin/env bash
#PBS -q SQUID
#PBS --group=hp220285
#PBS -l elapstim_req=1:00:00,cpunum_job=16,gpunum_job=1,memsz_job=4GB
module load BaseCPU

PATH="${HOME}/.local/bin:${HOME}/.ghcup/bin:${PATH}"
cd $PBS_O_WORKDIR
time stack exec -- circles spirals -L 16,25,40 -n 1000 +RTS -N16 -s

