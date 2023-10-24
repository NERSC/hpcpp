#!/bin/bash -le

#
# Reminder: Revert any changes to nvstdpar/CMakeLists.txt and
# nvstdpar/apps/heat-equation/CMakeLists.txt that you did
# for GCC compiler script before running this.
#

#SBATCH -A nstaff_g
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --gpu-bind=none
#SBATCH -o fft-gpu.o%j
#SBATCH -e fft-gpu.e%j
#SBATCH -J FFT-GPU

set +x

mkdir -p ${HOME}/repos/nvstdpar/build-fft-gpu
cd ${HOME}/repos/nvstdpar/build-fft-gpu
rm -rf ./*

ml unload cudatoolkit
ml PrgEnv-nvhpc
ml use /global/cfs/cdirs/m1759/wwei/nvhpc_23_7/modulefiles
ml nvhpc/23.7

cmake .. -DSTDPAR=gpu -DOMP=gpu

make -j fft-stdexec fft-stdpar

cd ${HOME}/repos/nvstdpar/build-fft-gpu/apps/fft

D=1073741824

set -x

srun -n 1 ./fft-stdexec -N ${D} --time --sch=gpu 2>&1 |& tee fft-stdexec.txt

srun -n 1  ./fft-stdpar -N ${D} --time 2>&1 |& tee fft-stdpar.txt

srun -n 1 ./fft-stdpar -N ${D} --time --sch=multigpu 2>&1 |& tee fft-stdexec-multigpu.txt
