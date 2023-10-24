#!/bin/bash -le

#
# Reminder: Revert any changes to nvstdpar/CMakeLists.txt and
# nvstdpar/apps/heat-equation/CMakeLists.txt that you did
# for GCC compiler script before running this.
#

#SBATCH -A nstaff
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH -o fft-cpu.o%j
#SBATCH -e fft-cpu.e%j
#SBATCH -J FFT-CPU

set +x

mkdir -p ${HOME}/repos/nvstdpar/build-fft-cpu
cd ${HOME}/repos/nvstdpar/build-fft-cpu
rm -rf ./*

ml use /global/cfs/cdirs/m1759/wwei/nvhpc_23_7/modulefiles
ml nvhpc/23.7
ml cmake/3.24

cmake .. -DSTDPAR=multicore -DOMP=multicore

make -j fft-serial fft-stdexec fft-stdpar

cd ${HOME}/repos/nvstdpar/build-fft-cpu/apps/fft

D=1073741824

set -x

srun -n 1 --cpu-bind=cores ./fft-serial -N ${D} --time 2>&1 |& tee fft-serial.txt

unset OMP_NUM_THREADS

# parallel runs
T=(128 64 32 16 8 4 2 1)

for i in "${T[@]}"; do
    srun -n 1 --cpu-bind=cores ./fft-stdpar -N ${D} --time --sch=cpu --nthreads=${i} 2>&1 |& tee fft-stdpar-cpu-${i}.txt
done