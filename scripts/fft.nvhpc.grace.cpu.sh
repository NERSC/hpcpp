#!/bin/bash -le

#
# Reminder: Revert any changes to nvstdpar/CMakeLists.txt and
# nvstdpar/apps/heat-equation/CMakeLists.txt that you did
# for GCC compiler script before running this.
#

#SBATCH -N 1
#SBATCH -p cg4-cpu4x120gb-gpu4x80gb
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -o fft-cpu.o%j
#SBATCH -e fft-cpu.e%j
#SBATCH -J FFT-CPU

set +x

mkdir -p ${HOME}/repos/nvstdpar/build-fft-cpu
cd ${HOME}/repos/nvstdpar/build-fft-cpu
rm -rf ./*

module unload gcc; module load gcc/12.3.0; module load nvhpc/23.5; module load slurm
export PATH=/home/wwei/install/cmake_3_27_3/bin/:$PATH

cmake .. -DCMAKE_BUILD_TYPE=Release -DSTDPAR=multicore -DOMP=multicore -DCMAKE_CXX_COMPILER=$(which nvc++)

make -j fft-serial fft-stdexec fft-stdpar

cd ${HOME}/repos/nvstdpar/build-fft-cpu/apps/fft

D=(536870912 1073741824)

# parallel runs
T=(256 128 64 32 16 8 4 2 1)

for d in "${D[@]}"; do
    for i in "${T[@]}"; do
        echo "stdexec:cpu for ${d}, threads=${i}"
        srun -n 1 --cpu-bind=none ./fft-stdexec -N ${d} --time --sch=cpu --nthreads=${i}

        echo "stdpar:cpu for ${d}, threads=${i}"
        export OMP_NUM_THREADS=${i}
        srun -n 1 --cpu-bind=none ./fft-stdpar -N ${d} --time --nthreads=${i}
    done
done

unset OMP_NUM_THREADS

for d in "${D[@]}"; do
    echo "serial for ${d}"
    srun -n 1 --cpu-bind=none ./fft-serial -N ${d} --time
done
