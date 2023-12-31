#!/bin/bash -le

#
# Reminder: Revert any changes to nvstdpar/CMakeLists.txt and
# nvstdpar/apps/heat-equation/CMakeLists.txt that you did
# for GCC compiler script before running this.
#

#SBATCH -A nstaff
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH -o fft-cpu.o%j
#SBATCH -e fft-cpu.e%j
#SBATCH -J FFT-CPU

set +x

BUILD_HOME=${HOME}/repos/nvstdpar/build-fft-cpu

mkdir -p ${BUILD_HOME}
cd ${BUILD_HOME}
rm -rf ./*

ml unload cudatoolkit
ml use /global/cfs/cdirs/m1759/wwei/nvhpc_23_7/modulefiles
ml nvhpc/23.7
# need this for GLIBC
ml gcc/12.2.0
ml cmake/3.24

cmake .. -DSTDPAR=multicore -DOMP=multicore -DCMAKE_CXX_COMPILER=$(which nvc++)

make -j fft-serial fft-stdexec fft-stdpar

cd ${BUILD_HOME}/apps/fft

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
