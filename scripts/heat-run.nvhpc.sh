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
#SBATCH -o heat-cpu.o%j
#SBATCH -e heat-cpu.e%j
#SBATCH -J HEAT-CPU

set +x

BUILD_HOME=${HOME}/repos/nvstdpar/build-heat-cpu

mkdir -p ${BUILD_HOME}
cd ${BUILD_HOME}
rm -rf ./*

ml unload cudatoolkit
ml use /global/cfs/cdirs/m1759/wwei/nvhpc_23_7/modulefiles
ml nvhpc/23.7
# needed for GLIBC
ml gcc/12.2.0
ml cmake/3.24

cmake .. -DSTDPAR=multicore -DOMP=multicore -DCMAKE_CXX_COMPILER=$(which nvc++)

make -j heat-equation-omp heat-equation-serial heat-equation-stdexec heat-equation-stdpar

cd ${BUILD_HOME}/apps/heat-equation

# parallel runs
T=(256 128 64 32 16 8 4 2 1)

unset OMP_NUM_THREADS

for i in "${T[@]}"; do
    echo "heat:omp, threads=${i}"
    srun -n 1 --cpu-bind=cores ./heat-equation-omp -s=50 -n=30000 --time --nthreads=${i}

    echo "heat:stdexec, threads=${i}"
    srun -n 1 --cpu-bind=cores ./heat-equation-stdexec -s=50 -n=30000 --time --nthreads=${i}
done

for i in "${T[@]}"; do
    echo "heat:stdpar, threads=${i}"
    export OMP_NUM_THREADS=${i}
    srun -n 1 --cpu-bind=cores ./heat-equation-stdpar -s=50 -n=30000 --time
done

unset OMP_NUM_THREADS

echo "heat:serial"
srun -n 1 --cpu-bind=cores ./heat-equation-serial -s=50 -n=30000 --time
