#!/bin/bash -le

#SBATCH -N 1
#SBATCH -p cg4-cpu4x120gb-gpu4x80gb
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -o 1d-cpu.o%j
#SBATCH -e 1d-cpu.e%j
#SBATCH -J 1D-CPU

set +x

BUILD_HOME=${HOME}/repos/nvstdpar/build-1d-cpu

mkdir -p ${BUILD_HOME}
cd ${BUILD_HOME}
rm -rf ./*

module unload gcc; module load gcc/12.3.0; module load nvhpc/23.5; module load slurm
export PATH=/home/wwei/install/cmake_3_27_3/bin/:$PATH

# export OMP_PLACES="{0:16},{16:16},{32:16},{48:16},{64:16},{80:16},{96:16},{112:16}"
# export OMP_PROC_BIND=close


oneDimension_size=1000000000
oneDimension_iterations=4000

cmake .. -DCMAKE_BUILD_TYPE=Release -DSTDPAR=multicore -DOMP=multicore -DCMAKE_CXX_COMPILER=$(which nvc++)
make -j

cd ${BUILD_HOME}/apps/1d_stencil

# parallel runs
T=(256 128 64 32 16 8 4 2 1)

unset OMP_NUM_THREADS

for i in "${T[@]}"; do
    echo "1d:omp, threads=${i}"
    srun -n 1 --cpu-bind=none ./stencil_omp --size $oneDimension_size --nt $oneDimension_iterations --nthreads=$i


    echo "1d:stdexec, threads=${i}"
    srun -n 1 --cpu-bind=none ./stencil_stdexec --sch cpu --size $oneDimension_size --nt $oneDimension_iterations --nthreads=$i
done

for i in "${T[@]}"; do
    echo "1d:stdpar, threads=${i}"
    export OMP_NUM_THREADS=${i}
    srun -n 1 --cpu-bind=none ./stencil_stdpar --size $oneDimension_size --nt $oneDimension_iterations
done

unset OMP_NUM_THREADS

echo "1d:serial"
srun -n 1 --cpu-bind=none ./stencil_serial --size $oneDimension_size --nt $oneDimension_iterations

