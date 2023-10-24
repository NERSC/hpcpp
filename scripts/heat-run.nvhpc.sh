#!/bin/bash -le

#
# Reminder: Revert any changes to nvstdpar/CMakeLists.txt and
# nvstdpar/apps/heat-equation/CMakeLists.txt that you did
# for GCC compiler script before running this.
#

set -x

mkdir -p ${HOME}/repos/nvstdpar/build-nvhpc
cd ${HOME}/repos/nvstdpar/build-nvhpc

module use /global/cfs/cdirs/m1759/wwei/nvhpc_23_7/modulefiles

rm -rf ./*
ml unload cudatoolkit cray-mpich
ml cmake/3.24 nvhpc/23.7

cmake .. -DSTDPAR=multicore -DOMP=multicore

make -j

cd ${HOME}/repos/nvstdpar/build-nvhpc/apps/heat-equation

./heat-equation-mdspan -s=50 -n=30000 --time 2>&1 |& tee md.txt

# parallel runs
T=(128 64 32 16 8 4 2 1)

for i in "${T[@]}"; do
    ./heat-equation-omp -s=50 -n=30000 --time --nthreads=${i} 2>&1 |& tee omp-${i}.txt
done

for i in "${T[@]}"; do
    export OMP_NUM_THREADS=${i}
    ./heat-equation-stdpar -s=50 -n=30000 --time 2>&1 |& tee stdpar-${i}.txt
done

unset OMP_NUM_THREADS
