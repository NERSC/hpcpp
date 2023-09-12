#!/bin/bash -le

#
# Read and do the following two steps before running this script:
#
#
# 1. In nvstdpar/CMakeLists.txt, replace the following line:
#
# set(CMAKE_CXX_FLAGS
#    "${CMAKE_CXX_FLAGS} -stdpar=${STDPAR} -mp=${OMP} --gcc-toolchain=/opt/cray/pe/gcc/12.2.0/bin/ -pthread"
#
# with
#
# set(CMAKE_CXX_FLAGS
#    "${CMAKE_CXX_FLAGS} -fopenmp -pthread"
#
#
# 2. In nvstdpar/apps/heat-equation/CMakeLists.txt, replace the following line:
#
# target_link_libraries(${exec_name} PUBLIC ${MPI_LIBS} stdexec)
#
# with
#
# target_link_libraries(${exec_name} PUBLIC ${MPI_LIBS} stdexec tbb)
#
#

set -x

mkdir -p ${HOME}/repos/nvstdpar/build-gcc
cd ${HOME}/repos/nvstdpar/build-gcc

rm -rf ./*
ml cmake/3.24 gcc/12.2 cudatoolkit/12.0
ml unload cray-mpich

cmake .. -DSTDPAR=multicore -DOMP=multicore -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_CUDA_HOST_COMPILER=$(which g++)

make -j heat-equation-omp heat-equation-mdspan heat-equation-stdpar

cd ${HOME}/repos/nvstdpar/build-gcc/apps/heat-equation

./heat-equation-mdspan -s=50 -n=30000 --time 2>&1 |& tee gcc-md.txt

# parallel runs
T=(128 64 32 16 8 4 2 1)

for i in "${T[@]}"; do
    ./heat-equation-omp -s=50 -n=30000 --time --nthreads=${i} 2>&1 |& tee gcc-omp-${i}.txt
done

# will use 128 threads anyway
./heat-equation-stdpar -s=50 -n=30000 --time 2>&1 |& tee gcc-stdpar-${i}.txt

