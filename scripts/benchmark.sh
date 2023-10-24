#!/bin/bash -le

#SBATCH -A nstaff

#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH -G 4 
#SBATCH -t 6:00:00
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

#SBATCH -o nvstdpar_stencil_final_benchmark.out
#SBATCH -e nvstdpar_stencil_final_benchmark.err

ml use /global/cfs/cdirs/m1759/wwei/nvhpc_23_7/modulefiles
ml unload cudatoolkit
ml gcc/12.2 cmake/3.24 nvhpc-hpcx/23.7

oneDimention_size=1000000000
oneDimention_iterations=4000 
twoDimention_size=46000
twoDimention_iterations=1000

cpu_build=build_multicore_benchmark

# mkdir -p ${HOME}/src/nvstdpar/$cpu_build
# cd ${HOME}/src/nvstdpar/$cpu_build
# rm -rf ./*
# cmake .. -DCMAKE_BUILD_TYPE=Release -DSTDPAR=multicore -DOMP=multicore
# make -j

gpu_build=build_gpu_benchmark

# mkdir -p ${HOME}/src/nvstdpar/$gpu_build
# cd ${HOME}/src/nvstdpar/$gpu_build
# rm -rf ./*
# cmake .. -DCMAKE_BUILD_TYPE=Release -DSTDPAR=gpu -DOMP=gpu
# make -j

cd ${HOME}/src/nvstdpar/$cpu_build/apps/1d_stencil
echo "1D_serial"
time ./stencil_serial --size $oneDimention_size --nt $oneDimention_iterations
echo "1D_stdpar cpu" 
time ./stencil_stdpar --size $oneDimention_size --nt $oneDimention_iterations

cd ${HOME}/src/nvstdpar/$gpu_build/apps/1d_stencil
echo "1D_stdpar gpu" 
time ./stencil_stdpar --size $oneDimention_size --nt $oneDimention_iterations
echo "1D_stdexec gpu"
time ./stencil_stdexec --sch gpu --size $oneDimention_size --nt $oneDimention_iterations
echo "1D_stdexec multigpu"
time ./stencil_stdexec --sch multigpu --size $oneDimention_size --nt $oneDimention_iterations
echo "1D_cuda"
time ./stencil_cuda --size $oneDimention_size --nt $oneDimention_iterations

cd ${HOME}/src/nvstdpar/$cpu_build/apps/heat-equation
echo "2D_serial"
time ./heat-equation-serial -n=$twoDimention_size -s=$twoDimention_iterations --time
echo "2D_stdpar cpu" 
time ./heat-equation-stdpar -n=$twoDimention_size -s=$twoDimention_iterations --time

cd ${HOME}/src/nvstdpar/$gpu_build/apps/heat-equation
echo "2D_stdpar gpu" 
time ./heat-equation-stdpar -n=$twoDimention_size -s=$twoDimention_iterations --time
echo "2D_stdexec gpu"
time ./heat-equation-stdexec --sch gpu -n=$twoDimention_size -s=$twoDimention_iterations --time
echo "2D_stdexec multigpu"
time ./heat-equation-stdexec --sch multigpu -n=$twoDimention_size -s=$twoDimention_iterations --time
echo "2D_cuda"
time ./heat-equation-cuda -n=$twoDimention_size -s=$twoDimention_iterations --time
