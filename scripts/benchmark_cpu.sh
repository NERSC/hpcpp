#!/bin/bash -le

#SBATCH -A nstaff

#SBATCH -C cpu 
#SBATCH --qos=regular
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=128

#SBATCH -o nvstdpar_cpu_node.out
#SBATCH -e nvstdpar_cpu_node.err

export OMP_PLACES="{0:16},{16:16},{32:16},{48:16},{64:16},{80:16},{96:16},{112:16}"
export OMP_PROC_BIND=close

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
# cmake .. -DSTDPAR=multicore -DOMP=multicore
# make -j

gpu_build=build_gpu_benchmark
# mkdir -p ${HOME}/src/nvstdpar/$gpu_build
# cd ${HOME}/src/nvstdpar/$gpu_build
# rm -rf ./*
# cmake .. -DSTDPAR=gpu -DOMP=gpu
# make -j

# parallel runs
T=(128 64 32 16 8 4 2 1)

echo "running static scheduler"
export OMP_SCHEDULE="static"
for i in "${T[@]}"; do
    export OMP_NUM_THREADS=$i
    echo "running number of threads: $i"
    cd ${HOME}/src/nvstdpar/$cpu_build/apps/1d_stencil
    echo "1D_stdpar cpu" 
    time srun -n 1 --cpu-bind=cores ./stencil_stdpar --size $oneDimention_size --nt $oneDimention_iterations
    echo "1D_stdexec cpu"
    time srun -n 1 --cpu-bind=cores ./stencil_stdexec --sch cpu --size $oneDimention_size --nt $oneDimention_iterations --nthreads=$i
    echo "1D_omp"
    time srun -n 1 --cpu-bind=cores ./stencil_omp --size $oneDimention_size --nt $oneDimention_iterations --nthreads=$i

    cd ${HOME}/src/nvstdpar/$cpu_build/apps/heat-equation
    echo "2D_stdpar cpu" 
    time srun -n 1 --cpu-bind=cores ./heat-equation-stdpar -n=$twoDimention_size -s=$twoDimention_iterations --time 
    echo "2D_stdexec cpu"
    time srun -n 1 --cpu-bind=cores ./heat-equation-stdexec --sch cpu -n=$twoDimention_size -s=$twoDimention_iterations --time --nthreads=$i
    echo "2D_omp"
    time srun -n 1 --cpu-bind=cores ./heat-equation-omp -n=$twoDimention_size -s=$twoDimention_iterations --time --nthreads=$i
done

echo "running dynamic scheduler"
export OMP_SCHEDULE="dynamic"
for i in "${T[@]}"; do
    export OMP_NUM_THREADS=$i
    echo "running number of threads: $i"
    cd ${HOME}/src/nvstdpar/$cpu_build/apps/1d_stencil
    echo "1D_stdpar cpu" 
    time srun -n 1 --cpu-bind=cores ./stencil_stdpar --size $oneDimention_size --nt $oneDimention_iterations
    echo "1D_stdexec cpu"
    time srun -n 1 --cpu-bind=cores ./stencil_stdexec --sch cpu --size $oneDimention_size --nt $oneDimention_iterations --nthreads=$i
    echo "1D_omp"
    time srun -n 1 --cpu-bind=cores ./stencil_omp --size $oneDimention_size --nt $oneDimention_iterations --nthreads=$i

    cd ${HOME}/src/nvstdpar/$cpu_build/apps/heat-equation
    echo "2D_stdpar cpu" 
    time srun -n 1 --cpu-bind=cores ./heat-equation-stdpar -n=$twoDimention_size -s=$twoDimention_iterations --time 
    echo "2D_stdexec cpu"
    time srun -n 1 --cpu-bind=cores ./heat-equation-stdexec --sch cpu -n=$twoDimention_size -s=$twoDimention_iterations --time --nthreads=$i
    echo "2D_omp"
    time srun -n 1 --cpu-bind=cores ./heat-equation-omp -n=$twoDimention_size -s=$twoDimention_iterations --time --nthreads=$i
done
