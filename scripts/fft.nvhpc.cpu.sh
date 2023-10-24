#!/bin/bash -le

#
# Reminder: Revert any changes to nvstdpar/CMakeLists.txt and
# nvstdpar/apps/heat-equation/CMakeLists.txt that you did
# for GCC compiler script before running this.
#

#SBATCH -A nstaff
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=10:00:00
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

D=(536870912 1073741824)

# parallel runs
T=(256 128 64 32 16 8 4 2 1)

for d in "${D[@]}"; do
    for i in "${T[@]}"; do
        echo "stdexec:cpu for ${d}, threads=${i}"
        srun -n 1 --cpu-bind=cores ./fft-stdexec -N ${d} --time --sch=cpu --nthreads=${i}

        echo "stdpar:cpu for ${d}, threads=${i}"
        export OMP_NUM_THREADS=${i}
        srun -n 1 --cpu-bind=cores ./fft-stdpar -N ${d} --time --nthreads=${i}
    done
done

unset OMP_NUM_THREADS

for d in "${D[@]}"; do
    echo "serial for ${d}"
    srun -n 1 --cpu-bind=cores ./fft-serial -N ${d} --time
done
