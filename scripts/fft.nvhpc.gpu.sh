#!/bin/bash -le

#
# Reminder: Revert any changes to nvstdpar/CMakeLists.txt and
# nvstdpar/apps/heat-equation/CMakeLists.txt that you did
# for GCC compiler script before running this.
#

#SBATCH -A nstaff_g
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --gpu-bind=none
#SBATCH -o fft-gpu.o%j
#SBATCH -e fft-gpu.e%j
#SBATCH -J FFT-GPU

set +x

mkdir -p ${HOME}/repos/nvstdpar/build-fft-gpu
cd ${HOME}/repos/nvstdpar/build-fft-gpu
rm -rf ./*

ml unload cudatoolkit
ml use /global/cfs/cdirs/m1759/wwei/nvhpc_23_7/modulefiles
ml nvhpc/23.7
# need this for GLIBC
ml gcc/12.2.0
ml cmake/3.24

cmake .. -DSTDPAR=gpu -DOMP=gpu -DCMAKE_CXX_COMPILER=$(which nvc++)

make -j fft-stdexec fft-stdpar

cd ${HOME}/repos/nvstdpar/build-fft-gpu/apps/fft

D=(536870912 1073741824)

for d in "${D[@]}"; do
    echo "stdexec:gpu for ${d}"
    srun -n 1 ./fft-stdexec -N ${d} --time --sch=gpu

    echo "stdpar:gpu for ${d}"
    srun -n 1  ./fft-stdpar -N ${d} --time 2>&1
done

for d in "${D[@]}"; do
    echo "stdexec:multi_gpu for ${d}"
    srun -n 1 ./fft-stdexec -N ${d} --time --sch=multigpu 2>&1
done

