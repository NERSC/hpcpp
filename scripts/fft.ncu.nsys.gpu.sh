#!/bin/bash -le

#
# Reminder: Revert any changes to nvstdpar/CMakeLists.txt and
# nvstdpar/apps/heat-equation/CMakeLists.txt that you did
# for GCC compiler script before running this.
#

#SBATCH -A nstaff_g
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --gpu-bind=none
#SBATCH -o ncu-nsys-fft-gpu.o%j
#SBATCH -e ncu-nsys-fft-gpu.e%j
#SBATCH -J FFT-GPU-PERF

set +x

# config setting
BUILD_HOME=${HOME}/repos/nvstdpar/build-fft-gpu-nsight

# build stuff
mkdir -p ${BUILD_HOME}
cd ${BUILD_HOME}
rm -rf ./*

ml unload cudatoolkit
ml use /global/cfs/cdirs/m1759/wwei/nvhpc_23_7/modulefiles
ml nvhpc/23.7

cmake .. -DSTDPAR=gpu -DOMP=gpu -DCMAKE_CXX_COMPILER=$(which nvc++)
make -j fft-stdexec fft-stdpar

# always run NCU and Nsys from $SCRATCH to avoid errors on Perlmutter
mkdir -p ${SCRATCH}/fft-gpu-nsight
cd ${SCRATCH}/fft-gpu-nsight
rm -rf ./*


# pause dcgmi
srun --ntasks-per-node 1 dcgmi profile --pause

# Problem size (increasing this beyond 4024000 may take long time for multigpu runs)
SIZE=4024000

# Run Nsys

# stdexec-single-gpu
srun nsys profile --force-overwrite true -o fft-gpu-stdexec.nsys --stats=true ${BUILD_HOME}/apps/fft/fft-stdexec --sch=gpu -N ${SIZE} |& tee nsys-fft-stdexec-gpu.log

# stdpar-gpu (not sure if more than one)
srun nsys profile --force-overwrite true -o fft-gpu-stdpar.nsys --stats=true ${BUILD_HOME}/apps/fft/fft-stdpar -N ${SIZE} |& tee nsys-fft-stdpar-gpu.log

# stdexec-multigpu
srun nsys profile --force-overwrite true -o fft-multigpu-stdexec.nsys --stats=true ${BUILD_HOME}/apps/fft/fft-stdexec --sch=multigpu -N ${SIZE} |& tee nsys-fft-multigpu-stdexec.log


# Run NCU (set full)

# stdexec-single-gpu (full)
srun ncu -f -o fft-gpu-stdexec.ncu  --target-processes all --print-summary per-gpu --replay-mode application  --set full ${BUILD_HOME}/apps/fft/fft-stdexec -N ${SIZE} --sch=gpu |& tee ncu-fft-stdexec-gpu.log

# stdpar-gpu (full)
srun ncu -f -o fft-gpu-stdpar.ncu  --target-processes all --print-summary per-gpu --replay-mode application  --set full ${BUILD_HOME}/apps/fft/fft-stdpar -N ${SIZE} |& tee ncu-fft-stdpar-gpu.log

# stdexec-multigpu (full)
srun ncu -f -o fft-multigpu-stdexec.log  --target-processes all --print-summary per-gpu --replay-mode application  --set full ${BUILD_HOME}/apps/fft/fft-stdexec -N ${SIZE} --sch=multigpu |& tee ncu-fft-multigpu-stdexec.log


# Run NCU (set roofline only)

# stdexec-single-gpu (roofline)
ncu -f -o fft-gpu-stdexec-roofline.ncu  --target-processes all --print-summary per-gpu --replay-mode application  --set roofline ${BUILD_HOME}/apps/fft/fft-stdexec -N ${SIZE} --sch=gpu |& tee ncu-fft-stdexec-gpu-roofline.log

# stdpar-gpu (roofline)
srun ncu -f -o fft-gpu-stdpar-roofline.ncu  --target-processes all --print-summary per-gpu --replay-mode application  --set full ${BUILD_HOME}/apps/fft/fft-stdpar -N ${SIZE} |& tee ncu-fft-stdpar-gpu-roofline.log

# stdexec-multigpu (roofline)
srun ncu -f -o fft-multigpu-stdexec-roofline.log  --target-processes all --print-summary per-gpu --replay-mode application  --set roofline ${BUILD_HOME}/apps/fft/fft-stdexec -N ${SIZE} --sch=multigpu |& tee ncu-fft-multigpu-stdexec-roofline.log

# resume dcgmi
srun --ntasks-per-node 1 dcgmi profile --resume
