ml nvhpc/23.9
ml cmake/3.24

BASE=${BASE:-$(pwd)/../..}
cd ${BASE}

git clone --recursive https://github.com/NERSC/hpcpp.git
cd hpcpp

export GCCLOCALRC=scripts/localrc_muller

cd /global/homes/w/wwei/src/hpcpp/
mkdir -p build_gpu && cd build_gpu
cmake -DCMAKE_BUILD_TYPE=Debug -DSTDPAR=gpu -DOMP=gpu ..
make -j10
echo "build succeed"
./apps/1d-stencil/1d-stdexec --sch gpu --size 10 --nt 10
echo "test succeed"
