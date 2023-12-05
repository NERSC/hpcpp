# hpcpp

Standard C++26 High-Performance Computing (HPC) applications that run on CPUs and GPUs. 

## Build

```bash
git clone https://github.com/NERSC/hpcpp.git
cd hpcpp; mkdir build ; cd build
ml nvhpc/23.7 cmake 3.24

# enable GPU support by setting -DSTDPAR=gpu (default)
cmake .. -DSTDPAR=<gpu/multicore> -DOMP=<gpu/multicore; 
make -j 10
```

**Note**: Make sure your `localrc` file (located at `/path/to/nvhpc/bin`) is properly configured to `GCC >= 11.2.0` paths.

### NERSC Users

You can also use the pre-configured `localrc` file included in this repo. To use it, run:

```bash
export GCCLOCALRC=/path/to/hpcpp/scripts/pm-localrc/localrc
```

**Note**: Please uncomment the following line in `apps/fft/CMakeLists.txt` if using `nvc++` version < 23.7?

```bash
  # uncomment only if using nvc++ earlier than 23.7 to find libcublas
  # target_link_directories(${exec_name} PRIVATE /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/lib64)
```

## Run Apps

```bash
cd hpcpp/build
srun -n 1 -N 1 -G <> -A <acct> -t 30 -C <cpu/gpu> ./apps/<appname>/<appname> [ARGS]
```

Use `--help` to see help with arguments.

## Contributors

(in alphabetical order of last name)
- [Muhammad Haseeb](https://nersc.gov/muhammad-haseeb)
- [Weile Wei](https://nersc.gov/weile-wei)
- [Chuanqiu He](https://github.com/hcq9102)

## License
Copyright (C) The Regents of the University of California, 2023 (See [LICENSE](LICENSE) for details).
