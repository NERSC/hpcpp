# nvstdpar
NV-STDEXEC & C++26 powered GPU-accelerated scientific proxy apps & kernels

## Build Apps (Perlmutter)

```bash
git clone --recursive https://github.com/mhaseeb123/nvstdpar.git
cd nvstdpar ; mkdir build ; cd build
ml nvhpc/23.1 cmake 3.24
cmake .. -DUSE_MDSPAN=ON ; make -j
```

## Build Apps (Elsewhere)
Same as above but please edit `nvstdpar/CMakeLists.txt` before running `cmake` and add path to your `GCC/11.2.0+` compiler `/bin` to `--gcc-toolchain` as shown [here](https://github.com/mhaseeb123/nvstdpar/blob/main/CMakeLists.txt#L100).


## Run Apps

```bash
cd nvstdpar/build
srun -n 1 -N 1 -G 1 -A <acct> -C <gpu> ./apps/<appname>/<appname> [ARGS]
```

Use `--help` to see help with arguments.

## License
The Regents of the University of California (C) 2023
