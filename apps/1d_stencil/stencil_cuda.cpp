#include "argparse/argparse.hpp"
#include "commons.hpp"

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

// parameters
struct args_params_t : public argparse::Args {
  bool& results = kwarg("results", "print generated results (default: false)")
                      .set_default(false);
  std::uint64_t& nx =
      kwarg("nx", "Local x dimension (of each partition)").set_default(10);
  std::uint64_t& nt = kwarg("nt", "Number of time steps").set_default(45);
  std::uint64_t& np = kwarg("np", "Number of partitions").set_default(10);
  bool& k = kwarg("k", "Heat transfer coefficient").set_default(0.5);
  double& dt = kwarg("dt", "Timestep unit (default: 1.0[s])").set_default(1.0);
  double& dx = kwarg("dx", "Local x dimension").set_default(1.0);
  bool& no_header =
      kwarg("no-header", "Do not print csv header row (default: false)")
          .set_default(false);
  bool& help = flag("h, help", "print help");
  bool& time = kwarg("t, time", "print time").set_default(true);
};

///////////////////////////////////////////////////////////////////////////////
// Command-line variables
bool header = true;        // print csv heading
constexpr double k = 0.5;  // heat transfer coefficient
constexpr double dt = 1.;  // time step
constexpr double dx = 1.;  // grid spacing

// Our operator
__device__ double heat(double left, double middle, double right) {
  return middle + (k * dt / (dx * dx)) * (left - 2 * middle + right);
}

__global__ void heat_equation(double* current, double* next, std::size_t size) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    std::size_t left = (i == 0) ? size - 1 : i - 1;
    std::size_t right = (i == size - 1) ? 0 : i + 1;
    next[i] = heat(current[left], current[i], current[right]);
  }
}

int benchmark(args_params_t const& args) {
  // Parameters (for simplicity, some are hardcoded)
  std::uint64_t np = args.np;  // Number of partitions.
  std::uint64_t nx = args.nx;  // Number of grid points.
  std::uint64_t nt = args.nt;  // Number of steps.
  std::size_t size = np * nx;

  double* h_current = nullptr;
  double* h_next = nullptr;

  // Measure execution time.
  Timer timer;

  // Memory allocation
  if (args.results) {
    h_current = new double[size];
    h_next = new double[size];
  }

  double* d_current;
  double* d_next;
  cudaMalloc(&d_current, size * sizeof(double));
  cudaMalloc(&d_next, size * sizeof(double));
  thrust::sequence(thrust::device, d_current, d_current + size, 0);
  thrust::sequence(thrust::device, d_next, d_next + size, 0);

  // CUDA kernel execution parameters
  const int threadsPerBlock = std::min(1024, (int)size);
  const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  // Actual time step loop
  for (std::size_t t = 0; t < nt; ++t) {
    heat_equation<<<blocks, threadsPerBlock>>>(d_current, d_next, size);
    std::swap(d_current, d_next);
  }
  cudaDeviceSynchronize();
  auto time = timer.stop();

  if (args.results) {
    // Copy result back to host
    cudaMemcpy(h_current, d_current, size * sizeof(double),
               cudaMemcpyDeviceToHost);

    // Print results
    for (std::size_t i = 0; i < np; ++i) {
      std::cout << "U[" << i << "] = {";
      for (std::size_t j = 0; j < nx; ++j) {
        std::cout << h_current[i * nx + j] << " ";
      }
      std::cout << "}\n";
    }
    // Cleanup
    delete[] h_current;
    delete[] h_next;
  }

  cudaFree(d_current);
  cudaFree(d_next);

  if (args.time) {
    std::cout << "Duration: " << time << " ms."
              << "\n";
  }

  return 0;
}

int main(int argc, char* argv[]) {
  // parse params
  args_params_t args = argparse::parse<args_params_t>(argc, argv);
  // see if help wanted
  if (args.help) {
    args.print();  // prints all variables
    return 0;
  }

  benchmark(args);

  return 0;
}
