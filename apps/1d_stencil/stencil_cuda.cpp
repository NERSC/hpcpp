#include "argparse/argparse.hpp"
#include "commons.hpp"

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

// parameters
struct args_params_t : public argparse::Args {
  bool& results = kwarg("results", "print generated results (default: false)")
                      .set_default(false);
  std::uint64_t& nt = kwarg("nt", "Number of time steps").set_default(45);
  std::uint64_t& size = kwarg("size", "Number of elements").set_default(10);
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
using Real_t = double;
bool header = true;        // print csv heading
Real_t k = 0.5;  // heat transfer coefficient
Real_t dt = 1.;  // time step
Real_t dx = 1.;  // grid spacing

// Our operator
__device__ Real_t heat(const Real_t left, const Real_t middle, const Real_t right, 
  const Real_t k, const Real_t dt, const Real_t dx) {
  return middle + (k * dt / (dx * dx)) * (left - 2 * middle + right);
}

__global__ void heat_equation(Real_t* current, Real_t* next, std::size_t size, 
  const Real_t k, const Real_t dt, const Real_t dx) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    std::size_t left = (i == 0) ? size - 1 : i - 1;
    std::size_t right = (i == size - 1) ? 0 : i + 1;
    next[i] = heat(current[left], current[i], current[right], k, dt, dx);
  }
}

int benchmark(args_params_t const& args) {
  std::uint64_t size = args.size;  // Number of elements.
  std::uint64_t nt = args.nt;  // Number of steps.

  Real_t* h_current = nullptr;
  Real_t* h_next = nullptr;

  // Measure execution time.
  Timer timer;

  // Memory allocation
  if (args.results) {
    h_current = new Real_t[size];
    h_next = new Real_t[size];
  }

  Real_t* d_current;
  Real_t* d_next;
  cudaMalloc(&d_current, size * sizeof(Real_t));
  cudaMalloc(&d_next, size * sizeof(Real_t));
  thrust::sequence(thrust::device, d_current, d_current + size, 0);
  thrust::sequence(thrust::device, d_next, d_next + size, 0);

  // CUDA kernel execution parameters
  const int threadsPerBlock = std::min(1024, (int)size);
  const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  // Actual time step loop
  for (std::size_t t = 0; t < nt; ++t) {
    heat_equation<<<blocks, threadsPerBlock>>>(d_current, d_next, size, k, dt, dx);
    std::swap(d_current, d_next);
  }
  cudaDeviceSynchronize();
  auto time = timer.stop();

  if (args.results) {
    // Copy result back to host
    cudaMemcpy(h_current, d_current, size * sizeof(Real_t),
               cudaMemcpyDeviceToHost);

    // Print results
    for (std::size_t i = 0; i != size; ++i) {
      std::cout << h_current[i] << " ";
    }
    std::cout << "\n";
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
