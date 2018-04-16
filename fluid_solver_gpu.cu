#include "fluid_solver_gpu.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template<typename T>
__global__ void add_sources_kernel(T*           values,
                                   T const*     source_values,
                                   size_t const rows,
                                   size_t const cols,
                                   float const  dt)
{
  for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < rows - 1; i += blockDim.x * gridDim.x)
  {
    for(size_t j = blockIdx.y * blockDim.y + threadIdx.y + 1; j < cols - 1; j += blockDim.y * gridDim.y)
    {
      values[i * cols + j] = values[i * cols + j] + dt * source_values[i * cols + j];
    }
  }
}

template<typename T>
__global__ void diffuse_iteration_kernel(T*           values,
                                         T const*     prev_values,
                                         size_t const rows,
                                         size_t const cols,
                                         T const      diffusion_rate,
                                         float const  dt)
{
  for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < rows - 1; i += blockDim.x * gridDim.x)
  {
    for(size_t j = blockIdx.y * blockDim.y + threadIdx.y + 1; j < cols - 1; j += blockDim.y * gridDim.y)
    {
      size_t center_idx = i * cols + j;
      size_t left_idx = i * cols + (j - 1);
      size_t right_idx = i * cols + (j + 1);
      size_t top_idx = (i - 1) * cols + j;
      size_t bottom_idx = (i + 1) * cols + j;
      T a = dt * static_cast<T>(rows * cols) * diffusion_rate;

      values[center_idx] = (prev_values[center_idx] + a * (prev_values[left_idx] + prev_values[right_idx] + prev_values[top_idx] + prev_values[bottom_idx])) / (1.0 + 4.0 * a);
    }
  }
}

fluid_solver_gpu::fluid_solver_gpu(size_t const rows, size_t const cols)
  : m_rows{ rows }
  , m_cols{ cols }
{
  cudaMalloc(&m_density_buffer, sizeof(float) * m_rows * m_cols);
  cudaMalloc(&m_density_source_buffer, sizeof(float) * m_rows * m_cols);
  cudaMalloc(&m_density_previous_buffer, sizeof(float) * m_rows * m_cols);
}

fluid_solver_gpu::~fluid_solver_gpu()
{
  cudaFree(m_density_buffer);
  cudaFree(m_density_source_buffer);
  cudaFree(m_density_previous_buffer);
}

void fluid_solver_gpu::solve(grid<float>& density_grid, 
                             grid<float> const& density_source_grid, 
                             float const diffusion_rate,
                             float const dt)
{
  cudaMemcpy(m_density_buffer, density_grid.data(), sizeof(float) * m_rows * m_cols, cudaMemcpyHostToDevice);
  cudaMemcpy(m_density_source_buffer, density_source_grid.data(), sizeof(float) * m_rows * m_cols, cudaMemcpyHostToDevice);

  unsigned int block_dim = 32;
  unsigned int grid_dim_x = static_cast<unsigned int>((m_cols + block_dim - 1) / block_dim);
  unsigned int grid_dim_y = static_cast<unsigned int>((m_rows + block_dim - 1) / block_dim);

  // Add the sources
  add_sources_kernel << <dim3(grid_dim_x, grid_dim_y), dim3(block_dim, block_dim) >> > (m_density_buffer, m_density_source_buffer, m_rows, m_cols, dt);
  cudaDeviceSynchronize();

  // Perform diffusion
  for(size_t k = 0; k < 15; k++)
  {
    cudaMemcpy(m_density_previous_buffer, m_density_buffer, sizeof(float) * m_rows * m_cols, cudaMemcpyDeviceToDevice);

    diffuse_iteration_kernel << <dim3(grid_dim_x, grid_dim_y), dim3(block_dim, block_dim) >> > (m_density_buffer, m_density_previous_buffer, m_rows, m_cols, diffusion_rate, dt);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(density_grid.data(), m_density_buffer, sizeof(float) * m_rows * m_cols, cudaMemcpyDeviceToHost);
}