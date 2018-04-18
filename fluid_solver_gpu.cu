#include "fluid_solver_gpu.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <array>

__host__ __device__ inline static size_t index(size_t const i, size_t const j, size_t const cols)
{
  return i * cols + j;
}

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
                                         T const*     initial_values,
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

      values[center_idx] = (initial_values[center_idx] + a * (prev_values[left_idx] + prev_values[right_idx] + prev_values[top_idx] + prev_values[bottom_idx])) / (1.0 + 4.0 * a);
    }
  }
}

template<typename T>
__global__ void set_boundary_continuous_kernel(T*           values,
                                               size_t const rows,
                                               size_t const cols)
{
  for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < rows - 1; i += blockDim.x * gridDim.x)
  {
    values[index(i, 0, cols)] = values[index(i, 1, cols)];
    values[index(i, cols - 1, cols)] = values[index(i, cols - 2, cols)];
  }

  for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < cols - 1; i += blockDim.x * gridDim.x)
  {
    values[index(0, i, cols)] = values[index(1, i, cols)];
    values[index(rows - 1, i, cols)] = values[index(rows - 2, i, cols)];
  }
}

fluid_solver_gpu::fluid_solver_gpu(size_t const rows, size_t const cols)
  : m_rows{ rows }
  , m_cols{ cols }
{
  cudaMalloc(&m_density_buffer, buffer_size());
  cudaMalloc(&m_source_buffer, buffer_size());
  cudaMalloc(&m_diffuse_initial_buffer, buffer_size());
  cudaMalloc(&m_diffuse_previous_buffer, buffer_size());
}

fluid_solver_gpu::~fluid_solver_gpu()
{
  cudaFree(m_density_buffer);
  cudaFree(m_source_buffer);
  cudaFree(m_diffuse_initial_buffer);
  cudaFree(m_diffuse_previous_buffer);
}

void fluid_solver_gpu::solve(grid<float>& density_grid,
                             grid<float> const& density_source_grid,
                             float const diffusion_rate,
                             float const dt)
{
  cudaMemcpy(m_density_buffer, density_grid.data(), buffer_size(), cudaMemcpyHostToDevice);

  add_sources(m_density_buffer, density_source_grid.data(), dt);

  diffuse(m_density_buffer, diffusion_rate, dt, 15);

  cudaMemcpy(density_grid.data(), m_density_buffer, buffer_size(), cudaMemcpyDeviceToHost);
}

void fluid_solver_gpu::add_sources(float* values_buffer,
                                   float const* source_data,
                                   float const dt)
{
  cudaMemcpy(m_source_buffer, source_data, buffer_size(), cudaMemcpyHostToDevice);

  unsigned int block_dim = 32;
  unsigned int grid_dim_x = static_cast<unsigned int>((m_cols + block_dim - 1) / block_dim);
  unsigned int grid_dim_y = static_cast<unsigned int>((m_rows + block_dim - 1) / block_dim);

  // Add the sources
  add_sources_kernel << <dim3(grid_dim_x, grid_dim_y), dim3(block_dim, block_dim) >> > (values_buffer, m_source_buffer, m_rows, m_cols, dt);
  cudaDeviceSynchronize();
}

void fluid_solver_gpu::diffuse(float* values_buffer, float const rate, float const dt, size_t iteration_count)
{
  unsigned int block_dim = 32;
  unsigned int grid_dim_x = static_cast<unsigned int>((m_cols + block_dim - 1) / block_dim);
  unsigned int grid_dim_y = static_cast<unsigned int>((m_rows + block_dim - 1) / block_dim);

  cudaMemcpy(m_diffuse_initial_buffer, values_buffer, buffer_size(), cudaMemcpyDeviceToDevice);
  for(size_t k = 0; k < iteration_count; k++)
  {
    cudaMemcpy(m_diffuse_previous_buffer, values_buffer, buffer_size(), cudaMemcpyDeviceToDevice);

    diffuse_iteration_kernel << <dim3(grid_dim_x, grid_dim_y), dim3(block_dim, block_dim) >> > (values_buffer,
                                                                                                m_diffuse_previous_buffer,
                                                                                                m_diffuse_initial_buffer,
                                                                                                m_rows, m_cols,
                                                                                                rate, dt);
    cudaDeviceSynchronize();

    set_boundary(values_buffer);
  }
}

void fluid_solver_gpu::set_boundary(float * values_buffer)
{
  unsigned int boundary_block_dim = 64;
  unsigned int boundary_grid_dim = static_cast<unsigned int>((std::max(m_cols, m_rows) + boundary_block_dim - 1) / boundary_block_dim);

  set_boundary_continuous_kernel << <boundary_grid_dim, boundary_block_dim >> > (values_buffer, m_rows, m_cols);

  cudaDeviceSynchronize();

  // Handle the corners
  std::array<float, 8> input_values;
  cudaMemcpy(&input_values[0], values_buffer + index(0, 1                  , m_cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[1], values_buffer + index(0, m_cols - 2         , m_cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[2], values_buffer + index(1, 0                  , m_cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[3], values_buffer + index(1, m_cols - 1         , m_cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[4], values_buffer + index(m_rows - 2, 0         , m_cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[5], values_buffer + index(m_rows - 2, m_cols - 1, m_cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[6], values_buffer + index(m_rows - 1, 1         , m_cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[7], values_buffer + index(m_rows - 1, m_cols - 2, m_cols), sizeof(float), cudaMemcpyDeviceToHost);

  std::array<float, 4> corner_values;
  corner_values[0] = 0.5f * (input_values[0] + input_values[2]);
  corner_values[1] = 0.5f * (input_values[1] + input_values[3]);
  corner_values[2] = 0.5f * (input_values[4] + input_values[6]);
  corner_values[3] = 0.5f * (input_values[5] + input_values[7]);

  cudaMemcpy(values_buffer + index(0, 0                  , m_cols), &corner_values[0], sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(values_buffer + index(0, m_cols - 1         , m_cols), &corner_values[1], sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(values_buffer + index(m_rows - 1, 0         , m_cols), &corner_values[2], sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(values_buffer + index(m_rows - 1, m_cols - 1, m_cols), &corner_values[3], sizeof(float), cudaMemcpyHostToDevice);
}
