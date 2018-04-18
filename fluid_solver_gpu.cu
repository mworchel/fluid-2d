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

template<typename T>
__global__ void set_boundary_opposite_horizontal_kernel(T*           values,
                                                        size_t const rows,
                                                        size_t const cols)
{
  for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < rows - 1; i += blockDim.x * gridDim.x)
  {
    values[index(i, 0, cols)] = -values[index(i, 1, cols)];
    values[index(i, cols - 1, cols)] = -values[index(i, cols - 2, cols)];
  }

  for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < cols - 1; i += blockDim.x * gridDim.x)
  {
    values[index(0, i, cols)] = values[index(1, i, cols)];
    values[index(rows - 1, i, cols)] = values[index(rows - 2, i, cols)];
  }
}

template<typename T>
__global__ void set_boundary_opposite_vertical_kernel(T*           values,
                                                      size_t const rows,
                                                      size_t const cols)
{
  for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < cols - 1; i += blockDim.x * gridDim.x)
  {
    values[index(0, i, cols)] = -values[index(1, i, cols)];
    values[index(rows - 1, i, cols)] = -values[index(rows - 2, i, cols)];
  }

  for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < rows - 1; i += blockDim.x * gridDim.x)
  {
    values[index(i, 0, cols)] = values[index(i, 1, cols)];
    values[index(i, cols - 1, cols)] = values[index(i, cols - 2, cols)];
  }
}

template<typename T>
__global__ void add_sources_kernel(T*           values,
                                   T const*     source_values,
                                   size_t const rows,
                                   size_t const cols,
                                   float const  dt)
{
  for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y)
  {
    for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x)
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
  for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y)
  {
    for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x)
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
__global__ void advect_kernel(T*           values,
                              T const*     initial_values,
                              T const*     horizontal_velocities,
                              T const*     vertical_velocities,
                              size_t const rows,
                              size_t const cols,
                              float const  dt0)
{


  for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y)
  {
    for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x)
    {
      float x = j - dt0 * horizontal_velocities[index(i, j, cols)];
      float y = i - dt0 * vertical_velocities[index(i, j, cols)];
      x = max(1.5f, min(cols - 1.5f, x));
      y = max(1.5f, min(rows - 1.5f, y));

      size_t j0 = static_cast<size_t>(x);
      size_t i0 = static_cast<size_t>(y);
      size_t j1 = j0 + 1;
      size_t i1 = i0 + 1;
      float s0 = x - j0;
      float s1 = 1 - s0;
      float s2 = y - i0;
      float s3 = 1 - s2;

      values[index(i, j, cols)] = s3 * (s1 * initial_values[index(i0, j0, cols)] + s0 * initial_values[index(i0, j1, cols)]) +
        s2 * (s1 * initial_values[index(i1, j0, cols)] + s0 * initial_values[index(i1, j1, cols)]);
    }
  }
}

template<typename T>
__global__ void calculate_divergence_kernel(T* divergence,
                                            T const* horizontal_velocities,
                                            T const* vertical_velocities,
                                            size_t rows,
                                            size_t cols,
                                            float const h)
{
  for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y)
  {
    for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x)
    {
      divergence[index(i, j, cols)] = -0.5f * h * (horizontal_velocities[index(i, j + 1, cols)] - horizontal_velocities[index(i, j - 1, cols)] +
                                                   vertical_velocities[index(i + 1, j, cols)] - vertical_velocities[index(i - 1, j, cols)]);
    }
  }
}

template<typename T>
__global__ void p_iteration_kernel(T* p,
                                   T const* p_previous,
                                   T const* divergence,
                                   size_t rows,
                                   size_t cols)
{
  for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y)
  {
    for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x)
    {
      p[index(i, j, cols)] = (divergence[index(i, j, cols)] + p_previous[index(i, j + 1, cols)] + p_previous[index(i, j - 1, cols)] +
                              p_previous[index(i + 1, j, cols)] + p_previous[index(i - 1, j, cols)]) / 4.0f;
    }
  }
}

template<typename T>
__global__ void remove_p_kernel(T* horizontal_velocities,
                                T* vertical_velocities,
                                T const* p,
                                size_t rows,
                                size_t cols,
                                float const h)
{
  for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y)
  {
    for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x)
    {
      horizontal_velocities[index(i, j, cols)] -= 0.5f * (p[index(i, j + 1, cols)] - p[index(i, j - 1, cols)]) / h;
      vertical_velocities[index(i, j, cols)] -= 0.5f * (p[index(i + 1, j, cols)] - p[index(i - 1, j, cols)]) / h;
    }
  }
}


fluid_solver_gpu::fluid_solver_gpu(size_t const rows, size_t const cols)
  : m_rows{ rows }
  , m_cols{ cols }
{
  cudaMalloc(&m_density_buffer, buffer_size());
  cudaMalloc(&m_horizontal_velocity_buffer, buffer_size());
  cudaMalloc(&m_vertical_velocity_buffer, buffer_size());
  cudaMalloc(&m_source_buffer, buffer_size());
  cudaMalloc(&m_temp_buffer_1, buffer_size());
  cudaMalloc(&m_temp_buffer_2, buffer_size());
  cudaMalloc(&m_temp_buffer_3, buffer_size());
}

fluid_solver_gpu::~fluid_solver_gpu()
{
  cudaFree(m_density_buffer);
  cudaFree(m_horizontal_velocity_buffer);
  cudaFree(m_vertical_velocity_buffer);
  cudaFree(m_source_buffer);
  cudaFree(m_temp_buffer_1);
  cudaFree(m_temp_buffer_2);
  cudaFree(m_temp_buffer_3);
}

void fluid_solver_gpu::solve(grid<float>& density_grid,
                             grid<float> const& density_source_grid,
                             float const diffusion_rate,
                             grid<float>& horizontal_velocity_grid,
                             grid<float>& vertical_velocity_grid,
                             grid<float> const& horizontal_velocity_source_grid,
                             grid<float> const& vertical_velocity_source_grid,
                             float const viscosity,
                             float const dt)
{
  // Upload density and velocities to gpu
  cudaMemcpy(m_density_buffer, density_grid.data(), buffer_size(), cudaMemcpyHostToDevice);
  cudaMemcpy(m_horizontal_velocity_buffer, horizontal_velocity_grid.data(), buffer_size(), cudaMemcpyHostToDevice);
  cudaMemcpy(m_vertical_velocity_buffer, vertical_velocity_grid.data(), buffer_size(), cudaMemcpyHostToDevice);

  // Solve density related terms
  add_sources(m_density_buffer, density_source_grid.data(), dt);
  diffuse(m_density_buffer, &fluid_solver_gpu::set_boundary_continuous, diffusion_rate, dt, 15);
  advect(m_density_buffer, m_horizontal_velocity_buffer, m_vertical_velocity_buffer, &fluid_solver_gpu::set_boundary_continuous, dt);

  // Solve velocity related terms
  add_sources(m_horizontal_velocity_buffer, horizontal_velocity_source_grid.data(), dt);
  add_sources(m_vertical_velocity_buffer, vertical_velocity_source_grid.data(), dt);
  diffuse(m_horizontal_velocity_buffer, &fluid_solver_gpu::set_boundary_opposite_horizontal, viscosity, dt, 15);
  diffuse(m_vertical_velocity_buffer, &fluid_solver_gpu::set_boundary_opposite_vertical, viscosity, dt, 15);
  project(m_horizontal_velocity_buffer, m_vertical_velocity_buffer);
  cudaMemcpy(m_temp_buffer_2, m_horizontal_velocity_buffer, buffer_size(), cudaMemcpyDeviceToDevice);
  cudaMemcpy(m_temp_buffer_3, m_vertical_velocity_buffer, buffer_size(), cudaMemcpyDeviceToDevice);
  advect(m_horizontal_velocity_buffer, m_temp_buffer_2, m_temp_buffer_3, &fluid_solver_gpu::set_boundary_opposite_horizontal, dt);
  advect(m_vertical_velocity_buffer, m_temp_buffer_2, m_temp_buffer_3, &fluid_solver_gpu::set_boundary_opposite_vertical, dt);
  project(m_horizontal_velocity_buffer, m_vertical_velocity_buffer);

  // Download density and velocities to the cpu
  cudaMemcpy(density_grid.data(), m_density_buffer, buffer_size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(horizontal_velocity_grid.data(), m_horizontal_velocity_buffer, buffer_size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(vertical_velocity_grid.data(), m_vertical_velocity_buffer, buffer_size(), cudaMemcpyDeviceToHost);
}

void fluid_solver_gpu::set_corners(float * values_buffer, size_t rows, size_t cols)
{
  std::array<float, 8> input_values;
  cudaMemcpy(&input_values[0], values_buffer + index(0, 1, cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[1], values_buffer + index(0, cols - 2, cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[2], values_buffer + index(1, 0, cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[3], values_buffer + index(1, cols - 1, cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[4], values_buffer + index(rows - 2, 0, cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[5], values_buffer + index(rows - 2, cols - 1, cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[6], values_buffer + index(rows - 1, 1, cols), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_values[7], values_buffer + index(rows - 1, cols - 2, cols), sizeof(float), cudaMemcpyDeviceToHost);

  std::array<float, 4> corner_values;
  corner_values[0] = 0.5f * (input_values[0] + input_values[2]);
  corner_values[1] = 0.5f * (input_values[1] + input_values[3]);
  corner_values[2] = 0.5f * (input_values[4] + input_values[6]);
  corner_values[3] = 0.5f * (input_values[5] + input_values[7]);

  cudaMemcpy(values_buffer + index(0, 0, cols), &corner_values[0], sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(values_buffer + index(0, cols - 1, cols), &corner_values[1], sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(values_buffer + index(rows - 1, 0, cols), &corner_values[2], sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(values_buffer + index(rows - 1, cols - 1, cols), &corner_values[3], sizeof(float), cudaMemcpyHostToDevice);
}

void fluid_solver_gpu::set_boundary_continuous(float* values_buffer, size_t rows, size_t cols)
{
  unsigned int boundary_block_dim = 64;
  unsigned int boundary_grid_dim = static_cast<unsigned int>((std::max(cols, rows) + boundary_block_dim - 1) / boundary_block_dim);

  set_boundary_continuous_kernel << <boundary_grid_dim, boundary_block_dim >> > (values_buffer, rows, cols);
  cudaDeviceSynchronize();

  //set_corners(values_buffer, rows, cols);
}

void fluid_solver_gpu::set_boundary_opposite_horizontal(float * values_buffer, size_t rows, size_t cols)
{
  unsigned int boundary_block_dim = 64;
  unsigned int boundary_grid_dim = static_cast<unsigned int>((std::max(cols, rows) + boundary_block_dim - 1) / boundary_block_dim);

  set_boundary_opposite_horizontal_kernel << <boundary_grid_dim, boundary_block_dim >> > (values_buffer, rows, cols);
  cudaDeviceSynchronize();

  //set_corners(values_buffer, rows, cols);
}

void fluid_solver_gpu::set_boundary_opposite_vertical(float * values_buffer, size_t rows, size_t cols)
{
  unsigned int boundary_block_dim = 64;
  unsigned int boundary_grid_dim = static_cast<unsigned int>((std::max(cols, rows) + boundary_block_dim - 1) / boundary_block_dim);

  set_boundary_opposite_vertical_kernel << <boundary_grid_dim, boundary_block_dim >> > (values_buffer, rows, cols);
  cudaDeviceSynchronize();

  //set_corners(values_buffer, rows, cols);
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

void fluid_solver_gpu::diffuse(float* values_buffer,
                               std::function<void(float*, size_t, size_t)> set_boundary,
                               float const rate,
                               float const dt,
                               size_t iteration_count)
{
  unsigned int block_dim = 32;
  unsigned int grid_dim_x = static_cast<unsigned int>((m_cols + block_dim - 1) / block_dim);
  unsigned int grid_dim_y = static_cast<unsigned int>((m_rows + block_dim - 1) / block_dim);

  // First temporary buffer contains the initial values
  // Second temporary buffer contains the values of the previous iteration
  cudaMemcpy(m_temp_buffer_1, values_buffer, buffer_size(), cudaMemcpyDeviceToDevice);
  for(size_t k = 0; k < iteration_count; k++)
  {
    cudaMemcpy(m_temp_buffer_2, values_buffer, buffer_size(), cudaMemcpyDeviceToDevice);

    diffuse_iteration_kernel << <dim3(grid_dim_x, grid_dim_y), dim3(block_dim, block_dim) >> > (values_buffer,
                                                                                                m_temp_buffer_2,
                                                                                                m_temp_buffer_1,
                                                                                                m_rows, m_cols,
                                                                                                rate, dt);
    cudaDeviceSynchronize();

    set_boundary(values_buffer, m_rows, m_cols);
  }
}

void fluid_solver_gpu::advect(float * values_buffer,
                              float * horizontal_velocity_buffer,
                              float * vertical_velocity_buffer,
                              std::function<void(float*, size_t, size_t)> set_boundary,
                              float const dt)
{
  // First temp buffer contains the initial values
  cudaMemcpy(m_temp_buffer_1, values_buffer, buffer_size(), cudaMemcpyDeviceToDevice);

  float dt0 = sqrt(m_rows * m_cols) * dt;

  unsigned int block_dim = 32;
  unsigned int grid_dim_x = static_cast<unsigned int>((m_cols + block_dim - 1) / block_dim);
  unsigned int grid_dim_y = static_cast<unsigned int>((m_rows + block_dim - 1) / block_dim);

  advect_kernel << <dim3(grid_dim_x, grid_dim_y), dim3(block_dim, block_dim) >> > (values_buffer, m_temp_buffer_1,
                                                                                   horizontal_velocity_buffer, vertical_velocity_buffer,
                                                                                   m_rows, m_cols,
                                                                                   dt0);
  cudaDeviceSynchronize();

  set_boundary(values_buffer, m_rows, m_cols);
}

void fluid_solver_gpu::project(float * horizontal_velocity_buffer, float * vertical_velocity_buffer)
{
  float* divergence_buffer = m_temp_buffer_1;
  float* p_buffer = m_temp_buffer_2;
  cudaMemset(divergence_buffer, 0, buffer_size());
  cudaMemset(p_buffer, 0, buffer_size());

  float h = 1.0f / sqrtf(m_rows * m_cols);

  unsigned int block_dim = 32;
  unsigned int grid_dim_x = static_cast<unsigned int>((m_cols + block_dim - 1) / block_dim);
  unsigned int grid_dim_y = static_cast<unsigned int>((m_rows + block_dim - 1) / block_dim);

  calculate_divergence_kernel << <dim3(grid_dim_x, grid_dim_y), dim3(block_dim, block_dim) >> > (divergence_buffer,
                                                                                                 horizontal_velocity_buffer,
                                                                                                 vertical_velocity_buffer,
                                                                                                 m_rows,
                                                                                                 m_cols,
                                                                                                 h);
  cudaDeviceSynchronize();
  set_boundary_continuous(divergence_buffer, m_rows, m_cols);

  float* p_previous_buffer = m_temp_buffer_3;
  for(size_t k = 0; k < 20; ++k)
  {
    cudaMemcpy(p_previous_buffer, p_buffer, buffer_size(), cudaMemcpyDeviceToDevice);

    p_iteration_kernel << <dim3(grid_dim_x, grid_dim_y), dim3(block_dim, block_dim) >> > (p_buffer,
                                                                                          p_previous_buffer,
                                                                                          divergence_buffer,
                                                                                          m_rows,
                                                                                          m_cols);
    cudaDeviceSynchronize();

    set_boundary_continuous(p_buffer, m_rows, m_cols);
  }

  remove_p_kernel << <dim3(grid_dim_x, grid_dim_y), dim3(block_dim, block_dim) >> > (horizontal_velocity_buffer,
                                                                                     vertical_velocity_buffer,
                                                                                     p_buffer,
                                                                                     m_rows,
                                                                                     m_cols,
                                                                                     h);
  cudaDeviceSynchronize();

  set_boundary_opposite_horizontal(horizontal_velocity_buffer, m_rows, m_cols);
  set_boundary_opposite_vertical(vertical_velocity_buffer, m_rows, m_cols);
}
