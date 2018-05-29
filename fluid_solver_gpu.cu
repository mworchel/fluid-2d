#include "fluid_solver_gpu.cuh"
#include "kernel_launcher.hpp"
#include "utilities.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <array>

template<typename T>
__global__ void set_boundary_continuous_kernel(element_accessor<T> values,
                                               size_t const        rows,
                                               size_t const        cols) {
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < rows - 1; i += blockDim.x * gridDim.x) {
        values.at(0, i) = values.at(1, i);
        values.at(cols - 1, i) = values.at(cols - 2, i);
    }

    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < cols - 1; i += blockDim.x * gridDim.x) {
        values.at(i, 0) = values.at(i, 1);
        values.at(i, rows - 1) = values.at(i, rows - 2);
    }
}

template<typename T>
__global__ void set_boundary_opposite_horizontal_kernel(element_accessor<T> values,
                                                        size_t const        rows,
                                                        size_t const        cols) {
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < rows - 1; i += blockDim.x * gridDim.x) {
        values.at(0, i) = -values.at(1, i);
        values.at(cols - 1, i) = -values.at(cols - 2, i);
    }

    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < cols - 1; i += blockDim.x * gridDim.x) {
        values.at(i, 0) = values.at(i, 1);
        values.at(i, rows - 1) = values.at(i, rows - 2);
    }
}

template<typename T>
__global__ void set_boundary_opposite_vertical_kernel(element_accessor<T> values,
                                                      size_t const        rows,
                                                      size_t const        cols) {
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < cols - 1; i += blockDim.x * gridDim.x) {
        values.at(i, 0) = -values.at(i, 1);
        values.at(i, rows - 1) = -values.at(i, rows - 2);
    }

    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; i < rows - 1; i += blockDim.x * gridDim.x) {
        values.at(0, i) = values.at(1, i);
        values.at(cols - 1, i) = values.at(cols - 2, i);
    }
}

template<typename T>
__global__ void add_sources_kernel(element_accessor<T>       values,
                                   element_accessor<T> const source_values,
                                   size_t const              rows,
                                   size_t const              cols,
                                   float const               dt) {
    for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y) {
        for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x) {
            values.at(j, i) = values.at(j, i) + dt * source_values.at(j, i);
        }
    }
}

template<typename T>
__global__ void diffuse_iteration_kernel(element_accessor<T>       values,
                                         element_accessor<T> const prev_values,
                                         element_accessor<T> const initial_values,
                                         size_t const              rows,
                                         size_t const              cols,
                                         T const                   diffusion_rate,
                                         float const               dt) {
    for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y) {
        for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x) {
            T a = dt * static_cast<T>(rows * cols) * diffusion_rate;

            values.at(j, i) = (initial_values.at(j, i) + a * (prev_values.at(j - 1, i) + prev_values.at(j + 1, i) +
                                                              prev_values.at(j, i - 1) + prev_values.at(j, i + 1))) / (1.0 + 4.0 * a);
        }
    }
}

template<typename T>
__global__ void smooth_kernel(element_accessor<T>       values,
                              element_accessor<T> const initial_values,
                              size_t const              rows,
                              size_t const              cols) {
    for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y) {
        for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x) {
            values.at(j, i) = 0.2f * (initial_values.at(j, i) + initial_values.at(j - 1, i) + initial_values.at(j + 1, i) + initial_values.at(j, i - 1) + initial_values.at(j, i + 1));
        }
    }
}

template<typename T>
__global__ void advect_kernel(element_accessor<T>       values,
                              element_accessor<T> const initial_values,
                              element_accessor<T> const horizontal_velocities,
                              element_accessor<T> const vertical_velocities,
                              size_t const              rows,
                              size_t const              cols,
                              float const               dt0) {


    for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y) {
        for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x) {
            float x = j - dt0 * horizontal_velocities.at(j, i);
            float y = i - dt0 * vertical_velocities.at(j, i);
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

            values.at(j, i) = s3 * (s1 * initial_values.at(j0, i0) + s0 * initial_values.at(j1, i0)) +
                s2 * (s1 * initial_values.at(j0, i1) + s0 * initial_values.at(j1, i1));
        }
    }
}

template<typename T>
__global__ void advect_trace_kernel(element_accessor<T>       values,
                                    element_accessor<T> const initial_values,
                                    element_accessor<T> const horizontal_velocities,
                                    element_accessor<T> const vertical_velocities,
                                    size_t const              rows,
                                    size_t const              cols,
                                    float const               dt0) {
    for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y) {
        for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x) {
            float x = j + dt0 * horizontal_velocities.at(j, i);
            float y = i + dt0 * vertical_velocities.at(j, i);

            if(x < 0.5f || x > cols - 1.5f || y < 0.5f || y > rows - 1.5f) continue;

            size_t j0 = static_cast<size_t>(x);
            size_t i0 = static_cast<size_t>(y);
            size_t j1 = j0 + 1;
            size_t i1 = i0 + 1;

            float s0 = x - j0;
            float s1 = 1 - s0;
            float s2 = y - i0;
            float s3 = 1 - s2;

            atomicAdd(&values.at(j0, i0), s1 * s3 * initial_values.at(j, i));
            atomicAdd(&values.at(j0, i1), s1 * s2 * initial_values.at(j, i));
            atomicAdd(&values.at(j1, i0), s0 * s3 * initial_values.at(j, i));
            atomicAdd(&values.at(j1, i1), s0 * s2 * initial_values.at(j, i));
        }
    }
}

template<typename T>
__global__ void calculate_divergence_kernel(element_accessor<T>       divergence,
                                            element_accessor<T> const horizontal_velocities,
                                            element_accessor<T> const vertical_velocities,
                                            size_t                    rows,
                                            size_t                    cols,
                                            float const               h) {
    for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y) {
        for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x) {
            divergence.at(j, i) = -0.5f * h * (horizontal_velocities.at(j + 1, i) - horizontal_velocities.at(j - 1, i) +
                                               vertical_velocities.at(j, i + 1) - vertical_velocities.at(j, i - 1));
        }
    }
}

template<typename T>
__global__ void p_iteration_kernel(element_accessor<T>       p,
                                   element_accessor<T> const p_previous,
                                   element_accessor<T> const divergence,
                                   size_t                    rows,
                                   size_t                    cols) {
    for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y) {
        for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x) {
            p.at(j, i) = (divergence.at(j, i) + p_previous.at(j + 1, i) + p_previous.at(j - 1, i) +
                          p_previous.at(j, i + 1) + p_previous.at(j, i - 1)) / 4.0f;
        }
    }
}

template<typename T>
__global__ void remove_p_kernel(element_accessor<T>       horizontal_velocities,
                                element_accessor<T>       vertical_velocities,
                                element_accessor<T> const p,
                                size_t                    rows,
                                size_t                    cols,
                                float const               h) {
    for(size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1; i < rows - 1; i += blockDim.y * gridDim.y) {
        for(size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1; j < cols - 1; j += blockDim.x * gridDim.x) {
            horizontal_velocities.at(j, i) -= 0.5f * (p.at(j + 1, i) - p.at(j - 1, i)) / h;
            vertical_velocities.at(j, i) -= 0.5f * (p.at(j, i + 1) - p.at(j, i - 1)) / h;
        }
    }
}


fluid_solver_gpu::fluid_solver_gpu(size_t const rows, size_t const cols)
    : m_rows{ rows }
    , m_cols{ cols }
    , m_density_buffer{ cols, rows }
    , m_horizontal_velocity_buffer{ cols, rows }
    , m_vertical_velocity_buffer{ cols, rows }
    , m_source_buffer{ cols, rows }
    , m_temp_buffer_1{ cols, rows }
    , m_temp_buffer_2{ cols, rows }
    , m_temp_buffer_3{ cols, rows } {}

fluid_solver_gpu::~fluid_solver_gpu() {}

void fluid_solver_gpu::solve(grid<float>& density_grid,
                             grid<float> const& density_source_grid,
                             float const diffusion_rate,
                             grid<float>& horizontal_velocity_grid,
                             grid<float>& vertical_velocity_grid,
                             grid<float> const& horizontal_velocity_source_grid,
                             grid<float> const& vertical_velocity_source_grid,
                             float const viscosity,
                             float const dt) {
    // Upload density and velocities to gpu
    copy(m_density_buffer, density_grid);
    copy(m_horizontal_velocity_buffer, horizontal_velocity_grid);
    copy(m_vertical_velocity_buffer, vertical_velocity_grid);

    // Solve density related terms
    add_sources(m_density_buffer, density_source_grid, dt);
    diffuse(m_density_buffer, &fluid_solver_gpu::set_boundary_continuous, diffusion_rate, dt, 15);
    advect(m_density_buffer, m_horizontal_velocity_buffer, m_vertical_velocity_buffer, &fluid_solver_gpu::set_boundary_continuous, dt, true);
    smooth(m_density_buffer);

    // Solve velocity related terms
    add_sources(m_horizontal_velocity_buffer, horizontal_velocity_source_grid, dt);
    add_sources(m_vertical_velocity_buffer, vertical_velocity_source_grid, dt);
    diffuse(m_horizontal_velocity_buffer, &fluid_solver_gpu::set_boundary_opposite_horizontal, viscosity, dt, 15);
    diffuse(m_vertical_velocity_buffer, &fluid_solver_gpu::set_boundary_opposite_vertical, viscosity, dt, 15);
    project(m_horizontal_velocity_buffer, m_vertical_velocity_buffer, 20);
    copy(m_temp_buffer_2, m_horizontal_velocity_buffer);
    copy(m_temp_buffer_3, m_vertical_velocity_buffer);
    advect(m_horizontal_velocity_buffer, m_temp_buffer_2, m_temp_buffer_3, &fluid_solver_gpu::set_boundary_opposite_horizontal, dt, false);
    advect(m_vertical_velocity_buffer, m_temp_buffer_2, m_temp_buffer_3, &fluid_solver_gpu::set_boundary_opposite_vertical, dt, false);
    project(m_horizontal_velocity_buffer, m_vertical_velocity_buffer, 20);

    // Download density and velocities to the cpu
    copy(density_grid, m_density_buffer);
    copy(horizontal_velocity_grid, m_horizontal_velocity_buffer);
    copy(vertical_velocity_grid, m_vertical_velocity_buffer);
}

void fluid_solver_gpu::set_boundary_continuous(linear_buffer<float>& values_buffer, size_t rows, size_t cols) {
    kernel_launcher::launch_1d(&set_boundary_continuous_kernel<float>, std::max(cols, rows),
                               values_buffer.accessor(), rows, cols);
    cudaDeviceSynchronize();
}

void fluid_solver_gpu::set_boundary_opposite_horizontal(linear_buffer<float>& values_buffer, size_t rows, size_t cols) {
    kernel_launcher::launch_1d(&set_boundary_opposite_horizontal_kernel<float>, std::max(cols, rows),
                               values_buffer.accessor(), rows, cols);
    cudaDeviceSynchronize();
}

void fluid_solver_gpu::set_boundary_opposite_vertical(linear_buffer<float>& values_buffer, size_t rows, size_t cols) {
    kernel_launcher::launch_1d(&set_boundary_opposite_vertical_kernel<float>, std::max(cols, rows),
                               values_buffer.accessor(), rows, cols);
    cudaDeviceSynchronize();
}

void fluid_solver_gpu::add_sources(linear_buffer<float>& values_buffer,
                                   grid<float> const  source_grid,
                                   float const        dt) {
    copy(m_source_buffer, source_grid);

    // Add the sources
    kernel_launcher::launch_2d(&add_sources_kernel<float>, m_cols, m_rows,
                               values_buffer.accessor(), m_source_buffer.accessor(),
                               m_rows, m_cols, dt);
    cudaDeviceSynchronize();
}

void fluid_solver_gpu::diffuse(linear_buffer<float>& values_buffer,
                               std::function<void(linear_buffer<float>&, size_t, size_t)> set_boundary,
                               float const rate,
                               float const dt,
                               size_t iteration_count) {
    linear_buffer<float>& initial_values_buffer = m_temp_buffer_1;
    linear_buffer<float>& previous_values_buffer = m_temp_buffer_2;

    // First temporary buffer contains the initial values
    // Second temporary buffer contains the values of the previous iteration
    copy(initial_values_buffer, values_buffer);
    for(size_t k = 0; k < iteration_count; ++k) {
        copy(previous_values_buffer, values_buffer);

        kernel_launcher::launch_2d(&diffuse_iteration_kernel<float>, m_cols, m_rows,
                                   values_buffer.accessor(), previous_values_buffer.accessor(), initial_values_buffer.accessor(),
                                   m_rows, m_cols,
                                   rate, dt);
        cudaDeviceSynchronize();

        set_boundary(values_buffer, m_rows, m_cols);
    }
}

void fluid_solver_gpu::smooth(linear_buffer<float>& values_buffer) {
    linear_buffer<float>& initial_values = m_temp_buffer_1;

    copy(initial_values, values_buffer);

    kernel_launcher::launch_2d(&smooth_kernel<float>, m_cols, m_rows,
                               values_buffer.accessor(), initial_values.accessor(), m_rows, m_cols);

    cudaDeviceSynchronize();
}

void fluid_solver_gpu::advect(linear_buffer<float>& values_buffer,
                              linear_buffer<float> const& horizontal_velocity_buffer,
                              linear_buffer<float> const& vertical_velocity_buffer,
                              std::function<void(linear_buffer<float>&, size_t, size_t)> set_boundary,
                              float const dt,
                              bool trace) {
    linear_buffer<float>& initial_values = m_temp_buffer_1;
    copy(initial_values, values_buffer);

    float dt0 = sqrt(m_rows * m_cols) * dt;

    if(trace) {
        values_buffer.clear(0);
        cudaDeviceSynchronize();

        kernel_launcher::launch_2d(&advect_trace_kernel<float>, m_cols, m_rows,
                                   values_buffer.accessor(), initial_values.accessor(),
                                   horizontal_velocity_buffer.accessor(), vertical_velocity_buffer.accessor(),
                                   m_rows, m_cols,
                                   dt0);
        cudaDeviceSynchronize();
    } else {
        kernel_launcher::launch_2d(&advect_kernel<float>, m_cols, m_rows,
                                   values_buffer.accessor(), initial_values.accessor(),
                                   horizontal_velocity_buffer.accessor(), vertical_velocity_buffer.accessor(),
                                   m_rows, m_cols,
                                   dt0);
        cudaDeviceSynchronize();
    }

    set_boundary(values_buffer, m_rows, m_cols);
}

void fluid_solver_gpu::project(linear_buffer<float>& horizontal_velocity_buffer,
                               linear_buffer<float>& vertical_velocity_buffer,
                               size_t iteration_count) {
    linear_buffer<float>& divergence_buffer = m_temp_buffer_1;
    linear_buffer<float>& p_buffer = m_temp_buffer_2;
    divergence_buffer.clear(0);
    p_buffer.clear(0);
    cudaDeviceSynchronize();

    float h = 1.0f / sqrtf(m_rows * m_cols);
    kernel_launcher::launch_2d(&calculate_divergence_kernel<float>, m_cols, m_rows,
                               divergence_buffer.accessor(),
                               horizontal_velocity_buffer.accessor(),
                               vertical_velocity_buffer.accessor(),
                               m_rows,
                               m_cols,
                               h);
    cudaDeviceSynchronize();
    set_boundary_continuous(divergence_buffer, m_rows, m_cols);

    linear_buffer<float>& p_previous_buffer = m_temp_buffer_3;
    for(size_t k = 0; k < iteration_count; ++k) {
        copy(p_previous_buffer, p_buffer);

        kernel_launcher::launch_2d(&p_iteration_kernel<float>, m_cols, m_rows,
                                   p_buffer.accessor(),
                                   p_previous_buffer.accessor(),
                                   divergence_buffer.accessor(),
                                   m_rows,
                                   m_cols);
        cudaDeviceSynchronize();

        set_boundary_continuous(p_buffer, m_rows, m_cols);
    }

    kernel_launcher::launch_2d(&remove_p_kernel<float>, m_cols, m_rows,
                               horizontal_velocity_buffer.accessor(),
                               vertical_velocity_buffer.accessor(),
                               p_buffer.accessor(),
                               m_rows,
                               m_cols,
                               h);
    cudaDeviceSynchronize();

    set_boundary_opposite_horizontal(horizontal_velocity_buffer, m_rows, m_cols);
    set_boundary_opposite_vertical(vertical_velocity_buffer, m_rows, m_cols);
}
