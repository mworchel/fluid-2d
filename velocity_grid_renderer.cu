#include "velocity_grid_renderer.cuh"
#include "kernel_launcher.hpp"
#include "utilities.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template<typename T>
__global__ void velocity_to_lines_kernel(element_accessor<T> horizontal_velocities,
                                         element_accessor<T> vertical_velocities,
                                         element_accessor<sf_line> line_vertices,
                                         size_t rows, size_t cols,
                                         float horizontal_scale, float vertical_scale) {
    for(size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < rows; i += blockDim.y * gridDim.y) {
        for(size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < cols; j += blockDim.x * gridDim.x) {
            sf_line& line = line_vertices.at(j, i);

            line.start.position.x = j * horizontal_scale;
            line.start.position.y = i * vertical_scale;
            line.end.position.x = j * horizontal_scale;
            line.end.position.y = i * vertical_scale;

            // Determine line end point by following the velocity direction
            float2 velocity{ horizontal_velocities.at(j, i), vertical_velocities.at(j, i) };
            //float magnitude = sqrtf(velocity.x * velocity.x + velocity.y * velocity.y);
            //if(magnitude > 0.f) {
            line.end.position.x += 10000.f * velocity.x / sqrtf(rows * cols);
            line.end.position.y += 10000.f * velocity.y / sqrtf(rows * cols);

            line.start.color.r = line.start.color.g = line.start.color.b = line.start.color.a = 255;
            line.end.color.r = line.end.color.g = line.end.color.b = line.start.color.a = 255;
            //}
        }
    }
}

velocity_grid_renderer::velocity_grid_renderer(size_t const _rows, size_t const _cols)
    : grid_renderer(_rows, _cols)
    , m_line_vertex_buffer{ _cols, _rows }
    , m_horizontal_velocity_buffer{ _cols, _rows }
    , m_vertical_velocity_buffer{ _cols, _rows } {
    m_line_vertices.resize(_rows * _cols);
}

velocity_grid_renderer::~velocity_grid_renderer() {}

void velocity_grid_renderer::draw(sf::RenderTarget& target, grid<float> const& horizontal_velocity, grid<float> const& vertical_velocity) {
    // Upload both velocities to gpu
    cudaError error = copy(m_horizontal_velocity_buffer, horizontal_velocity);
    error = copy(m_vertical_velocity_buffer, vertical_velocity);

    // Transform the separate velocities to lines
    error = m_line_vertex_buffer.clear(0);
    kernel_launcher::launch_2d(&velocity_to_lines_kernel<float>, cols(), rows(),
                               m_horizontal_velocity_buffer.accessor(), m_vertical_velocity_buffer.accessor(),
                               m_line_vertex_buffer.accessor(),
                               rows(), cols(), static_cast<float>(target.getSize().x) / cols(), static_cast<float>(target.getSize().y) / rows());
    error = cudaDeviceSynchronize();

    // 
    error = copy(m_line_vertices.data(), cols(), rows(), m_line_vertex_buffer, cudaMemcpyDeviceToHost);

    target.draw(reinterpret_cast<sf::Vertex*>(m_line_vertices.data()), 2 * m_line_vertices.size(), sf::PrimitiveType::Lines);
}