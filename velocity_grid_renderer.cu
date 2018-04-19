#include "velocity_grid_renderer.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void velocity_to_lines_kernel(element_accessor<float> horizontal_velocities,
										 element_accessor<float> vertical_velocities,
										 element_accessor<sf_line> line_vertices,
										 size_t rows, size_t cols)
{
	for (size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < rows; i += blockDim.y * gridDim.y)
	{
		for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < cols; j += blockDim.x * gridDim.x)
		{
			sf_line& line = line_vertices.at(j, i);

			line.start.position.x = j;
			line.start.position.y = i;
			line.end.position.x = j;
			line.end.position.y = i;

			float2 velocity{ horizontal_velocities.at(j, i), vertical_velocities.at(j, i) };
			float magnitude = sqrtf(velocity.x * velocity.x + velocity.y * velocity.y);
			if (magnitude > 0.f)
			{
				line.end.position.x = j + velocity.x / magnitude;
				line.end.position.y = i + velocity.y / magnitude;

				line.start.color.r = line.start.color.g = line.start.color.b = line.start.color.a = 255;
				line.end.color.r = line.end.color.g = line.end.color.b = line.start.color.a = 255;
			}
		}
	}
}

velocity_grid_renderer::velocity_grid_renderer(size_t const _rows, size_t const _cols)
	: grid_renderer{ _rows, _cols }
	, m_line_vertex_buffer{ _cols, _rows }
	, m_horizontal_velocity_buffer{ _cols, _rows }
	, m_vertical_velocity_buffer{ _cols, _rows }
{
	m_line_vertices.resize(2 * _rows * _cols);
}

velocity_grid_renderer::~velocity_grid_renderer()
{
}

void velocity_grid_renderer::draw_gpu(grid<float> const& horizontal_velocity, grid<float> const& vertical_velocity, sf::RenderTarget& target)
{
	// Upload velocities to gpu
	cudaError error = cudaMemcpy2D(m_horizontal_velocity_buffer.buffer(), m_horizontal_velocity_buffer.pitch(),
								   horizontal_velocity.data(), sizeof(float) * cols(),
								   sizeof(float) * cols(), rows(),
								   cudaMemcpyHostToDevice);

	error = cudaMemcpy2D(m_vertical_velocity_buffer.buffer(), m_vertical_velocity_buffer.pitch(),
						 vertical_velocity.data(), sizeof(float) * cols(),
						 sizeof(float) * cols(), rows(),
						 cudaMemcpyHostToDevice);

	error = cudaMemset2D(m_line_vertex_buffer.buffer(), m_line_vertex_buffer.pitch(), 0, sizeof(sf_line) * cols(), rows());

	unsigned int block_dim = 32;
	unsigned int grid_dim_x = static_cast<unsigned int>((cols() + block_dim - 1) / block_dim);
	unsigned int grid_dim_y = static_cast<unsigned int>((rows() + block_dim - 1) / block_dim);


	velocity_to_lines_kernel << <dim3(grid_dim_x, grid_dim_y), dim3(block_dim, block_dim) >> > (m_horizontal_velocity_buffer.accessor(),
																								m_vertical_velocity_buffer.accessor(),
																								m_line_vertex_buffer.accessor(),
																								rows(), cols());
	error = cudaDeviceSynchronize();

	// 
	error = cudaMemcpy2D(m_line_vertices.data(), sizeof(sf_line) * cols(), 
						 m_line_vertex_buffer.buffer(), m_line_vertex_buffer.pitch(),
						 sizeof(sf_line) * cols(), rows(), cudaMemcpyDeviceToHost);

	target.draw(reinterpret_cast<sf::Vertex*>(m_line_vertices.data()), m_line_vertices.size(), sf::PrimitiveType::Lines);
}

void velocity_grid_renderer::draw(grid<float> const& horizontal_velocity, grid<float> const& vertical_velocity, sf::RenderTarget& target)
{

}