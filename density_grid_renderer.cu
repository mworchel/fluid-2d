#include "density_grid_renderer.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <SFML/Graphics.hpp>

template<typename T>
__global__ void grid_to_image_kernel(element_accessor<T>        grid,
									 element_accessor<sf_pixel> image,
                                     size_t                     rows,
                                     size_t                     cols)
{
  for(size_t i = blockDim.y * blockIdx.y + threadIdx.y; i < rows; i += blockDim.y * gridDim.y)
  {
    for(size_t j = blockDim.x * blockIdx.x + threadIdx.x; j < cols; j += blockDim.x * gridDim.x)
    {
	  const T&  grid_value = grid.at(j, i);
      sf_pixel& image_value = image.at(j, i);

      float clamped_value = max(0.0f, min(1.0f, grid_value));

      image_value.R = static_cast<uint8_t>(max(0.f, min(255.f, 200.f * grid_value)));
      image_value.G = static_cast<uint8_t>(max(0.f, min(255.f, 100.f * grid_value)));
      image_value.B = static_cast<uint8_t>(max(0.f, min(255.f, 50.f * grid_value)));
      image_value.A = 255U;
    }
  }
}

density_grid_renderer::density_grid_renderer(size_t const _rows, size_t const _cols)
	: grid_renderer{ _rows, _cols }
	, m_grid_buffer{ _cols, _rows }
	, m_image_buffer{ _cols, _rows }
{
  m_texture.create(static_cast<unsigned int>(_cols), static_cast<unsigned int>(_rows));
  m_image.resize(_rows * _cols);
}

density_grid_renderer::~density_grid_renderer()
{
}

void density_grid_renderer::draw_gpu(grid<float> const& grid, sf::RenderTarget& target)
{
  // Copy the grid to the buffer
  auto error = cudaMemcpy2D(m_grid_buffer.buffer(), m_grid_buffer.pitch(), grid.data(), cols() * sizeof(float), cols() * sizeof(float), rows(), cudaMemcpyHostToDevice);

  unsigned int block_dim = 32;
  unsigned int grid_dim_x = static_cast<unsigned int>((cols() + block_dim - 1) / block_dim);
  unsigned int grid_dim_y = static_cast<unsigned int>((rows() + block_dim - 1) / block_dim);

  grid_to_image_kernel << <dim3(grid_dim_x, grid_dim_y), dim3(block_dim, block_dim) >> > (m_grid_buffer.accessor(), m_image_buffer.accessor(), rows(), cols());
  error = cudaDeviceSynchronize();

  // 
  error = cudaMemcpy2D(m_image.data(), sizeof(sf_pixel) * cols(), m_image_buffer.buffer(), m_image_buffer.pitch(), sizeof(sf_pixel) * cols(), rows(), cudaMemcpyDeviceToHost);
  m_texture.update(reinterpret_cast<uint8_t*>(m_image.data()), static_cast<unsigned int>(cols()), static_cast<unsigned int>(rows()), 0U, 0U);
  
  sf::Sprite sprite{ m_texture };
  sprite.setScale(target.getSize().x / sprite.getLocalBounds().width, target.getSize().y / sprite.getLocalBounds().height);
  target.draw(sprite);
}

void density_grid_renderer::draw(grid<float> const& grid, sf::RenderTarget& target)
{
  const auto target_size = target.getSize();

  float rectangle_width = target_size.x / static_cast<float>(grid.cols());
  float rectangle_height = target_size.y / static_cast<float>(grid.rows());

  auto shape = sf::RectangleShape{ { rectangle_width, rectangle_height } };

  for(size_t y = 0; y < rows(); ++y)
  {
    for(size_t x = 0; x < cols(); ++x)
    {
      const auto& gray_value = static_cast<uint8_t>(std::min(1.f, std::max(0.f, grid(y, x))) * 255);
      sf::Color color{ gray_value, gray_value, gray_value };

      shape.setFillColor(color);
      shape.setPosition(x * shape.getSize().x, y * shape.getSize().y);
      target.draw(shape);
    }
  }
}