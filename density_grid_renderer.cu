#include "density_grid_renderer.cuh"
#include "kernel_launcher.hpp"
#include "utilities.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <SFML/Graphics.hpp>

template<typename T>
__global__ void grid_to_image_kernel(element_accessor<T> const  grid,
                                     element_accessor<sf_pixel> image,
                                     size_t                     rows,
                                     size_t                     cols,
                                     color_multipliers          multipliers)
{
  for(size_t i = blockDim.y * blockIdx.y + threadIdx.y; i < rows; i += blockDim.y * gridDim.y)
  {
    for(size_t j = blockDim.x * blockIdx.x + threadIdx.x; j < cols; j += blockDim.x * gridDim.x)
    {
      const T&  grid_value = grid.at(j, i);
      sf_pixel& image_value = image.at(j, i);

      float clamped_value = max(0.0f, min(1.0f, grid_value));

      image_value.R = static_cast<uint8_t>(max(0.f, min(255.f, multipliers.r * grid_value)));
      image_value.G = static_cast<uint8_t>(max(0.f, min(255.f, multipliers.g * grid_value)));
      image_value.B = static_cast<uint8_t>(max(0.f, min(255.f, multipliers.b * grid_value)));
      image_value.A = 255U;
    }
  }
}

density_grid_renderer::density_grid_renderer(size_t const _rows, size_t const _cols)
  : grid_renderer(_rows, _cols)
  , m_grid_buffer{ _cols, _rows }
  , m_image_buffer{ _cols, _rows }
{
  m_texture.create(static_cast<unsigned int>(_cols), static_cast<unsigned int>(_rows));
  m_image.resize(_rows * _cols);
}

density_grid_renderer::~density_grid_renderer()
{}

void density_grid_renderer::draw(sf::RenderTarget& target, grid<float> const& grid, color_multipliers const& multipliers)
{
  auto error = copy(m_grid_buffer, grid);

  kernel_launcher::launch_2d(&grid_to_image_kernel<float>, cols(), rows(),
                             m_grid_buffer.accessor(), m_image_buffer.accessor(), rows(), cols(), multipliers);
  error = cudaDeviceSynchronize();

  // 
  error = copy(m_image.data(), cols(), rows(), m_image_buffer, cudaMemcpyDeviceToHost);
  m_texture.update(reinterpret_cast<uint8_t*>(m_image.data()), static_cast<unsigned int>(cols()), static_cast<unsigned int>(rows()), 0U, 0U);

  sf::Sprite sprite{ m_texture };
  sprite.setScale(target.getSize().x / sprite.getLocalBounds().width, target.getSize().y / sprite.getLocalBounds().height);
  target.draw(sprite);
}