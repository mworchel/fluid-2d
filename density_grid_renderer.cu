#include "density_grid_renderer.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <SFML/Graphics.hpp>

template<typename T>
__global__ void grid_to_image_kernel(const T*     grid,
                                     const size_t grid_pitch,
                                     sf_pixel*    image,
                                     const size_t image_pitch,
                                     size_t       rows,
                                     size_t       cols)
{
  for(size_t i = blockDim.y * blockIdx.y + threadIdx.y; i < rows; i += blockDim.y * gridDim.y)
  {
    for(size_t j = blockDim.x * blockIdx.x + threadIdx.x; j < cols; j += blockDim.x * gridDim.x)
    {
      const T&  grid_value = *((T*)((char*)grid + i * grid_pitch) + j);
      sf_pixel& image_value = *((sf_pixel*)((char*)image + i * image_pitch) + j);

      image_value.R = image_value.G = image_value.B = static_cast<uint8_t>(max(0.0f, min(1.0f, grid_value)) * 255.0);
      image_value.A = 255U;
    }
  }
}

density_grid_renderer::density_grid_renderer(size_t const rows, size_t const cols)
  : m_rows{ rows }
  , m_cols{ cols }
{
  m_texture.create(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows));
  m_image.resize(m_rows * m_cols);
  cudaMallocPitch(&m_image_buffer, &m_image_buffer_pitch, sizeof(sf_pixel) * cols, rows);
  cudaMallocPitch(&m_grid_buffer, &m_grid_buffer_pitch, sizeof(float) * cols, rows);
}

density_grid_renderer::~density_grid_renderer()
{
  cudaFree(m_image_buffer);
  cudaFree(m_grid_buffer);
}

void density_grid_renderer::draw_gpu(grid<float> const& grid, sf::RenderTarget& target)
{
  // Copy the grid to the buffer
  auto error = cudaMemcpy2D(m_grid_buffer, m_grid_buffer_pitch, grid.data(), m_cols * sizeof(float), m_cols * sizeof(float), m_rows, cudaMemcpyHostToDevice);

  unsigned int block_dim = 32;
  unsigned int grid_dim_x = static_cast<unsigned int>((m_cols + block_dim - 1) / block_dim);
  unsigned int grid_dim_y = static_cast<unsigned int>((m_rows + block_dim - 1) / block_dim);

  grid_to_image_kernel << <dim3(grid_dim_x, grid_dim_y), dim3(block_dim, block_dim) >> > (m_grid_buffer, m_grid_buffer_pitch, m_image_buffer, m_image_buffer_pitch, m_rows, m_cols);
  error = cudaDeviceSynchronize();

  // 
  error = cudaMemcpy2D(m_image.data(), sizeof(sf_pixel) * m_cols, m_image_buffer, m_image_buffer_pitch, sizeof(sf_pixel) * m_cols, m_rows, cudaMemcpyDeviceToHost);
  m_texture.update(reinterpret_cast<uint8_t*>(m_image.data()), static_cast<unsigned int>(m_cols), static_cast<unsigned int>(m_rows), 0U, 0U);
  
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

  for(size_t y = 0; y < m_rows; ++y)
  {
    for(size_t x = 0; x < m_cols; ++x)
    {
      const auto& gray_value = static_cast<uint8_t>(std::min(1.f, std::max(0.f, grid(y, x))) * 255);
      sf::Color color{ gray_value, gray_value, gray_value };

      shape.setFillColor(color);
      shape.setPosition(x * shape.getSize().x, y * shape.getSize().y);
      target.draw(shape);
    }
  }
}

bool density_grid_renderer::coordinates_to_cell(float const x, float const y, const sf::RenderTarget& target, size_t& i, size_t& j)
{
  if(x < 0.f || y < 0.f || x >= target.getSize().x || y >= target.getSize().y)
  {
    return false;
  }

  auto target_size = target.getSize();
  i = static_cast<size_t>(m_rows * (y / target_size.y));
  j = static_cast<size_t>(m_cols * (x / target_size.x));

  return true;
}