#pragma once

#include "grid.hpp"

#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/Texture.hpp>

#include <vector>

struct sf_pixel
{
  uint8_t R;
  uint8_t G;
  uint8_t B;
  uint8_t A;
};

class density_grid_renderer
{
public:
  density_grid_renderer(size_t const rows, size_t const cols);

  ~density_grid_renderer();

  void draw_gpu(grid<float> const& grid, sf::RenderTarget& target);

  void draw(grid<float> const& grid, sf::RenderTarget& target);

  bool coordinates_to_cell(float const x, float const y, const sf::RenderTarget& target, size_t& i, size_t& j);

private:
  size_t                m_rows;
  size_t                m_cols;
  float*                m_grid_buffer;
  size_t                m_grid_buffer_pitch;
  sf::Texture           m_texture;
  std::vector<sf_pixel> m_image;
  sf_pixel*             m_image_buffer;
  size_t                m_image_buffer_pitch;
};