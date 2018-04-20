#pragma once

#include "grid.hpp"
#include "grid_renderer.hpp"
#include "linear_buffer.hpp"

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

struct color_multipliers
{
  float r;
  float g;
  float b;
};

class density_grid_renderer : public grid_renderer
{
public:
  density_grid_renderer(size_t const rows, size_t const cols);

  ~density_grid_renderer();

  void draw(sf::RenderTarget& target, grid<float> const& grid, color_multipliers const& multipliers);

private:
  linear_buffer<float>    m_grid_buffer;
  sf::Texture             m_texture;
  std::vector<sf_pixel>   m_image;
  linear_buffer<sf_pixel> m_image_buffer;
};