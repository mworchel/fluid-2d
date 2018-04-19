#pragma once

#include "grid.hpp"
#include "grid_renderer.hpp"
#include "pitched_buffer.hpp"

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

class density_grid_renderer : public grid_renderer
{
public:
  density_grid_renderer(size_t const rows, size_t const cols);

  ~density_grid_renderer();

  void draw_gpu(grid<float> const& grid, sf::RenderTarget& target);

  void draw(grid<float> const& grid, sf::RenderTarget& target);

private:
  pitched_buffer<float>    m_grid_buffer;
  sf::Texture              m_texture;
  std::vector<sf_pixel>    m_image;
  pitched_buffer<sf_pixel> m_image_buffer;
};