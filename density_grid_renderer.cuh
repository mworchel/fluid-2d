#pragma once

#include "grid.hpp"
#include "grid_renderer.hpp"
#include "linear_buffer.hpp"

#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/Texture.hpp>

#include <vector>

/**
 * Representation of a pixel in an SFML texture.
 */
struct sf_pixel {
    uint8_t R;
    uint8_t G;
    uint8_t B;
    uint8_t A;
};

/**
 * Multipliers for individual color channels.
 */
struct color_multipliers {
    float r;
    float g;
    float b;
};

/**
 * Class for drawing floating point grids (representing density) to a render target.
 */
class density_grid_renderer : public grid_renderer {
public:
    /**
     * Construct a new renderer.
     *
     * @param rows Number of rows of the grids that are drawn with the renderer.
     * @param cols Number of columns of the grids that are drawn with the renderer.
     */
    density_grid_renderer(size_t const rows, size_t const cols);

    ~density_grid_renderer();

    /**
     * Draw a density grid to a render target.
     *
     * @details Let r, g, b be the given multipliers, then a density d is drawn with with the color
     *          c = (r * d, g * d, b * d). Each channel is clamped to the range 0, ... , 255.
     */
    void draw(sf::RenderTarget& target, grid<float> const& grid, color_multipliers const& multipliers);

private:
    linear_buffer<float>    m_grid_buffer;
    sf::Texture             m_texture;
    std::vector<sf_pixel>   m_image;
    linear_buffer<sf_pixel> m_image_buffer;
};