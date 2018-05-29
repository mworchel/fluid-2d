#pragma once

#include "grid.hpp"
#include "grid_renderer.hpp"
#include "linear_buffer.hpp"

#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/Vertex.hpp>

/**
 * Representation of a SFML line as start and end vertex.
 */
struct sf_line {
    sf::Vertex start;
    sf::Vertex end;
};

/**
* Class for drawing two floating point grids (representing horizontal and vertical velocity) to a render target.
*/
class velocity_grid_renderer : public grid_renderer {
public:
    /**
    * Construct a new renderer.
    *
    * @param rows Number of rows of the grids that are drawn with the renderer.
    * @param cols Number of columns of the grids that are drawn with the renderer.
    */
    velocity_grid_renderer(size_t const rows, size_t const cols);

    ~velocity_grid_renderer();

    /**
     * Draw combined horizontal and vertical velocities to a target.
     */
    void draw(sf::RenderTarget& target, grid<float> const& horizontal_velocity, grid<float> const& vertical_velocity);

private:
    std::vector<sf_line>    m_line_vertices;
    linear_buffer<sf_line>  m_line_vertex_buffer;
    linear_buffer<float>    m_horizontal_velocity_buffer;
    linear_buffer<float>    m_vertical_velocity_buffer;
};