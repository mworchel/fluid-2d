#pragma once

#include "grid.hpp"
#include "grid_renderer.hpp"
#include "linear_buffer.hpp"

#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/Vertex.hpp>

struct sf_line
{
	sf::Vertex start;
	sf::Vertex end;
};

class velocity_grid_renderer : public grid_renderer
{
public:
	velocity_grid_renderer(size_t const rows, size_t const cols);

	~velocity_grid_renderer();

	void draw(grid<float> const& horizontal_velocity, grid<float> const& vertical_velocity, sf::RenderTarget& target);

private:
	std::vector<sf_line>    m_line_vertices;
	linear_buffer<sf_line>  m_line_vertex_buffer;
	linear_buffer<float>    m_horizontal_velocity_buffer;
	linear_buffer<float>    m_vertical_velocity_buffer;
};