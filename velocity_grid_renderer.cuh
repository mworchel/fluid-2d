#pragma once

#include "grid.hpp"
#include "grid_renderer.hpp"
#include "pitched_buffer.hpp"

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

	void draw_gpu(grid<float> const& horizontal_velocity, grid<float> const& vertical_velocity, sf::RenderTarget& target);

	void draw(grid<float> const& horizontal_velocity, grid<float> const& vertical_velocity, sf::RenderTarget& target);

private:
	std::vector<sf::Vertex> m_line_vertices;
	pitched_buffer<sf_line> m_line_vertex_buffer;
	pitched_buffer<float>   m_horizontal_velocity_buffer;
	pitched_buffer<float>   m_vertical_velocity_buffer;
};