#pragma once

#include <SFML/Graphics/RenderTarget.hpp>

class grid_renderer
{
public:
	virtual ~grid_renderer() = default;

	virtual bool coordinates_to_cell(float const x, float const y, const sf::RenderTarget& target, size_t& i, size_t& j) const;

protected:
	grid_renderer(size_t const rows, size_t const cols);

	inline size_t rows() const
	{
		return m_rows;
	}

	inline size_t cols() const
	{
		return m_cols;
	}

private:
	size_t                m_rows;
	size_t                m_cols;
};