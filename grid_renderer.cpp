#include "grid_renderer.hpp"

bool grid_renderer::coordinates_to_cell(float const x, float const y, const sf::RenderTarget & target, size_t & i, size_t & j) const {
    if(x < 0.f || y < 0.f || x >= target.getSize().x || y >= target.getSize().y) {
        return false;
    }

    // The grid is assumed to be stretched to the size of the target
    auto target_size = target.getSize();
    i = static_cast<size_t>(m_rows * (y / target_size.y));
    j = static_cast<size_t>(m_cols * (x / target_size.x));

    return true;
}

grid_renderer::grid_renderer(size_t const rows, size_t const cols)
    : m_rows(rows)
    , m_cols(cols) {}
