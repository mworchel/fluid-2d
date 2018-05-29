#include "simulation.hpp"
#include "fluid_solver_cpu.hpp"
#include "fluid_solver_gpu.cuh"

#include <algorithm>

simulation::simulation(simulation_config const& config)
    : m_config(config)
    , m_density_grid{ config.height, config.width, 0.f }
    , m_horizontal_velocity_grid{ config.height, config.width, 0.f }
    , m_vertical_velocity_grid{ config.height, config.width, 0.f }
    , m_horizontal_velocity_source_grid{ config.height, config.width, 0.f }
    , m_vertical_velocity_source_grid{ config.height, config.width, 0.f }
    , m_density_source_grid{ config.height, config.width, 0.f }
    , m_density_grid_renderer{ config.height, config.width }
    , m_velocity_grid_renderer{ config.height, config.width } {
    switch(config.solver_type) {
        case(solver_type::cpu):
            m_solver = std::make_unique<fluid_solver_cpu>();
            break;
        case(solver_type::gpu):
            m_solver = std::make_unique<fluid_solver_gpu>(config.height, config.width);
            break;
        default:
            break;
    }
}

void simulation::reset() {
    // Clear the density and velocity grid
    std::fill(std::begin(m_density_grid), std::end(m_density_grid), 0.f);
    std::fill(std::begin(m_horizontal_velocity_grid), std::end(m_horizontal_velocity_grid), 0.f);
    std::fill(std::begin(m_vertical_velocity_grid), std::end(m_vertical_velocity_grid), 0.f);
}

bool simulation::to_density_cell(float x, float y, const sf::RenderTarget & target, size_t & i, size_t & j) {
    return m_density_grid_renderer.coordinates_to_cell(x, y, target, i, j);
}

bool simulation::to_velocity_cell(float const x, float const y, sf::RenderTarget const & target, size_t & i, size_t & j) {
    return m_velocity_grid_renderer.coordinates_to_cell(x, y, target, i, j);
}

void simulation::add_density_source(size_t const i, size_t const j, float const value) {
    m_density_source_grid(i, j) += value * m_config.width * m_config.height;
}

void simulation::add_velocity_source(size_t const i, size_t const j, float const horizontal_value, float const vertical_value) {
    m_horizontal_velocity_source_grid(i, j) += horizontal_value * m_config.width * m_config.height;
    m_vertical_velocity_source_grid(i, j) += vertical_value * m_config.width * m_config.height;
}

void simulation::update(const std::chrono::duration<float>& dt) {
    m_solver->solve(m_density_grid, m_density_source_grid,
                    m_config.diffusion_rate,
                    m_horizontal_velocity_grid, m_vertical_velocity_grid,
                    m_horizontal_velocity_source_grid, m_vertical_velocity_source_grid,
                    m_config.viscosity,
                    dt.count());

    // Clear the source grid, since the sources have been considered in the solve step
    std::fill(std::begin(m_density_source_grid), std::end(m_density_source_grid), 0.0f);
    std::fill(std::begin(m_horizontal_velocity_source_grid), std::end(m_horizontal_velocity_source_grid), 0.0f);
    std::fill(std::begin(m_vertical_velocity_source_grid), std::end(m_vertical_velocity_source_grid), 0.0f);
}

void simulation::draw(sf::RenderTarget& target, color_multipliers const& density_color_multipliers, bool draw_density, bool draw_velocity) {
    if(draw_density)
        m_density_grid_renderer.draw(target, m_density_grid, density_color_multipliers);

    if(draw_velocity)
        m_velocity_grid_renderer.draw(target, m_horizontal_velocity_grid, m_vertical_velocity_grid);
}
