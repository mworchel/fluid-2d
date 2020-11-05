#pragma once

#include "grid.hpp"
#include "fluid_solver.hpp"
#include "density_grid_renderer.cuh"
#include "velocity_grid_renderer.cuh"

#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Window/Event.hpp>

#include <chrono>
#include <memory>

enum class solver_type { gpu, cpu };

/**
 * Simulation parameters.
 */
struct simulation_config {
    size_t width = 0U;
    size_t height = 0U;
    solver_type solver = solver_type::gpu;
    float diffusion_rate = 0.001f;
    float viscosity = 1.0f;
};

/**
 * Representation of the fluid simulation.
 */
class simulation {
public:
    simulation(simulation_config const& config);

    void reset();

    /**
     * Transform a target position (x, y) into a density cell index (i, j).
     */
    bool to_density_cell(float const x, float const y, sf::RenderTarget const& target, size_t& i, size_t& j);

    /**
     * Transform a target position (x, y) into a velocity cell index (i, j).
     */
    bool to_velocity_cell(float const x, float const y, sf::RenderTarget const& target, size_t& i, size_t& j);

    /**
     * Add source density to the cell (i, j).
     */
    void add_density_source(size_t const i, size_t const j, float const value);

    /**
     * Add source velocity to the cell (i, j).
     */
    void add_velocity_source(size_t const i, size_t const j, float const horizontal_value, float const vertical_value);

    /**
     * Perform one update step of the simulation.
     */
    void update(const std::chrono::duration<float>& dt);

    /**
     * Draw the simulation to a render target.
     */
    void draw(sf::RenderTarget& target, color_multipliers const& density_color_multipliers, bool draw_density, bool draw_velocity);

private:
    simulation_config             m_config;

    grid<float>                   m_density_grid;
    grid<float>                   m_density_source_grid;
    density_grid_renderer         m_density_grid_renderer;
    grid<float>                   m_horizontal_velocity_grid;
    grid<float>                   m_vertical_velocity_grid;
    grid<float>                   m_horizontal_velocity_source_grid;
    grid<float>                   m_vertical_velocity_source_grid;
    velocity_grid_renderer        m_velocity_grid_renderer;
    std::unique_ptr<fluid_solver> m_solver;
};
