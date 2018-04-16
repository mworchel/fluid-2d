#pragma once

#include "grid.hpp"
#include "fluid_solver.hpp"
#include "density_grid_renderer.cuh"

#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Window/Event.hpp>

#include <chrono>
#include <memory>

enum class solver_type { gpu, cpu };

struct simulation_config
{
  size_t width            = 0U;
  size_t height           = 0U;
  solver_type solver_type = solver_type::gpu;
  float diffusion_rate    = 0.001f;
  float viscosity         = 1.0f;
};

class simulation
{
public:
  simulation(simulation_config const& config);

  bool to_density_cell(float const x, float const y, sf::RenderTarget const& target, size_t& i, size_t& j);

  void add_density_source(size_t const i, size_t const j, float const value);

  void update(const std::chrono::duration<float>& dt);

  void draw(sf::RenderTarget& target);

private:
  simulation_config             m_config;

  grid<float>                   m_density_grid;
  grid<float>                   m_density_source_grid;
  density_grid_renderer         m_density_grid_renderer;
  std::unique_ptr<fluid_solver> m_solver;
};
