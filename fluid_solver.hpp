#pragma once

#include "grid.hpp"

class fluid_solver
{
public:
  virtual ~fluid_solver() = default;

  virtual void solve(grid<float>& density_grid, 
                     grid<float> const& density_source_grid,
                     float const diffusion_rate,
                     grid<float>& horizontal_velocity_grid,
                     grid<float>& vertical_velocity_grid,
                     grid<float> const& horizontal_velocity_source_grid,
                     grid<float> const& vertical_velocity_source_grid,
                     float const viscosity,
                     float const dt) = 0;
};
