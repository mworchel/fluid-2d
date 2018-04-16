#pragma once

#include "grid.hpp"

class fluid_solver
{
public:
  virtual ~fluid_solver() = default;

  virtual void solve(grid<float>& density_grid, 
                     grid<float> const& density_source_grid,
                     float const diffusion_rate,
                     /*grid<T>& velocity_grid,*/
                     float const dt) = 0;
};
