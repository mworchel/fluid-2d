#pragma once

#include "fluid_solver.hpp"
#include "grid.hpp"

class fluid_solver_cpu : public fluid_solver
{
public:
  void solve(grid<float>& density_grid,
             grid<float> const& density_source_grid,
             float const diffusion_rate,
             float const dt) override;

private:
  void add_sources(grid<float>& _grid,
                   grid<float> const& source_grid,
                   float const dt);

  void diffuse(grid<float>& _grid,
               float const rate,
               float const dt);
};
