#pragma once

#include "fluid_solver.hpp"
#include "grid.hpp"

#include <functional>

class fluid_solver_cpu : public fluid_solver
{
public:
  void solve(grid<float>& density_grid,
             grid<float> const& density_source_grid,
             float const diffusion_rate,
             float const dt) override;

private:
  static void boundary_continuity(grid<float>& _grid);

  static void boundary_opposite(grid<float>& _grid);

  void add_sources(grid<float>& _grid,
                   grid<float> const& source_grid,
                   float const dt);

  void diffuse(grid<float>& _grid,
               std::function<void(grid<float>&)> set_boundary,
               float const rate,
               float const dt);
};
