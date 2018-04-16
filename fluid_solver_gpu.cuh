#pragma once

#include "fluid_solver.hpp"

class fluid_solver_gpu : public fluid_solver
{
public:
  fluid_solver_gpu(size_t const rows, size_t const cols);

  ~fluid_solver_gpu();

  void solve(grid<float>& density_grid, 
             grid<float> const& density_source_grid,
             float const diffusion_rate,
             float const dt) override;

private:
  size_t m_rows;
  size_t m_cols;
  float* m_density_buffer;
  float* m_density_source_buffer;
  float* m_density_previous_buffer;
};