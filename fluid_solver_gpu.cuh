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
  void add_sources(float* values_buffer, 
                   float const* source_data, 
                   float const dt);

  void diffuse(float* values_buffer,
               float const rate,
               float const dt,
               size_t iteration_count
               /*TODO: Boundary*/);

  void set_boundary(float* values_buffer);

  inline size_t buffer_size()
  {
    return sizeof(float) * m_rows * m_cols;
  }

  size_t m_rows;
  size_t m_cols;
  float* m_density_buffer;
  float* m_source_buffer;
  float* m_diffuse_initial_buffer;
  float* m_diffuse_previous_buffer;
};