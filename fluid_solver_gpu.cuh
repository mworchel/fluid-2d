#pragma once

#include "fluid_solver.hpp"

#include <functional>

class fluid_solver_gpu : public fluid_solver
{
public:
  fluid_solver_gpu(size_t const rows, size_t const cols);

  ~fluid_solver_gpu();

  void solve(grid<float>& density_grid, 
             grid<float> const& density_source_grid,
             float const diffusion_rate,
             grid<float>& horizontal_velocity_grid,
             grid<float>& vertical_velocity_grid,
             grid<float> const& horizontal_velocity_source_grid,
             grid<float> const& vertical_velocity_source_grid,
             float const viscosity,
             float const dt) override;

private:
  static void set_corners(float* values_buffer, size_t rows, size_t cols);

  static void set_boundary_continuous(float* values_buffer, size_t rows, size_t cols);
  
  static void set_boundary_opposite_horizontal(float* values_buffer, size_t rows, size_t cols);
  
  static void set_boundary_opposite_vertical(float* values_buffer, size_t rows, size_t cols);

  void add_sources(float* values_buffer, 
                   float const* source_data, 
                   float const dt);

  void diffuse(float* values_buffer,
               std::function<void(float*, size_t, size_t)> set_boundary,
               float const rate,
               float const dt,
               size_t iteration_count);

  void advect(float* values_buffer,
              float* horizontal_velocity_buffer,
              float* vertical_velocity_buffer,
              std::function<void(float*, size_t, size_t)> set_boundary,
              float const dt);

  void project(float* horizontal_velocity_buffer,
               float* vertical_velocity_buffer);

  inline size_t buffer_size()
  {
    return sizeof(float) * m_rows * m_cols;
  }

  size_t m_rows;
  size_t m_cols;
  float* m_density_buffer;
  float* m_horizontal_velocity_buffer;
  float* m_vertical_velocity_buffer;
  float* m_source_buffer;
  float* m_temp_buffer_1;
  float* m_temp_buffer_2;
  float* m_temp_buffer_3;
};