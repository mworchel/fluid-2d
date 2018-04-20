#pragma once

#include "fluid_solver.hpp"
#include "gpu_buffer.hpp"
#include "linear_buffer.hpp"

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
  static void set_boundary_continuous(linear_buffer<float>& values_buffer, size_t rows, size_t cols);
  
  static void set_boundary_opposite_horizontal(linear_buffer<float>& values_buffer, size_t rows, size_t cols);
  
  static void set_boundary_opposite_vertical(linear_buffer<float>& values_buffer, size_t rows, size_t cols);

  void add_sources(linear_buffer<float>& values_buffer,
                   grid<float> const  source_grid,
                   float const        dt);

  void diffuse(linear_buffer<float>& values_buffer,
               std::function<void(linear_buffer<float>&, size_t, size_t)> set_boundary,
               float const rate,
               float const dt,
               size_t iteration_count);

  void smooth(linear_buffer<float>& values_buffer);

  void advect(linear_buffer<float>& values_buffer,
              linear_buffer<float> const& horizontal_velocity_buffer,
              linear_buffer<float> const& vertical_velocity_buffer,
              std::function<void(linear_buffer<float>&, size_t, size_t)> set_boundary,
              float const dt,
              bool trace);

  void project(linear_buffer<float>& horizontal_velocity_buffer,
               linear_buffer<float>& vertical_velocity_buffer,
               size_t iteration_count);

  inline size_t buffer_size()
  {
    return sizeof(float) * m_rows * m_cols;
  }

  size_t m_rows;
  size_t m_cols;
  linear_buffer<float> m_density_buffer;
  linear_buffer<float> m_horizontal_velocity_buffer;
  linear_buffer<float> m_vertical_velocity_buffer;
  linear_buffer<float> m_source_buffer;
  linear_buffer<float> m_temp_buffer_1;
  linear_buffer<float> m_temp_buffer_2;
  linear_buffer<float> m_temp_buffer_3;
};