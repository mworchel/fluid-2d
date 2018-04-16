#include "fluid_solver_cpu.h"

void fluid_solver_cpu::solve(grid<float>& density_grid, 
                             grid<float> const& density_source_grid, 
                             float const diffusion_rate,
                             float dt)
{
  add_sources(density_grid, density_source_grid, dt);
  diffuse(density_grid, diffusion_rate, dt);
}

void fluid_solver_cpu::add_sources(grid<float>& _grid, 
                                   grid<float> const& source_grid, 
                                   float const dt)
{
  for(size_t i = 1; i < _grid.rows() - 1; i++)
  {
    for(size_t j = 1; j < _grid.cols() - 1; j++)
    {
      _grid(i, j) += dt * source_grid(i, j);
    }
  }
}

void fluid_solver_cpu::diffuse(grid<float>& _grid, 
                               float const diffusion_rate, 
                               float const dt)
{
  float a = dt * static_cast<float>(_grid.rows() * _grid.cols()) *diffusion_rate;

  grid<float> previous_grid = _grid;

  for(size_t k = 0; k < 20; k++)
  {
    for(size_t i = 1; i < _grid.rows() - 1; i++)
    {
      for(size_t j = 1; j < _grid.cols() - 1; j++)
      {
        _grid(i, j) = (previous_grid(i, j) + a * (_grid(i - 1, j) + previous_grid(i + 1, j) +
                                                  _grid(i, j - 1) + previous_grid(i, j + 1))) / (1.f + 4.f * a);
      }
    }

    for(size_t i = 0; i < _grid.rows(); i++)
    {
      _grid(i, 0) = _grid(i, 1);
      _grid(i, _grid.cols() - 1) = _grid(i, _grid.cols() - 2);
    }

    for(size_t j = 0; j < _grid.cols(); j++)
    {
      _grid(0, j) = _grid(1, j);
      _grid(_grid.rows() - 1, j) = _grid(_grid.rows() - 2, j);
    }

    previous_grid = _grid;
  }
}
