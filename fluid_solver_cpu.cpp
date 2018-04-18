#include "fluid_solver_cpu.hpp"

#include <algorithm>

void fluid_solver_cpu::solve(grid<float>& density_grid, 
                             grid<float> const& density_source_grid, 
                             float const diffusion_rate,
                             float dt)
{
  add_sources(density_grid, density_source_grid, dt);
  diffuse(density_grid, &fluid_solver_cpu::boundary_continuity, diffusion_rate, dt);

  grid<float> u{ density_grid.rows(), density_grid.cols(), 0.f };
  grid<float> v{ density_grid.rows(), density_grid.cols(), 0.001f };
  advect(density_grid, u, v, &fluid_solver_cpu::boundary_continuity, dt);
}

void fluid_solver_cpu::boundary_continuity(grid<float>& _grid)
{
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

  _grid(0, 0)                               = 0.5f * (_grid(0, 1) + _grid(1, 0));
  _grid(0, _grid.cols() - 1)                = 0.5f * (_grid(0, _grid.cols() - 2) + _grid(1, _grid.cols() - 1));
  _grid(_grid.rows() - 1, 0)                = 0.5f * (_grid(_grid.rows() - 1, 1) + _grid(_grid.rows() - 2, 0));
  _grid(_grid.rows() - 1, _grid.cols() - 1) = 0.5f * (_grid(_grid.rows() - 1, _grid.cols() - 2) + _grid(_grid.rows() - 2, _grid.cols() - 1));
}

void fluid_solver_cpu::boundary_opposite(grid<float>& _grid)
{
  // TODO
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
                               std::function<void(grid<float>&)> set_boundary,
                               float const diffusion_rate, 
                               float const dt)
{
  float a = dt * static_cast<float>(_grid.rows() * _grid.cols()) *diffusion_rate;

  grid<float> initial_grid = _grid;

  for(size_t k = 0; k < 20; k++)
  {
    for(size_t i = 1; i < _grid.rows() - 1; i++)
    {
      for(size_t j = 1; j < _grid.cols() - 1; j++)
      {
        _grid(i, j) = (initial_grid(i, j) + a * (_grid(i - 1, j) + _grid(i + 1, j) +
                                                 _grid(i, j - 1) + _grid(i, j + 1))) / (1.f + 4.f * a);
      }
    }

    set_boundary(_grid);
  }
}

void fluid_solver_cpu::advect(grid<float>& _grid, 
                              grid<float>& horizontal_velocity_grid, 
                              grid<float>& vertical_velocity_grid, 
                              std::function<void(grid<float>&)> set_boundary, 
                              float const dt)
{
  grid<float> initial_grid = _grid;

  float dt0 = _grid.rows() * _grid.cols() * dt;

  for(size_t i = 1; i < _grid.rows() - 1; i++)
  {
    for(size_t j = 1; j < _grid.cols() - 1; j++)
    {
      float x = j - dt0 * horizontal_velocity_grid(i, j);
      float y = i - dt0 * vertical_velocity_grid(i, j);
      x = std::max(1.5f, std::min(_grid.cols() - 1.5f, x));
      y = std::max(1.5f, std::min(_grid.rows() - 1.5f, y));

      size_t j0 = static_cast<size_t>(x);
      size_t i0 = static_cast<size_t>(y);
      size_t j1 = j0 + 1;
      size_t i1 = i0 + 1;
      float s0 = x - j0;
      float s1 = 1 - s0;
      float s2 = y - i0;
      float s3 = 1 - s2;

      _grid(i, j) = s3 * (s1 * initial_grid(i0, j0) + s0 * initial_grid(i0, j1)) +
        s2 * (s1 * initial_grid(i1, j0) + s0 * initial_grid(i1, j1));
    }
  }

  set_boundary(_grid);
}
