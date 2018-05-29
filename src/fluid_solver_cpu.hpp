#pragma once

#include "fluid_solver.hpp"
#include "grid.hpp"

#include <functional>

/**
 * Fluid solver implemented as single-threaded on the CPU.
 */
class fluid_solver_cpu : public fluid_solver {
public:
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
    static void set_boundary_continuous(grid<float>& grid_);

    static void set_boundary_opposite_horizontal(grid<float>& grid_);

    static void set_boundary_opposite_vertical(grid<float>& grid_);

    void add_sources(grid<float>& grid_,
                     grid<float> const& source_grid,
                     float const dt);

    void diffuse(grid<float>& grid_,
                 std::function<void(grid<float>&)> set_boundary,
                 float const rate,
                 float const dt,
                 size_t const iteration_count);

    void advect(grid<float>& grid_,
                grid<float> const& horizontal_velocity_grid,
                grid<float> const& vertical_velocity_grid,
                std::function<void(grid<float>&)> set_boundary,
                float const dt,
                bool trace);

    void project(grid<float>& horizontal_velocity_grid,
                 grid<float>& vertical_velocity_grid,
                 size_t const iteration_count);
};
