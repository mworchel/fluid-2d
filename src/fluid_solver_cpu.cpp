#include "fluid_solver_cpu.hpp"

#include <algorithm>
#include <cmath>

void fluid_solver_cpu::solve(grid<float>& density_grid,
                             grid<float> const& density_source_grid,
                             float const diffusion_rate,
                             grid<float>& horizontal_velocity_grid,
                             grid<float>& vertical_velocity_grid,
                             grid<float> const& horizontal_velocity_source_grid,
                             grid<float> const& vertical_velocity_source_grid,
                             float const viscosity,
                             float dt) {
    // Solve the density related terms
    add_sources(density_grid, density_source_grid, dt);
    diffuse(density_grid, &fluid_solver_cpu::set_boundary_continuous, diffusion_rate, dt, 20U);
    advect(density_grid, horizontal_velocity_grid, vertical_velocity_grid, &fluid_solver_cpu::set_boundary_continuous, dt, true);

    // Solve the velocity related terms
    add_sources(horizontal_velocity_grid, horizontal_velocity_source_grid, dt);
    add_sources(vertical_velocity_grid, vertical_velocity_source_grid, dt);
    diffuse(horizontal_velocity_grid, &fluid_solver_cpu::set_boundary_opposite_horizontal, viscosity, dt, 20U);
    diffuse(vertical_velocity_grid, &fluid_solver_cpu::set_boundary_opposite_vertical, viscosity, dt, 20U);
    project(horizontal_velocity_grid, vertical_velocity_grid, 20U);
    grid<float> temp_horizontal_velocity = horizontal_velocity_grid;
    grid<float> temp_vertical_velocity = vertical_velocity_grid;
    advect(horizontal_velocity_grid, temp_horizontal_velocity, temp_vertical_velocity, &fluid_solver_cpu::set_boundary_opposite_horizontal, dt, false);
    advect(vertical_velocity_grid, temp_horizontal_velocity, temp_vertical_velocity, &fluid_solver_cpu::set_boundary_opposite_vertical, dt, false);
    project(horizontal_velocity_grid, vertical_velocity_grid, 20U);
}

void fluid_solver_cpu::set_boundary_continuous(grid<float>& grid_) {
    for(size_t i = 0; i < grid_.rows(); ++i) {
        grid_(i, 0) = grid_(i, 1);
        grid_(i, grid_.cols() - 1) = grid_(i, grid_.cols() - 2);
    }

    for(size_t j = 0; j < grid_.cols(); ++j) {
        grid_(0, j) = grid_(1, j);
        grid_(grid_.rows() - 1, j) = grid_(grid_.rows() - 2, j);
    }

    grid_(0, 0) = 0.5f * (grid_(0, 1) + grid_(1, 0));
    grid_(0, grid_.cols() - 1) = 0.5f * (grid_(0, grid_.cols() - 2) + grid_(1, grid_.cols() - 1));
    grid_(grid_.rows() - 1, 0) = 0.5f * (grid_(grid_.rows() - 1, 1) + grid_(grid_.rows() - 2, 0));
    grid_(grid_.rows() - 1, grid_.cols() - 1) = 0.5f * (grid_(grid_.rows() - 1, grid_.cols() - 2) + grid_(grid_.rows() - 2, grid_.cols() - 1));
}

void fluid_solver_cpu::set_boundary_opposite_horizontal(grid<float>& grid_) {
    for(size_t i = 0; i < grid_.rows(); ++i) {
        grid_(i, 0) = -grid_(i, 1);
        grid_(i, grid_.cols() - 1) = -grid_(i, grid_.cols() - 2);
    }

    for(size_t j = 0; j < grid_.cols(); ++j) {
        grid_(0, j) = grid_(1, j);
        grid_(grid_.rows() - 1, j) = grid_(grid_.rows() - 2, j);
    }

    grid_(0, 0) = 0.5f * (grid_(0, 1) + grid_(1, 0));
    grid_(0, grid_.cols() - 1) = 0.5f * (grid_(0, grid_.cols() - 2) + grid_(1, grid_.cols() - 1));
    grid_(grid_.rows() - 1, 0) = 0.5f * (grid_(grid_.rows() - 1, 1) + grid_(grid_.rows() - 2, 0));
    grid_(grid_.rows() - 1, grid_.cols() - 1) = 0.5f * (grid_(grid_.rows() - 1, grid_.cols() - 2) + grid_(grid_.rows() - 2, grid_.cols() - 1));
}

void fluid_solver_cpu::set_boundary_opposite_vertical(grid<float>& grid_) {

    for(size_t j = 0; j < grid_.cols(); ++j) {
        grid_(0, j) = -grid_(1, j);
        grid_(grid_.rows() - 1, j) = -grid_(grid_.rows() - 2, j);
    }

    for(size_t i = 0; i < grid_.rows(); ++i) {
        grid_(i, 0) = grid_(i, 1);
        grid_(i, grid_.cols() - 1) = grid_(i, grid_.cols() - 2);
    }

    grid_(0, 0) = 0.5f * (grid_(0, 1) + grid_(1, 0));
    grid_(0, grid_.cols() - 1) = 0.5f * (grid_(0, grid_.cols() - 2) + grid_(1, grid_.cols() - 1));
    grid_(grid_.rows() - 1, 0) = 0.5f * (grid_(grid_.rows() - 1, 1) + grid_(grid_.rows() - 2, 0));
    grid_(grid_.rows() - 1, grid_.cols() - 1) = 0.5f * (grid_(grid_.rows() - 1, grid_.cols() - 2) + grid_(grid_.rows() - 2, grid_.cols() - 1));
}

void fluid_solver_cpu::add_sources(grid<float>& grid_,
                                   grid<float> const& source_grid,
                                   float const dt) {
    for(size_t i = 1; i < grid_.rows() - 1; ++i) {
        for(size_t j = 1; j < grid_.cols() - 1; ++j) {
            grid_(i, j) += dt * source_grid(i, j);
        }
    }
}

void fluid_solver_cpu::diffuse(grid<float>& grid_,
                               std::function<void(grid<float>&)> set_boundary,
                               float const diffusion_rate,
                               float const dt,
                               size_t const iteration_count) {
    float a = dt * static_cast<float>(grid_.rows() * grid_.cols()) *diffusion_rate;

    grid<float> initial_grid = grid_;

    for(size_t k = 0; k < iteration_count; ++k) {
        for(size_t i = 1; i < grid_.rows() - 1; ++i) {
            for(size_t j = 1; j < grid_.cols() - 1; ++j) {
                grid_(i, j) = (initial_grid(i, j) + a * (grid_(i - 1, j) + grid_(i + 1, j) +
                                                         grid_(i, j - 1) + grid_(i, j + 1))) / (1.f + 4.f * a);
            }
        }

        set_boundary(grid_);
    }
}

void fluid_solver_cpu::advect(grid<float>& grid_,
                              grid<float> const& horizontal_velocity_grid,
                              grid<float> const& vertical_velocity_grid,
                              std::function<void(grid<float>&)> set_boundary,
                              float const dt,
                              bool trace) {
    grid<float> initial_grid = grid_;
    size_t rows = grid_.rows();
    size_t cols = grid_.cols();

    float dt0 = std::sqrt(static_cast<float>(rows * cols)) * dt;
    if(trace) {
        std::fill(std::begin(grid_), std::end(grid_), 0.f);

        for(size_t i = 1; i < rows - 1; ++i) {
            for(size_t j = 1; j < cols - 1; ++j) {
                float x = j + dt0 * horizontal_velocity_grid(i, j);
                float y = i + dt0 * vertical_velocity_grid(i, j);

                if(x < 0.5f || x > cols - 1.5f || y < 0.5f || y > rows - 1.5f) continue;

                size_t j0 = static_cast<size_t>(x);
                size_t i0 = static_cast<size_t>(y);
                size_t j1 = j0 + 1;
                size_t i1 = i0 + 1;

                float s0 = x - j0;
                float s1 = 1 - s0;
                float s2 = y - i0;
                float s3 = 1 - s2;

                grid_(i0, j0) += s1 * s3 * initial_grid(i, j);
                grid_(i1, j0) += s1 * s2 * initial_grid(i, j);
                grid_(i0, j1) += s0 * s3 * initial_grid(i, j);
                grid_(i1, j1) += s0 * s2 * initial_grid(i, j);
            }
        }
    } else {
        for(size_t i = 1; i < rows - 1; ++i) {
            for(size_t j = 1; j < cols - 1; ++j) {
                float x = j - dt0 * horizontal_velocity_grid(i, j);
                float y = i - dt0 * vertical_velocity_grid(i, j);
                x = std::max(1.5f, std::min(cols - 1.5f, x));
                y = std::max(1.5f, std::min(rows - 1.5f, y));

                size_t j0 = static_cast<size_t>(x);
                size_t i0 = static_cast<size_t>(y);
                size_t j1 = j0 + 1;
                size_t i1 = i0 + 1;
                float s0 = x - j0;
                float s1 = 1 - s0;
                float s2 = y - i0;
                float s3 = 1 - s2;

                grid_(i, j) = s3 * (s1 * initial_grid(i0, j0) + s0 * initial_grid(i0, j1)) +
                    s2 * (s1 * initial_grid(i1, j0) + s0 * initial_grid(i1, j1));
            }
        }
    }

    set_boundary(grid_);
}

void fluid_solver_cpu::project(grid<float>& horizontal_velocity_grid, grid<float>& vertical_velocity_grid, size_t const iteration_count) {
    auto rows = horizontal_velocity_grid.rows();
    auto cols = horizontal_velocity_grid.cols();

    float h = 1.0f / std::sqrt(static_cast<float>(rows * cols));

    grid<float> divergence{ rows, cols, 0.f };
    grid<float> p{ rows, cols, 0.f };

    for(size_t i = 1; i < rows - 1; ++i) {
        for(size_t j = 1; j < cols - 1; ++j) {
            divergence(i, j) = -0.5f * h * (horizontal_velocity_grid(i, j + 1) - horizontal_velocity_grid(i, j - 1) +
                                            vertical_velocity_grid(i + 1, j) - vertical_velocity_grid(i - 1, j));
        }
    }
    set_boundary_continuous(divergence);

    for(size_t k = 0; k < iteration_count; ++k) {
        for(size_t i = 1; i < rows - 1; ++i) {
            for(size_t j = 1; j < cols - 1; ++j) {
                p(i, j) = (divergence(i, j) + p(i, j + 1) + p(i, j - 1) +
                           p(i + 1, j) + p(i - 1, j)) / 4.0f;
            }
        }
        set_boundary_continuous(p);
    }

    for(size_t i = 1; i < rows - 1; ++i) {
        for(size_t j = 1; j < cols - 1; ++j) {
            horizontal_velocity_grid(i, j) -= 0.5f * (p(i, j + 1) - p(i, j - 1)) / h;
            vertical_velocity_grid(i, j) -= 0.5f * (p(i + 1, j) - p(i - 1, j)) / h;
        }
    }

    set_boundary_opposite_horizontal(horizontal_velocity_grid);
    set_boundary_opposite_vertical(vertical_velocity_grid);
}
