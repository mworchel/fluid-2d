#pragma once

#include <utility>

class kernel_launcher {
public:
    inline static constexpr unsigned int block_dim_1d() {
        return 64U;
    }

    inline static constexpr unsigned int block_dim_2d() {
        return 32U;
    }

    inline static constexpr unsigned int grid_dim(unsigned int const block_dim, unsigned int const count) {
        return (count + block_dim - 1) / block_dim;
    }

    template<typename kernel_type, typename... arg_types>
    inline static void launch_1d(kernel_type kernel, unsigned int width, arg_types&&... _args) {
        kernel << <dim3(grid_dim(block_dim_1d(), width)), dim3(block_dim_1d()) >> > (std::forward<arg_types>(_args)...);
    }

    template<typename kernel_type, typename... arg_types>
    inline static void launch_2d(kernel_type kernel, unsigned int width, unsigned int height, arg_types&&... _args) {
        kernel << <dim3(grid_dim(block_dim_2d(), width), grid_dim(block_dim_2d(), height)), dim3(block_dim_2d(), block_dim_2d()) >> > (std::forward<arg_types>(_args)...);
    }
};
