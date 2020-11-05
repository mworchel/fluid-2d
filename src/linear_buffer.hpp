#pragma once

#include "gpu_buffer.hpp"

#include <cuda_runtime.h>

/**
* Representation of a two-dimensional linear GPU buffer
*/
template<typename T>
class linear_buffer : public gpu_buffer<T> {
public:
    linear_buffer(size_t _width, size_t _height)
        : gpu_buffer<T>(_width, _height) {
        cudaMalloc(&m_data, byte_size());
    }

    ~linear_buffer() {
        cudaFree(m_data);
    }

    virtual cudaError_t clear(int value) override {
        return cudaMemset(m_data, value, byte_size());
    }

    virtual T* data() override {
        return m_data;
    }

    virtual T const* data() const override {
        return m_data;
    }

    virtual element_accessor<T> accessor() override {
        return element_accessor<T>{ m_data, sizeof(T) * gpu_buffer<T>::width() };
    }

    virtual element_accessor<T> const accessor() const override {
        return element_accessor<T>{ m_data, sizeof(T) * gpu_buffer<T>::width() };
    }

    size_t byte_size() const {
        return sizeof(T) * gpu_buffer<T>::width() * gpu_buffer<T>::height();
    }

private:
    T*     m_data;
    size_t m_pitch;
};
