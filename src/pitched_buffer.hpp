#pragma once

#include "gpu_buffer.hpp"

#include <cuda_runtime.h>

/**
 * Representation of a two-dimensional pitched GPU buffer
 */
template<typename T>
class pitched_buffer : public gpu_buffer<T> {
public:
    pitched_buffer(size_t _width, size_t _height)
        : gpu_buffer(_width, _height) {
        cudaMallocPitch(&m_data, &m_pitch, byte_width(), height());
    }

    ~pitched_buffer() {
        cudaFree(m_data);
    }

    virtual cudaError_t clear(int value) override {
        return cudaMemset2D(m_data, m_pitch, value, byte_width(), height());
    }

    virtual T* data() override {
        return m_data;
    }

    virtual T const* data() const override {
        return m_data;
    }

    virtual element_accessor<T> accessor() override {
        return element_accessor<T>{ m_data, m_pitch };
    }

    virtual element_accessor<T> const accessor() const override {
        return element_accessor<T>{ m_data, m_pitch };
    }

    size_t byte_width() const {
        return sizeof(T) * width();
    }

    size_t pitch() const {
        return m_pitch;
    }

private:
    T*     m_data;
    size_t m_pitch;
};
