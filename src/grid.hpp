#pragma once

#include <vector>

/**
 * Representation of a two-dimensional grid of values.
 */
template<typename T>
class grid {
public:
    grid(size_t rows, size_t cols, T initial_value)
        : m_rows(rows)
        , m_cols(cols)
        , m_data(rows * cols, initial_value) {}

    size_t rows() const {
        return m_rows;
    }

    size_t cols() const {
        return m_rows;
    }

    const T* data() const {
        return m_data.data();
    }

    T* data() {
        return m_data.data();
    }

    /**
     * Access the grid value at row i and column j.
     */
    T operator() (const size_t i, const size_t j) const {
        return m_data[i * m_cols + j];
    }

    /**
    * Access the grid value at row i and column j.
    */
    T& operator() (const size_t i, const size_t j) {
        return m_data[i * m_cols + j];
    }

    auto begin() {
        return std::begin(m_data);
    }

    auto end() {
        return std::end(m_data);
    }

    auto cbegin() const {
        return std::cbegin(m_data);
    }

    auto cend() const {
        return std::cend(m_data);
    }

private:
    size_t         m_rows;
    size_t         m_cols;
    std::vector<T> m_data;
};
