#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <cmath>
#include <numeric>

namespace nnm {
    class Matrix {
    private:
        std::vector<float> data;
        size_t rows;
        size_t cols;

        static constexpr size_t STRASSEN_THRESHOLD = 64;

        Matrix multiplyAVX(const Matrix &other) const {
            if (cols != other.rows) {
                throw std::invalid_argument("Matrix dimensions do not match for multiplication");
            }
            Matrix result(rows, other.cols);
#pragma omp parallel for
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < other.cols; j += 4) {
                    __m256d sum = _mm256_setzero_pd();
                    for (size_t k = 0; k < cols; ++k) {
                        __m256d a = _mm256_set1_pd(static_cast<double>((*this)(i, k)));
                        __m256d b = _mm256_setr_pd(
                                static_cast<double>(other(k, j)),
                                static_cast<double>(other(k, j + 1)),
                                static_cast<double>(other(k, j + 2)),
                                static_cast<double>(other(k, j + 3))
                        );
                        sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
                    }
                    double temp[4];
                    _mm256_storeu_pd(temp, sum);
                    for (int k = 0; k < 4 && j + k < other.cols; ++k) {
                        result(i, j + k) = static_cast<float>(temp[k]);
                    }
                }
            }
            return result;
        }

        Matrix strassen(const Matrix &other) const {
            if (rows != cols || other.rows != other.cols || rows != other.rows) {
                throw std::invalid_argument("Matrices must be square and of the same size for Strassen's algorithm");
            }

            size_t n = rows;
            if (n <= STRASSEN_THRESHOLD) {
                return multiplyAVX(other);
            }

            size_t new_size = n / 2;
            Matrix a11(new_size, new_size), a12(new_size, new_size), a21(new_size, new_size), a22(new_size, new_size);
            Matrix b11(new_size, new_size), b12(new_size, new_size), b21(new_size, new_size), b22(new_size, new_size);

            // Split matrices
            for (size_t i = 0; i < new_size; ++i) {
                for (size_t j = 0; j < new_size; ++j) {
                    a11(i, j) = (*this)(i, j);
                    a12(i, j) = (*this)(i, j + new_size);
                    a21(i, j) = (*this)(i + new_size, j);
                    a22(i, j) = (*this)(i + new_size, j + new_size);

                    b11(i, j) = other(i, j);
                    b12(i, j) = other(i, j + new_size);
                    b21(i, j) = other(i + new_size, j);
                    b22(i, j) = other(i + new_size, j + new_size);
                }
            }

            // Recursive steps
            Matrix p1 = (a11 + a22).strassen(b11 + b22);
            Matrix p2 = (a21 + a22).strassen(b11);
            Matrix p3 = a11.strassen(b12 - b22);
            Matrix p4 = a22.strassen(b21 - b11);
            Matrix p5 = (a11 + a12).strassen(b22);
            Matrix p6 = (a21 - a11).strassen(b11 + b12);
            Matrix p7 = (a12 - a22).strassen(b21 + b22);

            // Calculate result quadrants
            Matrix c11 = p1 + p4 - p5 + p7;
            Matrix c12 = p3 + p5;
            Matrix c21 = p2 + p4;
            Matrix c22 = p1 - p2 + p3 + p6;

            // Combine result
            Matrix result(n, n);
            for (size_t i = 0; i < new_size; ++i) {
                for (size_t j = 0; j < new_size; ++j) {
                    result(i, j) = c11(i, j);
                    result(i, j + new_size) = c12(i, j);
                    result(i + new_size, j) = c21(i, j);
                    result(i + new_size, j + new_size) = c22(i, j);
                }
            }

            return result;
        }

    public:
        Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(rows * cols, 0.0f) {}

        Matrix(size_t rows, size_t cols, float value) : rows(rows), cols(cols), data(rows * cols, value) {}


        float &operator()(size_t i, size_t j) {
            return data[i * cols + j];
        }

        const float &operator()(size_t i, size_t j) const {
            return data[i * cols + j];
        }

        Matrix operator+(const Matrix &other) const {
            if (rows != other.rows || cols != other.cols) {
                throw std::invalid_argument("Matrix dimensions do not match for addition");
            }
            Matrix result(rows, cols);
            for (size_t i = 0; i < rows * cols; ++i) {
                result.data[i] = data[i] + other.data[i];
            }
            return result;
        }

        Matrix operator-(const Matrix &other) const {
            if (rows != other.rows || cols != other.cols) {
                throw std::invalid_argument("Matrix dimensions do not match for subtraction");
            }
            Matrix result(rows, cols);
            for (size_t i = 0; i < rows * cols; ++i) {
                result.data[i] = data[i] - other.data[i];
            }
            return result;
        }

        Matrix operator*(const Matrix &other) const {
            if (cols != other.rows) {
                throw std::invalid_argument("Matrix dimensions do not match for multiplication");
            }
            if (rows == cols && other.rows == other.cols && (rows & (rows - 1)) == 0) {
                return strassen(other);
            }
            return multiplyAVX(other);
        }

        void print() const {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    std::cout << (*this)(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }

        [[nodiscard]] Matrix transpose() const {
            Matrix result(cols, rows);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(j, i) = (*this)(i, j);
                }
            }
            return result;
        }

        void reshape(size_t new_rows, size_t new_cols) {
            if (new_rows * new_cols != rows * cols) {
                throw std::invalid_argument(
                        "New dimensions must have the same number of elements as the original matrix");
            }
            rows = new_rows;
            cols = new_cols;
        }


        [[nodiscard]] Matrix pad(size_t pad_h, size_t pad_w) const {
            Matrix padded(rows + 2 * pad_h, cols + 2 * pad_w);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    padded(i + pad_h, j + pad_w) = (*this)(i, j);
                }
            }
            return padded;
        }

        [[nodiscard]] Matrix subMatrix(size_t start_row, size_t start_col, size_t sub_rows, size_t sub_cols) const {
            Matrix sub(sub_rows, sub_cols);
            for (size_t i = 0; i < sub_rows; ++i) {
                for (size_t j = 0; j < sub_cols; ++j) {
                    sub(i, j) = (*this)(start_row + i, start_col + j);
                }
            }
            return sub;
        }

        [[nodiscard]] Matrix elementWiseMul(const Matrix &other) const {
            if (rows != other.rows || cols != other.cols) {
                throw std::invalid_argument("Matrix dimensions do not match for element-wise multiplication");
            }
            Matrix result(rows, cols);
            for (size_t i = 0; i < rows * cols; ++i) {
                result.data[i] = data[i] * other.data[i];
            }
            return result;
        }

        [[nodiscard]] float sum() const {
            return std::accumulate(data.begin(), data.end(), 0.0f);
        }


        [[nodiscard]] size_t getRows() const { return rows; }

        [[nodiscard]] size_t getCols() const { return cols; }

    };
} // namespace nnm