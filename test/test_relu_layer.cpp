#include <gtest/gtest.h>
#include "ReLULayer.h"
#include "Matrix.h"
#include <random>
#include <iostream>

namespace nnm {

    class ReLULayerTest : public ::testing::Test {
    protected:
        static Matrix generate_random_matrix(size_t rows, size_t cols, int min_val, int max_val) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(min_val, max_val);

            Matrix mat(rows, cols);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    mat(i, j) = static_cast<float>(dis(gen));
                }
            }
            return mat;
        }

        static void print_matrix(const Matrix& mat) {
            for (size_t i = 0; i < mat.getRows(); ++i) {
                for (size_t j = 0; j < mat.getCols(); ++j) {
                    std::cout << mat(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    };

    TEST_F(ReLULayerTest, ForwardPassTest) {
        // Create input
        size_t rows = 2, cols = 9;
        Matrix x = generate_random_matrix(rows, cols, -9, 9);

        // Create ReLULayer
        ReLULayer layer;

        // Perform forward pass
        Matrix result = layer.forward(x);

        // Print input and result
        std::cout << "Input:" << std::endl;
        print_matrix(x);
        std::cout << "Result:" << std::endl;
        print_matrix(result);

        // Check that all values are non-negative
        for (size_t i = 0; i < result.getRows(); ++i) {
            for (size_t j = 0; j < result.getCols(); ++j) {
                EXPECT_GE(result(i, j), 0.0f);
            }
        }

        // Check that positive values remain unchanged
        for (size_t i = 0; i < x.getRows(); ++i) {
            for (size_t j = 0; j < x.getCols(); ++j) {
                if (x(i, j) > 0) {
                    EXPECT_FLOAT_EQ(result(i, j), x(i, j));
                }
            }
        }
    }

} // namespace nnm

