#include <gtest/gtest.h>
#include "FullyConnectedLayer.h"
#include "Tensor4D.h"
#include "Matrix.h"
#include "Vector.h"
#include <cmath>
#include <iostream>

namespace nnm {

    class FullyConnectedLayerTest : public ::testing::Test {
    protected:
        static constexpr float epsilon = 1e-2f;

        static float absolute_error(const Matrix &x, const Matrix &y) {
            if (x.getRows() != y.getRows() || x.getCols() != y.getCols()) {
                throw std::invalid_argument("Matrices must have the same dimensions");
            }
            float sum = 0.0f;
            for (size_t i = 0; i < x.getRows(); ++i) {
                for (size_t j = 0; j < x.getCols(); ++j) {
                    sum += std::abs(x(i, j) - y(i, j));
                }
            }
            return sum;
        }

        static Tensor4D create_input_tensor(size_t N, size_t C, size_t H, size_t W) {
            Tensor4D input(N, C, H, W);
            float val = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                for (size_t c = 0; c < C; ++c) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            input(n, c, h, w) = val;
                            val += 1.0f;
                        }
                    }
                }
            }
            return input;
        }
    };

    TEST_F(FullyConnectedLayerTest, ForwardPassTest) {
        // Create input
        size_t N = 2, C = 3, H = 4, W = 4;
        Tensor4D x = create_input_tensor(N, C, H, W);

        // Create FullyConnectedLayer
        size_t input_size = C * H * W;
        size_t output_size = 10;
        FullyConnectedLayer layer(input_size, output_size);

        // Set weights and biases for reproducibility
        Matrix weights(input_size, output_size);
        for (size_t i = 0; i < input_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                weights(i, j) = static_cast<float>(i * output_size + j) / (input_size * output_size);
            }
        }
        layer.set_weights(weights);

        Vector biases(output_size);
        for (size_t i = 0; i < output_size; ++i) {
            biases[i] = static_cast<float>(i) / output_size;
        }
        layer.set_biases(biases);

        // Perform forward pass
        Matrix out = layer.forward(x);

        // Check output shape
        EXPECT_EQ(out.getRows(), N);
        EXPECT_EQ(out.getCols(), output_size);

        // Calculate expected output
        Matrix expected_out(N, output_size);
        for (size_t n = 0; n < N; ++n) {
            for (size_t m = 0; m < output_size; ++m) {
                float sum = 0.0f;
                for (size_t i = 0; i < input_size; ++i) {
                    sum += x(n, i / (H * W), (i % (H * W)) / W, i % W) * weights(i, m);
                }
                expected_out(n, m) = sum + biases[m];
            }
        }

        // Compare output with expected output
        float error = absolute_error(out, expected_out);
        EXPECT_LT(error, epsilon);
    }


} // namespace nnm