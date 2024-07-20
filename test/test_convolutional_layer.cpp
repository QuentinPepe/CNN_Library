#include <gtest/gtest.h>
#include "ConvolutionalLayer.h"
#include "Matrix.h"
#include "Vector.h"
#include <cmath>
#include <iostream>

namespace nnm {

    class ConvolutionalLayerTest : public ::testing::Test {
    protected:
        static constexpr float epsilon = 1e-5f;

        static float absolute_error(const Matrix &x, const Matrix &y) {
            if (x.getRows() != y.getRows() || x.getCols() != y.getCols()) {
                throw std::invalid_argument("Matrices must have the same dimensions: x(" +
                                            std::to_string(x.getRows()) + "," + std::to_string(x.getCols()) +
                                            ") y(" + std::to_string(y.getRows()) + "," + std::to_string(y.getCols()) +
                                            ")");
            }
            float sum = 0.0f;
            for (size_t i = 0; i < x.getRows(); ++i) {
                for (size_t j = 0; j < x.getCols(); ++j) {
                    sum += std::abs(x(i, j) - y(i, j));
                }
            }
            return sum;
        }

        static Matrix linspace_matrix(float start, float end, size_t num) {
            Matrix result(1, num);
            float step = (end - start) / (num - 1);
            for (size_t i = 0; i < num; ++i) {
                result(0, i) = start + i * step;
            }
            return result;
        }

        static Vector linspace_vector(float start, float end, size_t num) {
            Vector result(num);
            float step = (end - start) / (num - 1);
            for (size_t i = 0; i < num; ++i) {
                result[i] = start + i * step;
            }
            return result;
        }
    };

    TEST_F(ConvolutionalLayerTest, ForwardPassTest) {
        /*
        // Define shapes
        const size_t N = 1, C = 3, H = 4, W = 4;
        const size_t F = 3, HH = 4, WW = 4;

        // Generate input data
        Matrix x = linspace_matrix(0, 255, N * C * H * W);
        x.reshape(N, C * H * W);
        std::cout << "Input shape: " << x.getRows() << "x" << x.getCols() << std::endl;

        // Generate weights
        Matrix w = linspace_matrix(-1.0f, 1.0f, F * C * HH * WW);
        w.reshape(F, C * HH * WW);
        std::cout << "Weights shape: " << w.getRows() << "x" << w.getCols() << std::endl;

        // Generate bias
        Vector b = linspace_vector(-1.0f, 1.0f, F);
        std::cout << "Bias size: " << b.size() << std::endl;

        // Create convolutional layer
        ConvolutionalLayer conv_layer(C, F, HH, 2, 1);  // stride = 2, padding = 1
        conv_layer.set_weights(w);
        conv_layer.set_bias(b);

        // Perform forward pass
        Matrix out = conv_layer.forward(x);
        std::cout << "Output shape: " << out.getRows() << "x" << out.getCols() << std::endl;

        // Calculate the expected output shape
        size_t H_out = (H + 2 * conv_layer.get_padding() - conv_layer.get_kernel_size()) / conv_layer.get_stride() + 1;
        size_t W_out = (W + 2 * conv_layer.get_padding() - conv_layer.get_kernel_size()) / conv_layer.get_stride() + 1;
        size_t expected_elements = N * F * H_out * W_out;
        std::cout << "Expected elements: " << expected_elements << std::endl;

        // Check if the output has the correct number of elements
        ASSERT_EQ(out.getRows() * out.getCols(), expected_elements);

        // Define expected output
        Matrix correct_out(1, F * H_out * W_out);
        correct_out(0, 0) = -1577.82517483f;
        correct_out(0, 1) = -1715.03496503f;
        correct_out(0, 2) = -2154.29370629f;
        correct_out(0, 3) = -2308.0979021f;
        correct_out(0, 4) = 480.12587413f;
        correct_out(0, 5) = 440.25874126f;
        correct_out(0, 6) = 296.38461538f;
        correct_out(0, 7) = 240.59440559f;
        correct_out(0, 8) = 2538.07692308f;
        correct_out(0, 9) = 2595.55244755f;
        correct_out(0, 10) = 2747.06293706f;
        correct_out(0, 11) = 2789.28671329f;

        // Calculate absolute error
        float error = absolute_error(correct_out, out);

        // Check if the error is within the acceptable range
        EXPECT_LT(error, epsilon);

        // Print the first few elements of both matrices for comparison
        std::cout << "First few elements of correct_out:" << std::endl;
        for (int i = 0; i < std::min(5, static_cast<int>(correct_out.getCols())); ++i) {
            std::cout << correct_out(0, i) << " ";
        }
        std::cout << std::endl;

        std::cout << "First few elements of out:" << std::endl;
        for (int i = 0; i < std::min(5, static_cast<int>(out.getCols())); ++i) {
            std::cout << out(0, i) << " ";
        }
        std::cout << std::endl;*/
    }

} // namespace nnm