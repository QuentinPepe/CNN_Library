#include <gtest/gtest.h>
#include "ConvolutionalLayer.h"
#include "Matrix.h"
#include "Vector.h"
#include "Tensor4D.h"
#include <cmath>
#include <iostream>

namespace nnm {

    class ConvolutionalLayerTest : public ::testing::Test {
    protected:
        static constexpr float epsilon = 1e-5f;

        static float absolute_error(const Tensor4D &x, const Tensor4D &y) {
            if (x.getBatchSize() != y.getBatchSize() || x.getChannels() != y.getChannels() ||
                x.getHeight() != y.getHeight() || x.getWidth() != y.getWidth()) {
                throw std::invalid_argument("Tensors must have the same dimensions");
            }
            float sum = 0.0f;
            for (size_t n = 0; n < x.getBatchSize(); ++n) {
                for (size_t c = 0; c < x.getChannels(); ++c) {
                    for (size_t h = 0; h < x.getHeight(); ++h) {
                        for (size_t w = 0; w < x.getWidth(); ++w) {
                            sum += std::abs(x(n, c, h, w) - y(n, c, h, w));
                        }
                    }
                }
            }
            return sum;
        }

        static Tensor4D linspace_tensor4d(float start, float end, const std::vector<size_t> &shape) {
            size_t total_elements = shape[0] * shape[1] * shape[2] * shape[3];
            Tensor4D result(shape[0], shape[1], shape[2], shape[3]);
            float step = (end - start) / (total_elements - 1);
            size_t index = 0;
            for (size_t n = 0; n < shape[0]; ++n) {
                for (size_t c = 0; c < shape[1]; ++c) {
                    for (size_t h = 0; h < shape[2]; ++h) {
                        for (size_t w = 0; w < shape[3]; ++w) {
                            result(n, c, h, w) = start + index * step;
                            ++index;
                        }
                    }
                }
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
        // Define shapes for input data, weights and biases
        std::vector<size_t> x_shape = {1, 3, 4, 4};
        std::vector<size_t> w_shape = {3, 3, 4, 4};
        size_t b_shape = 3;

        // Generate data
        Tensor4D x = linspace_tensor4d(0, 255, x_shape);
        Tensor4D w = linspace_tensor4d(-1.0, 1.0, w_shape);
        Vector b = linspace_vector(-1.0, 1.0, b_shape);

        // Print input data for debugging
        std::cout << "Input x:" << std::endl;
        x.print();
        std::cout << "Weights w:" << std::endl;
        w.print();
        std::cout << "Bias b:" << std::endl;
        for (size_t i = 0; i < b.size(); ++i) {
            std::cout << b[i] << " ";
        }
        std::cout << std::endl;

        // Create ConvolutionalLayer
        ConvolutionalLayer layer(3, 3, 4, 2, 1);
        layer.set_weights(w);
        layer.set_bias(b);

        // Perform forward pass
        Tensor4D out = layer.forward(x);

        // Print output for debugging
        std::cout << "Output:" << std::endl;
        out.print();

        // Define correct output
        Tensor4D correct_out(1, 3, 2, 2);
        correct_out(0, 0, 0, 0) = -1577.82517483f;
        correct_out(0, 0, 0, 1) = -1715.03496503f;
        correct_out(0, 0, 1, 0) = -2154.29370629f;
        correct_out(0, 0, 1, 1) = -2308.0979021f;
        correct_out(0, 1, 0, 0) = 480.12587413f;
        correct_out(0, 1, 0, 1) = 440.25874126f;
        correct_out(0, 1, 1, 0) = 296.38461538f;
        correct_out(0, 1, 1, 1) = 240.59440559f;
        correct_out(0, 2, 0, 0) = 2538.07692308f;
        correct_out(0, 2, 0, 1) = 2595.55244755f;
        correct_out(0, 2, 1, 0) = 2747.06293706f;
        correct_out(0, 2, 1, 1) = 2789.28671329f;

        // Print correct output for debugging
        std::cout << "Correct output:" << std::endl;
        correct_out.print();

        // Check output shape
        EXPECT_EQ(out.getBatchSize(), 1);
        EXPECT_EQ(out.getChannels(), 3);
        EXPECT_EQ(out.getHeight(), 2);
        EXPECT_EQ(out.getWidth(), 2);

        // Calculate difference
        float error = absolute_error(correct_out, out);
        std::cout << "Error: " << error << std::endl;
        EXPECT_LT(error, epsilon);
    }

} // namespace nnm