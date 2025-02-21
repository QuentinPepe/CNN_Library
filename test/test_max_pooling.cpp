#include <gtest/gtest.h>
#include "MaxPoolingLayer.h"
#include "Vector.h"
#include "Tensor4D.h"
#include <cmath>
#include <iostream>

namespace nnm {

    class MaxPoolingLayerTest : public ::testing::Test {
    protected:
        static constexpr float epsilon = 1e-3f;

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
    };

    TEST_F(MaxPoolingLayerTest, ForwardPassTest) {
        // Define shapes for input data
        std::vector<size_t> x_shape = {2, 1, 4, 4};

        // Generate data
        Tensor4D x = linspace_tensor4d(0, 255, x_shape);

        // Print input data for debugging
        std::cout << "Input x:" << std::endl;
        x.print();

        // Create MaxPoolingLayer
        MaxPoolingLayer layer(2, 2, 2);

        // Perform forward pass
        Tensor4D out = layer.forward(x);

        // Print output for debugging
        std::cout << "Output:" << std::endl;
        out.print();

        // Define correct output
        Tensor4D correct_out(2, 1, 2, 2);
        correct_out(0, 0, 0, 0) = 41.12903226f;
        correct_out(0, 0, 0, 1) = 57.58064516f;
        correct_out(0, 0, 1, 0) = 106.93548387f;
        correct_out(0, 0, 1, 1) = 123.38709677f;
        correct_out(1, 0, 0, 0) = 172.74193548f;
        correct_out(1, 0, 0, 1) = 189.19354839f;
        correct_out(1, 0, 1, 0) = 238.5483871f;
        correct_out(1, 0, 1, 1) = 255.0f;

        std::cout << "Correct output:" << std::endl;
        correct_out.print();

        // Check output shape
        EXPECT_EQ(out.getBatchSize(), 2);
        EXPECT_EQ(out.getChannels(), 1);
        EXPECT_EQ(out.getHeight(), 2);
        EXPECT_EQ(out.getWidth(), 2);

        // Calculate difference
        float error = absolute_error(correct_out, out);
        std::cout << "Error: " << error << std::endl;
        EXPECT_LT(error, epsilon);
    }

} // namespace nnm