#include <gtest/gtest.h>
#include "ReLULayer.h"
#include "Tensor4D.h"
#include <random>
#include <iostream>

namespace nnm {

    class ReLULayerTest : public ::testing::Test {
    protected:
        static Tensor4D
        generate_random_tensor(size_t batch_size, size_t channels, size_t height, size_t width, int min_val,
                               int max_val) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(min_val, max_val);

            Tensor4D tensor(batch_size, channels, height, width);
            for (size_t n = 0; n < batch_size; ++n) {
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t h = 0; h < height; ++h) {
                        for (size_t w = 0; w < width; ++w) {
                            tensor(n, c, h, w) = static_cast<float>(dis(gen));
                        }
                    }
                }
            }
            return tensor;
        }

        static void print_tensor(const Tensor4D &tensor) {
            for (size_t n = 0; n < tensor.getBatchSize(); ++n) {
                std::cout << "Batch " << n << ":" << std::endl;
                for (size_t c = 0; c < tensor.getChannels(); ++c) {
                    std::cout << "Channel " << c << ":" << std::endl;
                    for (size_t h = 0; h < tensor.getHeight(); ++h) {
                        for (size_t w = 0; w < tensor.getWidth(); ++w) {
                            std::cout << tensor(n, c, h, w) << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                }
            }
        }
    };

    TEST_F(ReLULayerTest, ForwardPassTest) {
        // Create input
        size_t batch_size = 2, channels = 3, height = 4, width = 5;
        Tensor4D x = generate_random_tensor(batch_size, channels, height, width, -9, 9);

        // Create ReLULayer
        ReLULayer layer;

        // Perform forward pass
        Tensor4D result = layer.forward(x);

        // Print input and result
        std::cout << "Input:" << std::endl;
        print_tensor(x);
        std::cout << "Result:" << std::endl;
        print_tensor(result);

        // Check that all values are non-negative
        for (size_t n = 0; n < result.getBatchSize(); ++n) {
            for (size_t c = 0; c < result.getChannels(); ++c) {
                for (size_t h = 0; h < result.getHeight(); ++h) {
                    for (size_t w = 0; w < result.getWidth(); ++w) {
                        EXPECT_GE(result(n, c, h, w), 0.0f);
                    }
                }
            }
        }

        // Check that positive values remain unchanged
        for (size_t n = 0; n < x.getBatchSize(); ++n) {
            for (size_t c = 0; c < x.getChannels(); ++c) {
                for (size_t h = 0; h < x.getHeight(); ++h) {
                    for (size_t w = 0; w < x.getWidth(); ++w) {
                        if (x(n, c, h, w) > 0) {
                            EXPECT_FLOAT_EQ(result(n, c, h, w), x(n, c, h, w));
                        }
                    }
                }
            }
        }
    }

} // namespace nnm