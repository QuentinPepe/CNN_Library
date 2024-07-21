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

    TEST_F(ReLULayerTest, ForwardPass) {
        // Define input tensor
        nnm::Tensor4D input_tensor({
                                           {{{0.913767f, -0.006101f, 0.624794f, -0.243688f, 0.959681f, 0.982249f,
                                              -0.949719f},
                                             {0.089510f, 0.305282f, -0.759948f, -0.311146f, -0.498423f, 0.173939f,
                                              0.945470f},
                                             {-0.819082f, 0.082511f, 0.578858f, -0.243622f, 0.060011f, -0.611561f,
                                              0.337316f},
                                             {0.520352f, -0.990606f, -0.456336f, -0.546144f, 0.420601f, 0.395104f,
                                              -0.008163f},
                                             {0.258821f, 0.061651f, 0.758223f, 0.536159f, -0.655659f, 0.749548f,
                                              -0.206629f},
                                             {0.744870f, -0.886590f, -0.755991f, 0.129045f, 0.162923f, 0.119927f,
                                              -0.750695f}},
                                            {{-0.565771f, -0.962615f, -0.267785f, -0.310989f, -0.609022f, 0.237199f,
                                              -0.219713f},
                                             {0.189435f, 0.642495f, 0.495770f, -0.453646f, 0.738897f, -0.402099f,
                                              0.549405f},
                                             {0.066590f, 0.480355f, -0.203262f, 0.203656f, 0.887285f, -0.864264f,
                                              -0.405341f},
                                             {-0.955232f, -0.606676f, -0.756362f, 0.204503f, -0.830870f, -0.132958f,
                                              0.311410f},
                                             {-0.212348f, 0.320245f, -0.400694f, 0.089150f, 0.268854f, 0.443670f,
                                              0.519724f},
                                             {-0.985899f, 0.916421f, -0.202469f, 0.636974f, 0.011717f, 0.324403f,
                                              0.556153f}},
                                            {{0.632877f, 0.690361f, -0.734720f, 0.149826f, 0.617098f, -0.865979f,
                                              -0.983733f},
                                             {-0.298994f, 0.458905f, 0.086055f, 0.941038f, -0.355628f, -0.359102f,
                                              -0.713791f},
                                             {0.964135f, -0.550287f, -0.667809f, 0.151791f, -0.028399f, -0.683071f,
                                              -0.346847f},
                                             {-0.723539f, 0.388632f, 0.372277f, 0.224717f, 0.272561f, -0.469829f,
                                              -0.154419f},
                                             {-0.727466f, 0.673557f, -0.866230f, -0.151751f, 0.808193f, 0.948260f,
                                              -0.889715f},
                                             {0.509174f, -0.472452f, 0.177689f, -0.701257f, -0.432378f, -0.300093f,
                                              -0.149027f}}}
                                   });

// Expected output tensor
        nnm::Tensor4D expected_output({
                                              {{{0.913767f, 0.000000f, 0.624794f, 0.000000f, 0.959681f, 0.982249f,
                                                 0.000000f},
                                                {0.089510f, 0.305282f, 0.000000f, 0.000000f, 0.000000f, 0.173939f,
                                                 0.945470f},
                                                {0.000000f, 0.082511f, 0.578858f, 0.000000f, 0.060011f, 0.000000f,
                                                 0.337316f},
                                                {0.520352f, 0.000000f, 0.000000f, 0.000000f, 0.420601f, 0.395104f,
                                                 0.000000f},
                                                {0.258821f, 0.061651f, 0.758223f, 0.536159f, 0.000000f, 0.749548f,
                                                 0.000000f},
                                                {0.744870f, 0.000000f, 0.000000f, 0.129045f, 0.162923f, 0.119927f,
                                                 0.000000f}},
                                               {{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.237199f,
                                                 0.000000f},
                                                {0.189435f, 0.642495f, 0.495770f, 0.000000f, 0.738897f, 0.000000f,
                                                 0.549405f},
                                                {0.066590f, 0.480355f, 0.000000f, 0.203656f, 0.887285f, 0.000000f,
                                                 0.000000f},
                                                {0.000000f, 0.000000f, 0.000000f, 0.204503f, 0.000000f, 0.000000f,
                                                 0.311410f},
                                                {0.000000f, 0.320245f, 0.000000f, 0.089150f, 0.268854f, 0.443670f,
                                                 0.519724f},
                                                {0.000000f, 0.916421f, 0.000000f, 0.636974f, 0.011717f, 0.324403f,
                                                 0.556153f}},
                                               {{0.632877f, 0.690361f, 0.000000f, 0.149826f, 0.617098f, 0.000000f,
                                                 0.000000f},
                                                {0.000000f, 0.458905f, 0.086055f, 0.941038f, 0.000000f, 0.000000f,
                                                 0.000000f},
                                                {0.964135f, 0.000000f, 0.000000f, 0.151791f, 0.000000f, 0.000000f,
                                                 0.000000f},
                                                {0.000000f, 0.388632f, 0.372277f, 0.224717f, 0.272561f, 0.000000f,
                                                 0.000000f},
                                                {0.000000f, 0.673557f, 0.000000f, 0.000000f, 0.808193f, 0.948260f,
                                                 0.000000f},
                                                {0.509174f, 0.000000f, 0.177689f, 0.000000f, 0.000000f, 0.000000f,
                                                 0.000000f}}}
                                      });

        // Create ReLULayer
        nnm::ReLULayer relu_layer;

        // Perform forward pass
        nnm::Tensor4D output = relu_layer.forward(input_tensor);

        // Compare output with expected output
        ASSERT_EQ(output.getBatchSize(), expected_output.getBatchSize());
        ASSERT_EQ(output.getChannels(), expected_output.getChannels());
        ASSERT_EQ(output.getHeight(), expected_output.getHeight());
        ASSERT_EQ(output.getWidth(), expected_output.getWidth());

        for (size_t n = 0; n < output.getBatchSize(); ++n) {
            for (size_t c = 0; c < output.getChannels(); ++c) {
                for (size_t h = 0; h < output.getHeight(); ++h) {
                    for (size_t w = 0; w < output.getWidth(); ++w) {
                        EXPECT_NEAR(output(n, c, h, w), expected_output(n, c, h, w), 1e-6)
                                            << "Mismatch at position (" << n << ", " << c << ", " << h << ", " << w
                                            << ")";
                    }
                }
            }
        }
    }

} // namespace nnm