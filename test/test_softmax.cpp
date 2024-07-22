#include <gtest/gtest.h>
#include "Tensor4D.h"
#include "SoftMaxLayer.h"

#include <cmath>

namespace {

    bool are_close(float a, float b, float epsilon = 1e-3) {
        return std::fabs(a - b) < epsilon;
    }

    bool compareTensors(const nnm::Tensor4D &a, const nnm::Tensor4D &b, float epsilon = 1e-3) {
        if (a.getBatchSize() != b.getBatchSize() ||
            a.getChannels() != b.getChannels() ||
            a.getHeight() != b.getHeight() ||
            a.getWidth() != b.getWidth()) {
            std::cerr << "Tensors have different dimensions" << std::endl;
            std::cerr << "a: " << a.getBatchSize() << " " << a.getChannels() << " " << a.getHeight() << " "
                      << a.getWidth() << std::endl;
            std::cerr << "b: " << b.getBatchSize() << " " << b.getChannels() << " " << b.getHeight() << " "
                      << b.getWidth() << std::endl;
            return false;
        }

        for (size_t n = 0; n < a.getBatchSize(); ++n) {
            for (size_t c = 0; c < a.getChannels(); ++c) {
                for (size_t h = 0; h < a.getHeight(); ++h) {
                    for (size_t w = 0; w < a.getWidth(); ++w) {
                        if (std::abs(a(n, c, h, w) - b(n, c, h, w)) > epsilon) {
                            std::cerr << "Difference at (" << n << ", " << c << ", " << h << ", " << w << "): "
                                      << a(n, c, h, w) << " != " << b(n, c, h, w) << std::endl;
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    TEST(SoftMaxLayerTest, ChannelSoftMax) {
        nnm::Tensor4D input(1, 2, 2, 2);
        input(0, 0, 0, 0) = 1.0f;
        input(0, 0, 0, 1) = 2.0f;
        input(0, 0, 1, 0) = 3.0f;
        input(0, 0, 1, 1) = 4.0f;
        input(0, 1, 0, 0) = 5.0f;
        input(0, 1, 0, 1) = 6.0f;
        input(0, 1, 1, 0) = 7.0f;
        input(0, 1, 1, 1) = 8.0f;

        nnm::SoftMaxLayer softmax(1);  // 2 canaux, axe 1
        nnm::Tensor4D output = softmax.forward(input);

        EXPECT_TRUE(are_close(output(0, 0, 0, 0), 0.01798621f));
        EXPECT_TRUE(are_close(output(0, 0, 0, 1), 0.01798621f));
        EXPECT_TRUE(are_close(output(0, 0, 1, 0), 0.01798621f));
        EXPECT_TRUE(are_close(output(0, 0, 1, 1), 0.01798621f));
        EXPECT_TRUE(are_close(output(0, 1, 0, 0), 0.98201376f));
        EXPECT_TRUE(are_close(output(0, 1, 0, 1), 0.98201376f));
        EXPECT_TRUE(are_close(output(0, 1, 1, 0), 0.98201376f));
        EXPECT_TRUE(are_close(output(0, 1, 1, 1), 0.98201376f));
    }

    TEST(SoftMaxLayerTest, HeightSoftMax) {
        nnm::Tensor4D input(1, 1, 3, 3);
        input(0, 0, 0, 0) = 0.1f;
        input(0, 0, 0, 1) = 0.2f;
        input(0, 0, 0, 2) = 0.3f;
        input(0, 0, 1, 0) = 0.4f;
        input(0, 0, 1, 1) = 0.5f;
        input(0, 0, 1, 2) = 0.6f;
        input(0, 0, 2, 0) = 0.7f;
        input(0, 0, 2, 1) = 0.8f;
        input(0, 0, 2, 2) = 0.9f;

        nnm::SoftMaxLayer softmax(2);  // hauteur 3, axe 2
        nnm::Tensor4D output = softmax.forward(input);

        EXPECT_TRUE(are_close(output(0, 0, 0, 0), 0.23969449f));
        EXPECT_TRUE(are_close(output(0, 0, 0, 1), 0.23969446f));
        EXPECT_TRUE(are_close(output(0, 0, 0, 2), 0.23969449f));
        EXPECT_TRUE(are_close(output(0, 0, 1, 0), 0.3235537f));
        EXPECT_TRUE(are_close(output(0, 0, 1, 1), 0.32355368f));
        EXPECT_TRUE(are_close(output(0, 0, 1, 2), 0.3235537f));
        EXPECT_TRUE(are_close(output(0, 0, 2, 0), 0.4367518f));
        EXPECT_TRUE(are_close(output(0, 0, 2, 1), 0.4367518f));
        EXPECT_TRUE(are_close(output(0, 0, 2, 2), 0.4367518f));
    }

    TEST(SoftMaxLayerTest, WidthSoftMax) {
        nnm::Tensor4D input(1, 2, 2, 3);
        input(0, 0, 0, 0) = 1.0f;
        input(0, 0, 0, 1) = 2.0f;
        input(0, 0, 0, 2) = 3.0f;
        input(0, 0, 1, 0) = 4.0f;
        input(0, 0, 1, 1) = 5.0f;
        input(0, 0, 1, 2) = 6.0f;
        input(0, 1, 0, 0) = 7.0f;
        input(0, 1, 0, 1) = 8.0f;
        input(0, 1, 0, 2) = 9.0f;
        input(0, 1, 1, 0) = 10.0f;
        input(0, 1, 1, 1) = 11.0f;
        input(0, 1, 1, 2) = 12.0f;

        nnm::SoftMaxLayer softmax(3);  // largeur 3, axe 3
        nnm::Tensor4D output = softmax.forward(input);

        EXPECT_TRUE(are_close(output(0, 0, 0, 0), 0.09003057f));
        EXPECT_TRUE(are_close(output(0, 0, 0, 1), 0.24472848f));
        EXPECT_TRUE(are_close(output(0, 0, 0, 2), 0.66524094f));
        EXPECT_TRUE(are_close(output(0, 0, 1, 0), 0.09003057f));
        EXPECT_TRUE(are_close(output(0, 0, 1, 1), 0.24472848f));
        EXPECT_TRUE(are_close(output(0, 0, 1, 2), 0.66524094f));
        EXPECT_TRUE(are_close(output(0, 1, 0, 0), 0.09003057f));
        EXPECT_TRUE(are_close(output(0, 1, 0, 1), 0.24472848f));
        EXPECT_TRUE(are_close(output(0, 1, 0, 2), 0.66524094f));
        EXPECT_TRUE(are_close(output(0, 1, 1, 0), 0.09003057f));
        EXPECT_TRUE(are_close(output(0, 1, 1, 1), 0.24472848f));
        EXPECT_TRUE(are_close(output(0, 1, 1, 2), 0.66524094f));
    }

    TEST(SoftMaxLayerTest, Forward) {
        nnm::Tensor4D input(1, 10, 1, 1);
        std::vector<float> input_data = {0.0926, -0.0659, -0.1545, 0.1032, 0.0625, -0.2349, -0.1428, 0.1613, -0.0603,
                                         0.1443};
        for (size_t i = 0; i < input_data.size(); ++i) {
            input(0, i, 0, 0) = input_data[i];
        }
        nnm::Tensor4D expected_output(1, 10, 1, 1);
        std::vector<float> expected_output_data = {0.1098, 0.0937, 0.0857, 0.1110, 0.1065, 0.0791, 0.0868, 0.1176,
                                                   0.0942, 0.1156};
        for (size_t i = 0; i < expected_output_data.size(); ++i) {
            expected_output(0, i, 0, 0) = expected_output_data[i];
        }


        nnm::SoftMaxLayer softmax_layer(1);

        nnm::Tensor4D output = softmax_layer.forward(input);

        ASSERT_TRUE(compareTensors(output, expected_output));
    }

}  // namespace