#include <gtest/gtest.h>
#include "LossFunctions.h"
#include "Tensor4D.h"
#include <cmath>

namespace nnm {

    class SoftmaxLossTest : public ::testing::Test {
    protected:
        static constexpr float epsilon = 1e-4f;

        static bool is_close(float a, float b, float eps = epsilon) {
            return std::abs(a - b) < eps;
        }

        static bool tensor_is_close(const Tensor4D &a, const Tensor4D &b, float eps = epsilon) {
            if (a.getBatchSize() != b.getBatchSize() || a.getChannels() != b.getChannels() ||
                a.getHeight() != b.getHeight() || a.getWidth() != b.getWidth()) {
                return false;
            }
            for (size_t n = 0; n < a.getBatchSize(); ++n) {
                for (size_t c = 0; c < a.getChannels(); ++c) {
                    for (size_t h = 0; h < a.getHeight(); ++h) {
                        for (size_t w = 0; w < a.getWidth(); ++w) {
                            if (!is_close(a(n, c, h, w), b(n, c, h, w), eps)) {
                                return false;
                            }
                        }
                    }
                }
            }
            return true;
        }
    };

    TEST_F(SoftmaxLossTest, SoftmaxLossCalculation) {
        // Prepare input data
        Tensor4D x(2, 3, 1, 1);
        x(0, 0, 0, 0) = 1.0f;
        x(0, 1, 0, 0) = 2.0f;
        x(0, 2, 0, 0) = 3.0f;
        x(1, 0, 0, 0) = 4.0f;
        x(1, 1, 0, 0) = 5.0f;
        x(1, 2, 0, 0) = 6.0f;

        Tensor4D y(2, 1, 1, 1);
        y(0, 0, 0, 0) = 0;
        y(1, 0, 0, 0) = 2;

        // Calculate softmax loss
        auto result = LossFunctions::softmax_loss(x, y);

        // Check loss
        EXPECT_TRUE(is_close(result.loss, 1.4076058864593506f));

        // Check gradient
        Tensor4D expected_gradient(2, 3, 1, 1);
        expected_gradient(0, 0, 0, 0) = -0.4550f;
        expected_gradient(0, 1, 0, 0) = 0.1224f;
        expected_gradient(0, 2, 0, 0) = 0.3326f;
        expected_gradient(1, 0, 0, 0) = 0.0450f;
        expected_gradient(1, 1, 0, 0) = 0.1224f;
        expected_gradient(1, 2, 0, 0) = -0.1674f;

        EXPECT_TRUE(tensor_is_close(result.gradient, expected_gradient));

        // Tensor4D softmax_output = ...; // You need to modify your implementation to return this
        // Tensor4D expected_softmax(2, 3, 1, 1);
        // expected_softmax(0, 0, 0, 0) = 0.0900f; expected_softmax(0, 1, 0, 0) = 0.2447f; expected_softmax(0, 2, 0, 0) = 0.6652f;
        // expected_softmax(1, 0, 0, 0) = 0.0900f; expected_softmax(1, 1, 0, 0) = 0.2447f; expected_softmax(1, 2, 0, 0) = 0.6652f;
        // EXPECT_TRUE(tensor_is_close(softmax_output, expected_softmax));
    }

} // namespace nnm