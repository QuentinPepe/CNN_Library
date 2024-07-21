#include <gtest/gtest.h>
#include "LossFunctions.h"
#include "Matrix.h"
#include "Vector.h"
#include <cmath>

namespace nnm {

    class SoftmaxLossTest : public ::testing::Test {
    protected:
        static constexpr float epsilon = 1e-4f;

        static bool is_close(float a, float b, float eps = epsilon) {
            return std::abs(a - b) < eps;
        }

        static bool matrix_is_close(const Matrix &a, const Matrix &b, float eps = epsilon) {
            if (a.getRows() != b.getRows() || a.getCols() != b.getCols()) {
                return false;
            }
            for (size_t i = 0; i < a.getRows(); ++i) {
                for (size_t j = 0; j < a.getCols(); ++j) {
                    if (!is_close(a(i, j), b(i, j), eps)) {
                        return false;
                    }
                }
            }
            return true;
        }
    };

    TEST_F(SoftmaxLossTest, SoftmaxLossCalculation) {
        // Prepare input data
        Matrix x(2, 3);
        x(0, 0) = 1.0f;
        x(0, 1) = 2.0f;
        x(0, 2) = 3.0f;
        x(1, 0) = 4.0f;
        x(1, 1) = 5.0f;
        x(1, 2) = 6.0f;

        Vector y(2);
        y[0] = 0;
        y[1] = 2;

        // Calculate softmax loss
        auto result = LossFunctions::softmax_loss(x, y);


        // Check loss
        EXPECT_TRUE(is_close(result.loss, 1.4076058864593506f));

        // Check gradient
        Matrix expected_gradient(2, 3);
        expected_gradient(0, 0) = -0.4550f;
        expected_gradient(0, 1) = 0.1224f;
        expected_gradient(0, 2) = 0.3326f;
        expected_gradient(1, 0) = 0.0450f;
        expected_gradient(1, 1) = 0.1224f;
        expected_gradient(1, 2) = -0.1674f;

        EXPECT_TRUE(matrix_is_close(result.gradient, expected_gradient));

        // Matrix softmax_output = ...; // You need to modify your implementation to return this
        // Matrix expected_softmax(2, 3);
        // expected_softmax(0, 0) = 0.0900f; expected_softmax(0, 1) = 0.2447f; expected_softmax(0, 2) = 0.6652f;
        // expected_softmax(1, 0) = 0.0900f; expected_softmax(1, 1) = 0.2447f; expected_softmax(1, 2) = 0.6652f;
        // EXPECT_TRUE(matrix_is_close(softmax_output, expected_softmax));
    }

} // namespace nnm