#include <gtest/gtest.h>
#include "Tensor4D.h"
#include "Matrix.h"
#include "FlattenLayer.h"
#include <cmath>

namespace nnm {

    class FlattenTest : public ::testing::Test {
    protected:
        static constexpr float epsilon = 1e-6f;

        static bool is_close(float a, float b, float eps = epsilon) {
            return std::abs(a - b) < eps;
        }

        static Tensor4D create_input_tensor() {
            Tensor4D input(2, 3, 4, 4);
            input(0, 0, 0, 0) = 0.791865f;
            input(0, 0, 0, 1) = -0.436437f;
            input(0, 0, 0, 2) = -1.205499f;
            input(0, 0, 0, 3) = -1.637622f;
            input(0, 0, 1, 0) = -0.085604f;
            input(0, 0, 1, 1) = -0.340940f;
            input(0, 0, 1, 2) = 0.192264f;
            input(0, 0, 1, 3) = 0.463124f;
            input(0, 0, 2, 0) = -0.106627f;
            input(0, 0, 2, 1) = 1.303800f;
            input(0, 0, 2, 2) = 0.041493f;
            input(0, 0, 2, 3) = 1.508355f;
            input(0, 0, 3, 0) = -0.811528f;
            input(0, 0, 3, 1) = -0.886798f;
            input(0, 0, 3, 2) = 1.874877f;
            input(0, 0, 3, 3) = -0.849095f;
            input(0, 1, 0, 0) = -0.285756f;
            input(0, 1, 0, 1) = 0.068620f;
            input(0, 1, 0, 2) = -0.714663f;
            input(0, 1, 0, 3) = 0.866069f;
            input(0, 1, 1, 0) = -0.665780f;
            input(0, 1, 1, 1) = -1.576251f;
            input(0, 1, 1, 2) = -0.633278f;
            input(0, 1, 1, 3) = -1.776745f;
            input(0, 1, 2, 0) = 1.086935f;
            input(0, 1, 2, 1) = 0.964728f;
            input(0, 1, 2, 2) = 0.914117f;
            input(0, 1, 2, 3) = 1.737042f;
            input(0, 1, 3, 0) = -2.977160f;
            input(0, 1, 3, 1) = -0.484863f;
            input(0, 1, 3, 2) = 1.170451f;
            input(0, 1, 3, 3) = 0.586708f;
            input(0, 2, 0, 0) = -0.820451f;
            input(0, 2, 0, 1) = -1.616761f;
            input(0, 2, 0, 2) = 2.410156f;
            input(0, 2, 0, 3) = 0.490265f;
            input(0, 2, 1, 0) = -0.560522f;
            input(0, 2, 1, 1) = 0.793999f;
            input(0, 2, 1, 2) = -0.071875f;
            input(0, 2, 1, 3) = -0.867390f;
            input(0, 2, 2, 0) = -0.820906f;
            input(0, 2, 2, 1) = -1.633472f;
            input(0, 2, 2, 2) = -0.202649f;
            input(0, 2, 2, 3) = 0.064397f;
            input(0, 2, 3, 0) = 1.408932f;
            input(0, 2, 3, 1) = 1.316002f;
            input(0, 2, 3, 2) = 1.531291f;
            input(0, 2, 3, 3) = -0.615076f;
            input(1, 0, 0, 0) = -0.206175f;
            input(1, 0, 0, 1) = -0.173385f;
            input(1, 0, 0, 2) = 0.947787f;
            input(1, 0, 0, 3) = 1.937234f;
            input(1, 0, 1, 0) = 0.615880f;
            input(1, 0, 1, 1) = -1.227051f;
            input(1, 0, 1, 2) = -0.743896f;
            input(1, 0, 1, 3) = -0.010007f;
            input(1, 0, 2, 0) = 1.796220f;
            input(1, 0, 2, 1) = 0.479491f;
            input(1, 0, 2, 2) = 1.609346f;
            input(1, 0, 2, 3) = 1.113868f;
            input(1, 0, 3, 0) = -0.453360f;
            input(1, 0, 3, 1) = 0.514068f;
            input(1, 0, 3, 2) = 0.064999f;
            input(1, 0, 3, 3) = 1.366845f;
            input(1, 1, 0, 0) = -1.193340f;
            input(1, 1, 0, 1) = -0.282879f;
            input(1, 1, 0, 2) = -0.561044f;
            input(1, 1, 0, 3) = 0.358160f;
            input(1, 1, 1, 0) = 0.386496f;
            input(1, 1, 1, 1) = 2.400057f;
            input(1, 1, 1, 2) = -0.468238f;
            input(1, 1, 1, 3) = -1.025542f;
            input(1, 1, 2, 0) = 1.533068f;
            input(1, 1, 2, 1) = -0.633326f;
            input(1, 1, 2, 2) = -0.626271f;
            input(1, 1, 2, 3) = 0.322643f;
            input(1, 1, 3, 0) = -0.128170f;
            input(1, 1, 3, 1) = 0.029163f;
            input(1, 1, 3, 2) = -0.150362f;
            input(1, 1, 3, 3) = 0.275520f;
            input(1, 2, 0, 0) = -0.642941f;
            input(1, 2, 0, 1) = -1.900071f;
            input(1, 2, 0, 2) = 2.129185f;
            input(1, 2, 0, 3) = -1.216341f;
            input(1, 2, 1, 0) = -0.719883f;
            input(1, 2, 1, 1) = 1.795957f;
            input(1, 2, 1, 2) = 0.206365f;
            input(1, 2, 1, 3) = 0.858166f;
            input(1, 2, 2, 0) = 0.745697f;
            input(1, 2, 2, 1) = -0.253145f;
            input(1, 2, 2, 2) = -0.756790f;
            input(1, 2, 2, 3) = -0.431207f;
            input(1, 2, 3, 0) = 0.512952f;
            input(1, 2, 3, 1) = 0.938986f;
            input(1, 2, 3, 2) = 1.822877f;
            input(1, 2, 3, 3) = -1.022640f;

            return input;
        }
    };

    TEST_F(FlattenTest, ForwardPass) {
        // Create Flatten layer
        Flatten flatten(1, -1);

        // Create input tensor
        Tensor4D input = create_input_tensor();

        // Perform forward pass
        Tensor4D output = flatten.forward(input);

        Tensor4D expected_output(1, 2, 48, 1);
        expected_output(0, 0) = 0.791865f;
        expected_output(0, 1) = -0.436437f;
        expected_output(0, 2) = -1.205499f;
        expected_output(0, 3) = -1.637622f;
        expected_output(0, 4) = -0.085604f;
        expected_output(0, 5) = -0.340940f;
        expected_output(0, 6) = 0.192264f;
        expected_output(0, 7) = 0.463124f;
        expected_output(0, 8) = -0.106627f;
        expected_output(0, 9) = 1.303800f;
        expected_output(0, 10) = 0.041493f;
        expected_output(0, 11) = 1.508355f;
        expected_output(0, 12) = -0.811528f;
        expected_output(0, 13) = -0.886798f;
        expected_output(0, 14) = 1.874877f;
        expected_output(0, 15) = -0.849095f;
        expected_output(0, 16) = -0.285756f;
        expected_output(0, 17) = 0.068620f;
        expected_output(0, 18) = -0.714663f;
        expected_output(0, 19) = 0.866069f;
        expected_output(0, 20) = -0.665780f;
        expected_output(0, 21) = -1.576251f;
        expected_output(0, 22) = -0.633278f;
        expected_output(0, 23) = -1.776745f;
        expected_output(0, 24) = 1.086935f;
        expected_output(0, 25) = 0.964728f;
        expected_output(0, 26) = 0.914117f;
        expected_output(0, 27) = 1.737042f;
        expected_output(0, 28) = -2.977160f;
        expected_output(0, 29) = -0.484863f;
        expected_output(0, 30) = 1.170451f;
        expected_output(0, 31) = 0.586708f;
        expected_output(0, 32) = -0.820451f;
        expected_output(0, 33) = -1.616761f;
        expected_output(0, 34) = 2.410156f;
        expected_output(0, 35) = 0.490265f;
        expected_output(0, 36) = -0.560522f;
        expected_output(0, 37) = 0.793999f;
        expected_output(0, 38) = -0.071875f;
        expected_output(0, 39) = -0.867390f;
        expected_output(0, 40) = -0.820906f;
        expected_output(0, 41) = -1.633472f;
        expected_output(0, 42) = -0.202649f;
        expected_output(0, 43) = 0.064397f;
        expected_output(0, 44) = 1.408932f;
        expected_output(0, 45) = 1.316002f;
        expected_output(0, 46) = 1.531291f;
        expected_output(0, 47) = -0.615076f;
        expected_output(1, 0) = -0.206175f;
        expected_output(1, 1) = -0.173385f;
        expected_output(1, 2) = 0.947787f;
        expected_output(1, 3) = 1.937234f;
        expected_output(1, 4) = 0.615880f;
        expected_output(1, 5) = -1.227051f;
        expected_output(1, 6) = -0.743896f;
        expected_output(1, 7) = -0.010007f;
        expected_output(1, 8) = 1.796220f;
        expected_output(1, 9) = 0.479491f;
        expected_output(1, 10) = 1.609346f;
        expected_output(1, 11) = 1.113868f;
        expected_output(1, 12) = -0.453360f;
        expected_output(1, 13) = 0.514068f;
        expected_output(1, 14) = 0.064999f;
        expected_output(1, 15) = 1.366845f;
        expected_output(1, 16) = -1.193340f;
        expected_output(1, 17) = -0.282879f;
        expected_output(1, 18) = -0.561044f;
        expected_output(1, 19) = 0.358160f;
        expected_output(1, 20) = 0.386496f;
        expected_output(1, 21) = 2.400057f;
        expected_output(1, 22) = -0.468238f;
        expected_output(1, 23) = -1.025542f;
        expected_output(1, 24) = 1.533068f;
        expected_output(1, 25) = -0.633326f;
        expected_output(1, 26) = -0.626271f;
        expected_output(1, 27) = 0.322643f;
        expected_output(1, 28) = -0.128170f;
        expected_output(1, 29) = 0.029163f;
        expected_output(1, 30) = -0.150362f;
        expected_output(1, 31) = 0.275520f;
        expected_output(1, 32) = -0.642941f;
        expected_output(1, 33) = -1.900071f;
        expected_output(1, 34) = 2.129185f;
        expected_output(1, 35) = -1.216341f;
        expected_output(1, 36) = -0.719883f;
        expected_output(1, 37) = 1.795957f;
        expected_output(1, 38) = 0.206365f;
        expected_output(1, 39) = 0.858166f;
        expected_output(1, 40) = 0.745697f;
        expected_output(1, 41) = -0.253145f;
        expected_output(1, 42) = -0.756790f;
        expected_output(1, 43) = -0.431207f;
        expected_output(1, 44) = 0.512952f;
        expected_output(1, 45) = 0.938986f;
        expected_output(1, 46) = 1.822877f;
        expected_output(1, 47) = -1.022640f;

        // Check output dimensions
        EXPECT_EQ(output.getBatchSize(), expected_output.getBatchSize());
        EXPECT_EQ(output.getChannels(), expected_output.getChannels());
        EXPECT_EQ(output.getHeight(), expected_output.getHeight());
        EXPECT_EQ(output.getWidth(), expected_output.getWidth());

        // Check output values
        for (size_t n = 0; n < output.getBatchSize(); ++n) {
            for (size_t c = 0; c < output.getChannels(); ++c) {
                for (size_t h = 0; h < output.getHeight(); ++h) {
                    for (size_t w = 0; w < output.getWidth(); ++w) {
                        EXPECT_TRUE(is_close(output(n, c, h, w), expected_output(n, c, h, w)));
                    }
                }
            }
        }
    }

} // namespace nnm