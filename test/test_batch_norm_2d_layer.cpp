#include <gtest/gtest.h>
#include "BatchNorm2d.h"
#include "Tensor4D.h"
#include "Vector.h"
#include <cmath>

namespace nnm {

    class BatchNorm2dTest : public ::testing::Test {
    protected:
        static constexpr float epsilon = 1e-4f;

        static bool is_close(float a, float b, float eps = epsilon) {
            return std::abs(a - b) < eps;
        }

        static Tensor4D create_input_tensor() {
            Tensor4D input(2, 3, 4, 4);
            input(0, 0, 0, 0) = 1.645867f;
            input(0, 0, 0, 1) = -1.360169f;
            input(0, 0, 0, 2) = 0.344565f;
            input(0, 0, 0, 3) = 0.519868f;
            input(0, 0, 1, 0) = -2.613322f;
            input(0, 0, 1, 1) = -1.696474f;
            input(0, 0, 1, 2) = -0.228242f;
            input(0, 0, 1, 3) = 0.279955f;
            input(0, 0, 2, 0) = 0.246926f;
            input(0, 0, 2, 1) = 0.076887f;
            input(0, 0, 2, 2) = 0.338006f;
            input(0, 0, 2, 3) = 0.454402f;
            input(0, 0, 3, 0) = 0.456940f;
            input(0, 0, 3, 1) = -0.865371f;
            input(0, 0, 3, 2) = 0.781308f;
            input(0, 0, 3, 3) = -0.926789f;
            input(0, 1, 0, 0) = -0.218834f;
            input(0, 1, 0, 1) = -2.435065f;
            input(0, 1, 0, 2) = -0.072915f;
            input(0, 1, 0, 3) = -0.033986f;
            input(0, 1, 1, 0) = 0.962518f;
            input(0, 1, 1, 1) = 0.349168f;
            input(0, 1, 1, 2) = -0.921462f;
            input(0, 1, 1, 3) = -0.056195f;
            input(0, 1, 2, 0) = -0.622698f;
            input(0, 1, 2, 1) = -0.463722f;
            input(0, 1, 2, 2) = 1.921782f;
            input(0, 1, 2, 3) = -0.402546f;
            input(0, 1, 3, 0) = 0.123902f;
            input(0, 1, 3, 1) = 1.164783f;
            input(0, 1, 3, 2) = 0.923374f;
            input(0, 1, 3, 3) = 1.387295f;
            input(0, 2, 0, 0) = -0.883376f;
            input(0, 2, 0, 1) = -0.418913f;
            input(0, 2, 0, 2) = -0.804826f;
            input(0, 2, 0, 3) = 0.565610f;
            input(0, 2, 1, 0) = 0.610365f;
            input(0, 2, 1, 1) = 0.466884f;
            input(0, 2, 1, 2) = 1.950657f;
            input(0, 2, 1, 3) = -1.063099f;
            input(0, 2, 2, 0) = -0.077326f;
            input(0, 2, 2, 1) = 0.116399f;
            input(0, 2, 2, 2) = -0.593991f;
            input(0, 2, 2, 3) = -1.243928f;
            input(0, 2, 3, 0) = -0.102093f;
            input(0, 2, 3, 1) = -1.033548f;
            input(0, 2, 3, 2) = -0.312639f;
            input(0, 2, 3, 3) = 0.245785f;
            input(1, 0, 0, 0) = -0.259642f;
            input(1, 0, 0, 1) = 0.118337f;
            input(1, 0, 0, 2) = 0.243959f;
            input(1, 0, 0, 3) = 1.164601f;
            input(1, 0, 1, 0) = 0.288576f;
            input(1, 0, 1, 1) = 0.386598f;
            input(1, 0, 1, 2) = -0.201064f;
            input(1, 0, 1, 3) = -0.117927f;
            input(1, 0, 2, 0) = 0.192199f;
            input(1, 0, 2, 1) = -0.772157f;
            input(1, 0, 2, 2) = -1.900345f;
            input(1, 0, 2, 3) = 0.130677f;
            input(1, 0, 3, 0) = -0.704294f;
            input(1, 0, 3, 1) = 0.314721f;
            input(1, 0, 3, 2) = 0.157393f;
            input(1, 0, 3, 3) = 0.385363f;
            input(1, 1, 0, 0) = 0.967146f;
            input(1, 1, 0, 1) = -0.991083f;
            input(1, 1, 0, 2) = 0.301605f;
            input(1, 1, 0, 3) = -0.107317f;
            input(1, 1, 1, 0) = 0.998457f;
            input(1, 1, 1, 1) = -0.498715f;
            input(1, 1, 1, 2) = 0.761111f;
            input(1, 1, 1, 3) = 0.618301f;
            input(1, 1, 2, 0) = 0.314049f;
            input(1, 1, 2, 1) = 0.213333f;
            input(1, 1, 2, 2) = -0.120051f;
            input(1, 1, 2, 3) = 0.360460f;
            input(1, 1, 3, 0) = -0.314035f;
            input(1, 1, 3, 1) = -1.078708f;
            input(1, 1, 3, 2) = 0.240811f;
            input(1, 1, 3, 3) = -1.396227f;
            input(1, 2, 0, 0) = -0.066144f;
            input(1, 2, 0, 1) = -0.358355f;
            input(1, 2, 0, 2) = -1.561562f;
            input(1, 2, 0, 3) = -0.354643f;
            input(1, 2, 1, 0) = 1.081073f;
            input(1, 2, 1, 1) = 0.131478f;
            input(1, 2, 1, 2) = 1.573538f;
            input(1, 2, 1, 3) = 0.781430f;
            input(1, 2, 2, 0) = -1.078658f;
            input(1, 2, 2, 1) = -0.720910f;
            input(1, 2, 2, 2) = 1.470793f;
            input(1, 2, 2, 3) = 0.275635f;
            input(1, 2, 3, 0) = 0.666781f;
            input(1, 2, 3, 1) = -0.994390f;
            input(1, 2, 3, 2) = -1.189364f;
            input(1, 2, 3, 3) = -1.195949f;

            return input;
        }
    };

    TEST_F(BatchNorm2dTest, ForwardPass) {
        // Create BatchNorm2d layer
        BatchNorm2d bn(3);

        // Set parameters
        Tensor4D weight(1, 3, 1, 1);
        weight(0, 0, 0, 0) = 1.0f;
        weight(0, 1, 0, 0) = 2.0f;
        weight(0, 2, 0, 0) = 3.0f;

        Tensor4D bias(1, 3, 1, 1);
        bias(0, 0, 0, 0) = 0.1f;
        bias(0, 1, 0, 0) = 0.2f;
        bias(0, 2, 0, 0) = 0.3f;
        Tensor4D running_mean(1, 3, 1, 1);
        running_mean(0, 0, 0, 0) = 0.5f;
        running_mean(0, 1, 0, 0) = 1.0f;
        running_mean(0, 2, 0, 0) = 1.5f;

        Tensor4D running_var(1, 3, 1, 1);
        running_var(0, 0, 0, 0) = 1.0f;
        running_var(0, 1, 0, 0) = 2.0f;
        running_var(0, 2, 0, 0) = 3.0f;

        bn.set_parameters(weight, bias, running_mean, running_var);

        // Create input tensor
        Tensor4D input = create_input_tensor();

        // Perform forward pass
        Tensor4D output = bn.forward(input);

        // Check output dimensions
        EXPECT_EQ(output.getBatchSize(), 2);
        EXPECT_EQ(output.getChannels(), 3);
        EXPECT_EQ(output.getHeight(), 4);
        EXPECT_EQ(output.getWidth(), 4);

        // Check some output values
        Tensor4D expected_output(2, 3, 4, 4);
        expected_output(0, 0, 0, 0) = 1.245861f;
        expected_output(0, 0, 0, 1) = -1.760160f;
        expected_output(0, 0, 0, 2) = -0.055434f;
        expected_output(0, 0, 0, 3) = 0.119868f;
        expected_output(0, 0, 1, 0) = -3.013307f;
        expected_output(0, 0, 1, 1) = -2.096463f;
        expected_output(0, 0, 1, 2) = -0.628238f;
        expected_output(0, 0, 1, 3) = -0.120044f;
        expected_output(0, 0, 2, 0) = -0.153072f;
        expected_output(0, 0, 2, 1) = -0.323111f;
        expected_output(0, 0, 2, 2) = -0.061993f;
        expected_output(0, 0, 2, 3) = 0.054402f;
        expected_output(0, 0, 3, 0) = 0.056940f;
        expected_output(0, 0, 3, 1) = -1.265365f;
        expected_output(0, 0, 3, 2) = 0.381307f;
        expected_output(0, 0, 3, 3) = -1.326782f;
        expected_output(0, 1, 0, 0) = -1.523687f;
        expected_output(0, 1, 0, 1) = -4.657904f;
        expected_output(0, 1, 0, 2) = -1.317327f;
        expected_output(0, 1, 0, 3) = -1.262274f;
        expected_output(0, 1, 1, 0) = 0.146993f;
        expected_output(0, 1, 1, 1) = -0.720413f;
        expected_output(0, 1, 1, 2) = -2.517351f;
        expected_output(0, 1, 1, 3) = -1.293681f;
        expected_output(0, 1, 2, 0) = -2.094836f;
        expected_output(0, 1, 2, 1) = -1.870010f;
        expected_output(0, 1, 2, 2) = 1.503594f;
        expected_output(0, 1, 2, 3) = -1.783494f;
        expected_output(0, 1, 3, 0) = -1.038986f;
        expected_output(0, 1, 3, 1) = 0.433038f;
        expected_output(0, 1, 3, 2) = 0.091634f;
        expected_output(0, 1, 3, 3) = 0.747717f;
        expected_output(0, 2, 0, 0) = -3.828121f;
        expected_output(0, 2, 0, 1) = -3.023650f;
        expected_output(0, 2, 0, 2) = -3.692070f;
        expected_output(0, 2, 0, 3) = -1.318409f;
        expected_output(0, 2, 1, 0) = -1.240891f;
        expected_output(0, 2, 1, 1) = -1.489407f;
        expected_output(0, 2, 1, 2) = 1.080560f;
        expected_output(0, 2, 1, 3) = -4.139410f;
        expected_output(0, 2, 2, 0) = -2.432004f;
        expected_output(0, 2, 2, 1) = -2.096464f;
        expected_output(0, 2, 2, 2) = -3.326892f;
        expected_output(0, 2, 2, 3) = -4.452615f;
        expected_output(0, 2, 3, 0) = -2.474901f;
        expected_output(0, 2, 3, 1) = -4.088227f;
        expected_output(0, 2, 3, 2) = -2.839577f;
        expected_output(0, 2, 3, 3) = -1.872360f;
        expected_output(1, 0, 0, 0) = -0.659638f;
        expected_output(1, 0, 0, 1) = -0.281661f;
        expected_output(1, 0, 0, 2) = -0.156039f;
        expected_output(1, 0, 0, 3) = 0.764597f;
        expected_output(1, 0, 1, 0) = -0.111423f;
        expected_output(1, 0, 1, 1) = -0.013402f;
        expected_output(1, 0, 1, 2) = -0.601060f;
        expected_output(1, 0, 1, 3) = -0.517924f;
        expected_output(1, 0, 2, 0) = -0.207799f;
        expected_output(1, 0, 2, 1) = -1.172150f;
        expected_output(1, 0, 2, 2) = -2.300333f;
        expected_output(1, 0, 2, 3) = -0.269321f;
        expected_output(1, 0, 3, 0) = -1.104288f;
        expected_output(1, 0, 3, 1) = -0.085278f;
        expected_output(1, 0, 3, 2) = -0.242605f;
        expected_output(1, 0, 3, 3) = -0.014637f;
        expected_output(1, 1, 0, 0) = 0.153537f;
        expected_output(1, 1, 0, 1) = -2.615809f;
        expected_output(1, 1, 0, 2) = -0.787677f;
        expected_output(1, 1, 0, 3) = -1.365979f;
        expected_output(1, 1, 1, 0) = 0.197817f;
        expected_output(1, 1, 1, 1) = -1.919497f;
        expected_output(1, 1, 1, 2) = -0.137839f;
        expected_output(1, 1, 1, 3) = -0.339803f;
        expected_output(1, 1, 2, 0) = -0.770079f;
        expected_output(1, 1, 2, 1) = -0.912512f;
        expected_output(1, 1, 2, 2) = -1.383987f;
        expected_output(1, 1, 2, 3) = -0.704444f;
        expected_output(1, 1, 3, 0) = -1.658321f;
        expected_output(1, 1, 3, 1) = -2.739730f;
        expected_output(1, 1, 3, 2) = -0.873653f;
        expected_output(1, 1, 3, 3) = -3.188768f;
        expected_output(1, 2, 0, 0) = -2.412637f;
        expected_output(1, 2, 0, 1) = -2.918760f;
        expected_output(1, 2, 0, 2) = -5.002771f;
        expected_output(1, 2, 0, 3) = -2.912331f;
        expected_output(1, 2, 1, 0) = -0.425602f;
        expected_output(1, 2, 1, 1) = -2.070346f;
        expected_output(1, 2, 1, 2) = 0.427372f;
        expected_output(1, 2, 1, 3) = -0.944598f;
        expected_output(1, 2, 2, 0) = -4.166359f;
        expected_output(1, 2, 2, 1) = -3.546722f;
        expected_output(1, 2, 2, 2) = 0.249411f;
        expected_output(1, 2, 2, 3) = -1.820659f;
        expected_output(1, 2, 3, 0) = -1.143175f;
        expected_output(1, 2, 3, 1) = -4.020402f;
        expected_output(1, 2, 3, 2) = -4.358107f;
        expected_output(1, 2, 3, 3) = -4.369514f;

        for (size_t n = 0; n < 2; ++n) {
            for (size_t c = 0; c < 3; ++c) {
                for (size_t h = 0; h < 4; ++h) {
                    for (size_t w = 0; w < 4; ++w) {
                        EXPECT_TRUE(is_close(output(n, c, h, w), expected_output(n, c, h, w)));
                    }
                }
            }
        }
    }

} // namespace nnm