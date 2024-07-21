#include <gtest/gtest.h>
#include "Tanh.h"
#include "Tensor4D.h"
#include <cmath>

namespace nnm {

    class TanhTest : public ::testing::Test {
    protected:
        static constexpr float epsilon = 1e-5f;

        static bool is_close(float a, float b, float eps = epsilon) {
            return std::abs(a - b) < eps;
        }

        static Tensor4D create_input_tensor() {
            Tensor4D input(2, 3, 4, 4);
            input(0, 0, 0, 0) = -0.134056f;
            input(0, 0, 0, 1) = 0.190570f;
            input(0, 0, 0, 2) = 1.408148f;
            input(0, 0, 0, 3) = 0.561290f;
            input(0, 0, 1, 0) = 0.819460f;
            input(0, 0, 1, 1) = -0.017399f;
            input(0, 0, 1, 2) = -0.382460f;
            input(0, 0, 1, 3) = -1.894478f;
            input(0, 0, 2, 0) = -1.272947f;
            input(0, 0, 2, 1) = 0.346848f;
            input(0, 0, 2, 2) = -0.190590f;
            input(0, 0, 2, 3) = 0.251125f;
            input(0, 0, 3, 0) = 1.354181f;
            input(0, 0, 3, 1) = 2.168097f;
            input(0, 0, 3, 2) = 0.061362f;
            input(0, 0, 3, 3) = -1.198688f;
            input(0, 1, 0, 0) = 0.702076f;
            input(0, 1, 0, 1) = -0.067636f;
            input(0, 1, 0, 2) = 1.080532f;
            input(0, 1, 0, 3) = -0.555339f;
            input(0, 1, 1, 0) = -1.410267f;
            input(0, 1, 1, 1) = 1.118531f;
            input(0, 1, 1, 2) = -1.298394f;
            input(0, 1, 1, 3) = -0.194924f;
            input(0, 1, 2, 0) = 0.615407f;
            input(0, 1, 2, 1) = -0.219705f;
            input(0, 1, 2, 2) = 0.016174f;
            input(0, 1, 2, 3) = 1.980266f;
            input(0, 1, 3, 0) = 1.528478f;
            input(0, 1, 3, 1) = -1.234442f;
            input(0, 1, 3, 2) = 0.119240f;
            input(0, 1, 3, 3) = 1.385496f;
            input(0, 2, 0, 0) = 0.020588f;
            input(0, 2, 0, 1) = 0.419664f;
            input(0, 2, 0, 2) = -0.776302f;
            input(0, 2, 0, 3) = -1.245888f;
            input(0, 2, 1, 0) = 0.063344f;
            input(0, 2, 1, 1) = 0.341764f;
            input(0, 2, 1, 2) = -1.566137f;
            input(0, 2, 1, 3) = -0.131169f;
            input(0, 2, 2, 0) = 1.330379f;
            input(0, 2, 2, 1) = -0.042370f;
            input(0, 2, 2, 2) = -0.560258f;
            input(0, 2, 2, 3) = 0.694706f;
            input(0, 2, 3, 0) = -1.119833f;
            input(0, 2, 3, 1) = -1.034295f;
            input(0, 2, 3, 2) = 0.907062f;
            input(0, 2, 3, 3) = 0.488758f;
            input(1, 0, 0, 0) = -0.839473f;
            input(1, 0, 0, 1) = -0.774599f;
            input(1, 0, 0, 2) = -0.788852f;
            input(1, 0, 0, 3) = -0.547269f;
            input(1, 0, 1, 0) = -0.273061f;
            input(1, 0, 1, 1) = -0.515972f;
            input(1, 0, 1, 2) = -0.556523f;
            input(1, 0, 1, 3) = -1.181693f;
            input(1, 0, 2, 0) = -0.615457f;
            input(1, 0, 2, 1) = -1.299462f;
            input(1, 0, 2, 2) = 0.723680f;
            input(1, 0, 2, 3) = 1.682659f;
            input(1, 0, 3, 0) = -0.345002f;
            input(1, 0, 3, 1) = 0.567157f;
            input(1, 0, 3, 2) = 1.987562f;
            input(1, 0, 3, 3) = -0.520725f;
            input(1, 1, 0, 0) = -0.437542f;
            input(1, 1, 0, 1) = 0.991216f;
            input(1, 1, 0, 2) = 0.409636f;
            input(1, 1, 0, 3) = 1.050403f;
            input(1, 1, 1, 0) = -0.687172f;
            input(1, 1, 1, 1) = 0.646349f;
            input(1, 1, 1, 2) = 0.550258f;
            input(1, 1, 1, 3) = -2.036966f;
            input(1, 1, 2, 0) = -1.322377f;
            input(1, 1, 2, 1) = 0.480516f;
            input(1, 1, 2, 2) = -0.430488f;
            input(1, 1, 2, 3) = 0.036525f;
            input(1, 1, 3, 0) = 0.126931f;
            input(1, 1, 3, 1) = 1.569382f;
            input(1, 1, 3, 2) = 0.258351f;
            input(1, 1, 3, 3) = -0.655347f;
            input(1, 2, 0, 0) = 0.381050f;
            input(1, 2, 0, 1) = 1.716223f;
            input(1, 2, 0, 2) = -0.003921f;
            input(1, 2, 0, 3) = -2.056408f;
            input(1, 2, 1, 0) = -1.591933f;
            input(1, 2, 1, 1) = -2.021689f;
            input(1, 2, 1, 2) = 0.789633f;
            input(1, 2, 1, 3) = 0.538178f;
            input(1, 2, 2, 0) = 0.671160f;
            input(1, 2, 2, 1) = -1.003575f;
            input(1, 2, 2, 2) = 0.475604f;
            input(1, 2, 2, 3) = 0.837653f;
            input(1, 2, 3, 0) = 0.837364f;
            input(1, 2, 3, 1) = -1.213159f;
            input(1, 2, 3, 2) = 0.499103f;
            input(1, 2, 3, 3) = -0.374865f;

            return input;
        }

        static Tensor4D create_expected_output() {
            Tensor4D expected_output(2, 3, 4, 4);
            expected_output(0, 0, 0, 0) = -0.133259f;
            expected_output(0, 0, 0, 1) = 0.188296f;
            expected_output(0, 0, 0, 2) = 0.887100f;
            expected_output(0, 0, 0, 3) = 0.508934f;
            expected_output(0, 0, 1, 0) = 0.674776f;
            expected_output(0, 0, 1, 1) = -0.017397f;
            expected_output(0, 0, 1, 2) = -0.364842f;
            expected_output(0, 0, 1, 3) = -0.955762f;
            expected_output(0, 0, 2, 0) = -0.854594f;
            expected_output(0, 0, 2, 1) = 0.333578f;
            expected_output(0, 0, 2, 2) = -0.188315f;
            expected_output(0, 0, 2, 3) = 0.245976f;
            expected_output(0, 0, 3, 0) = 0.875036f;
            expected_output(0, 0, 3, 1) = 0.974166f;
            expected_output(0, 0, 3, 2) = 0.061285f;
            expected_output(0, 0, 3, 3) = -0.833254f;
            expected_output(0, 1, 0, 0) = 0.605684f;
            expected_output(0, 1, 0, 1) = -0.067533f;
            expected_output(0, 1, 0, 2) = 0.793396f;
            expected_output(0, 1, 0, 3) = -0.504511f;
            expected_output(0, 1, 1, 0) = -0.887551f;
            expected_output(0, 1, 1, 1) = 0.807057f;
            expected_output(0, 1, 1, 2) = -0.861309f;
            expected_output(0, 1, 1, 3) = -0.192492f;
            expected_output(0, 1, 2, 0) = 0.547922f;
            expected_output(0, 1, 2, 1) = -0.216237f;
            expected_output(0, 1, 2, 2) = 0.016172f;
            expected_output(0, 1, 2, 3) = 0.962607f;
            expected_output(0, 1, 3, 0) = 0.910164f;
            expected_output(0, 1, 3, 1) = -0.843863f;
            expected_output(0, 1, 3, 2) = 0.118678f;
            expected_output(0, 1, 3, 3) = 0.882176f;
            expected_output(0, 2, 0, 0) = 0.020585f;
            expected_output(0, 2, 0, 1) = 0.396648f;
            expected_output(0, 2, 0, 2) = -0.650579f;
            expected_output(0, 2, 0, 3) = -0.847127f;
            expected_output(0, 2, 1, 0) = 0.063259f;
            expected_output(0, 2, 1, 1) = 0.329051f;
            expected_output(0, 2, 1, 2) = -0.916409f;
            expected_output(0, 2, 1, 3) = -0.130421f;
            expected_output(0, 2, 2, 0) = 0.869342f;
            expected_output(0, 2, 2, 1) = -0.042345f;
            expected_output(0, 2, 2, 2) = -0.508169f;
            expected_output(0, 2, 2, 3) = 0.600997f;
            expected_output(0, 2, 3, 0) = -0.807511f;
            expected_output(0, 2, 3, 1) = -0.775625f;
            expected_output(0, 2, 3, 2) = 0.719719f;
            expected_output(0, 2, 3, 3) = 0.453230f;
            expected_output(1, 0, 0, 0) = -0.685530f;
            expected_output(1, 0, 0, 1) = -0.649596f;
            expected_output(1, 0, 0, 2) = -0.657758f;
            expected_output(1, 0, 0, 3) = -0.498470f;
            expected_output(1, 0, 1, 0) = -0.266471f;
            expected_output(1, 0, 1, 1) = -0.474585f;
            expected_output(1, 0, 1, 2) = -0.505393f;
            expected_output(1, 0, 1, 3) = -0.827985f;
            expected_output(1, 0, 2, 0) = -0.547957f;
            expected_output(1, 0, 2, 1) = -0.861585f;
            expected_output(1, 0, 2, 2) = 0.619184f;
            expected_output(1, 0, 2, 3) = 0.933206f;
            expected_output(1, 0, 3, 0) = -0.331936f;
            expected_output(1, 0, 3, 1) = 0.513268f;
            expected_output(1, 0, 3, 2) = 0.963138f;
            expected_output(1, 0, 3, 3) = -0.478260f;
            expected_output(1, 1, 0, 0) = -0.411605f;
            expected_output(1, 1, 0, 1) = 0.757880f;
            expected_output(1, 1, 0, 2) = 0.388164f;
            expected_output(1, 1, 0, 3) = 0.781963f;
            expected_output(1, 1, 1, 0) = -0.596162f;
            expected_output(1, 1, 1, 1) = 0.569207f;
            expected_output(1, 1, 1, 2) = 0.500714f;
            expected_output(1, 1, 1, 3) = -0.966548f;
            expected_output(1, 1, 2, 0) = -0.867374f;
            expected_output(1, 1, 2, 1) = 0.446657f;
            expected_output(1, 1, 2, 2) = -0.405729f;
            expected_output(1, 1, 2, 3) = 0.036508f;
            expected_output(1, 1, 3, 0) = 0.126253f;
            expected_output(1, 1, 3, 1) = 0.916927f;
            expected_output(1, 1, 3, 2) = 0.252752f;
            expected_output(1, 1, 3, 3) = -0.575259f;
            expected_output(1, 2, 0, 0) = 0.363619f;
            expected_output(1, 2, 0, 1) = 0.937407f;
            expected_output(1, 2, 0, 2) = -0.003921f;
            expected_output(1, 2, 0, 3) = -0.967804f;
            expected_output(1, 2, 1, 0) = -0.920445f;
            expected_output(1, 2, 1, 1) = -0.965528f;
            expected_output(1, 2, 1, 2) = 0.658201f;
            expected_output(1, 2, 1, 3) = 0.491607f;
            expected_output(1, 2, 2, 0) = 0.585742f;
            expected_output(1, 2, 2, 1) = -0.763091f;
            expected_output(1, 2, 2, 2) = 0.442716f;
            expected_output(1, 2, 2, 3) = 0.684564f;
            expected_output(1, 2, 3, 0) = 0.684410f;
            expected_output(1, 2, 3, 1) = -0.837625f;
            expected_output(1, 2, 3, 2) = 0.461411f;
            expected_output(1, 2, 3, 3) = -0.358239f;


            return expected_output;
        }
    };

    TEST_F(TanhTest, ForwardPass
    ) {
    // Create Tanh layer
    Tanh tanh;

    // Create input tensor
    Tensor4D input = create_input_tensor();

    // Create expected output tensor
    Tensor4D expected_output = create_expected_output();

    // Perform forward pass
    Tensor4D output = tanh.forward(input);

    // Check output dimensions
    EXPECT_EQ(output
    .

    getBatchSize(), expected_output

    .

    getBatchSize()

    );
    EXPECT_EQ(output
    .

    getChannels(), expected_output

    .

    getChannels()

    );
    EXPECT_EQ(output
    .

    getHeight(), expected_output

    .

    getHeight()

    );
    EXPECT_EQ(output
    .

    getWidth(), expected_output

    .

    getWidth()

    );

    // Check output values
    for (
    size_t n = 0;
    n<output.

    getBatchSize();

    ++n) {
    for (
    size_t c = 0;
    c<output.

    getChannels();

    ++c) {
    for (
    size_t h = 0;
    h<output.

    getHeight();

    ++h) {
    for (
    size_t w = 0;
    w<output.

    getWidth();

    ++w) {
    EXPECT_TRUE(is_close(output(n, c, h, w), expected_output(n, c, h, w))
    );
}
}
}
}
}

} // namespace nnm