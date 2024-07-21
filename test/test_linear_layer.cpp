#include "gtest/gtest.h"
#include "LinearLayer.h"

TEST(LinearLayerTest, PyTorchComparisonTest) {
    nnm::LinearLayer layer(4, 3);

    // Initialize weights and biases
    nnm::Tensor4D weights(1, 4, 3, 1);
    weights(0, 0, 0, 0) = -0.30910671;
    weights(0, 0, 1, 0) = 0.29826927;
    weights(0, 0, 2, 0) = 0.12235552;
    weights(0, 1, 0, 0) = -0.42296350;
    weights(0, 1, 1, 0) = -0.45327181;
    weights(0, 1, 2, 0) = -0.06585890;
    weights(0, 2, 0, 0) = 0.05843717;
    weights(0, 2, 1, 0) = -0.35586637;
    weights(0, 2, 2, 0) = -0.10410565;
    weights(0, 3, 0, 0) = -0.35041487;
    weights(0, 3, 1, 0) = 0.07261944;
    weights(0, 3, 2, 0) = -0.25347137;

    nnm::Tensor4D bias(1, 3, 1, 1);
    bias(0, 0, 0, 0) = -0.14462149;
    bias(0, 1, 0, 0) = 0.06485361;
    bias(0, 2, 0, 0) = -0.44880193;
    layer.set_weights(weights);
    layer.set_bias(bias);

    // Define the input
    nnm::Tensor4D input(2, 4, 1, 1);
    input(0, 0, 0, 0) = 1.00000000;
    input(0, 1, 0, 0) = 2.00000000;
    input(0, 2, 0, 0) = 3.00000000;
    input(0, 3, 0, 0) = 4.00000000;
    input(1, 0, 0, 0) = 5.00000000;
    input(1, 1, 0, 0) = 6.00000000;
    input(1, 2, 0, 0) = 7.00000000;
    input(1, 3, 0, 0) = 8.00000000;


    // Define the expected output
    nnm::Tensor4D expected_output(2, 3, 1, 1);
    expected_output(0, 0, 0, 0) = -2.52600336;
    expected_output(0, 1, 0, 0) = -1.32054210;
    expected_output(0, 2, 0, 0) = -1.78436661;
    expected_output(1, 0, 0, 0) = -6.62219477;
    expected_output(1, 1, 0, 0) = -3.07353973;
    expected_output(1, 2, 0, 0) = -2.98868847;


    // Calculate the actual output
    nnm::Tensor4D output = layer.forward(input);

    // Verify that the actual output matches the expected output
    for (size_t n = 0; n < 2; ++n) {
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(output(n, i, 0, 0), expected_output(n, i, 0, 0), 1e-5);
        }
    }
}

