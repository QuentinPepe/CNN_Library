#include <iostream>
#include <random>
#include "Tensor4D.h"
#include "ResNet.h"

using namespace nnm;

Tensor4D generateRandomTensor(size_t batch_size, size_t channels, size_t height, size_t width) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    Tensor4D tensor(batch_size, channels, height, width);
    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    tensor(n, c, h, w) = dis(gen);
                }
            }
        }
    }

    return tensor;
}

int main() {
    // Define tensor dimensions
    size_t batch_size = 1;
    size_t channels = 3;
    size_t height = 3;
    size_t width = 3;

    // Generate a random tensor
    Tensor4D randomTensor = generateRandomTensor(batch_size, channels, height, width);

    // Print the random tensor
    std::cout << "Random Tensor:" << std::endl;
    randomTensor.print();

    // Create a simple ResNet
    size_t num_resBlocks = 4;
    size_t num_hidden = 64;
    size_t action_size = 9;
    ResNet resnet(num_resBlocks, num_hidden, action_size, height, width);

    // Perform a forward pass
    auto [policy, value] = resnet.forward(randomTensor);

    // Print the policy and value tensors
    std::cout << "Policy Tensor:" << std::endl;
    policy.print();

    std::cout << "Value Tensor:" << std::endl;
    value.print();

    return 0;
}
