#pragma once

#include <torch/torch.h>

class TicTacToeModelImpl : public torch::nn::Module {
public:
    TicTacToeModelImpl(const std::string &device = "cuda") : device_(device) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(16));
        pool = register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
        fc1 = register_module("fc1", torch::nn::Linear(16 * 1 * 1, 32));
        fc2 = register_module("fc2", torch::nn::Linear(32, 9));
        fc3 = register_module("fc3", torch::nn::Linear(32, 1));

        this->to(torch::Device(device_));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, bool eval_mode = false) {
        if (eval_mode) {
            this->eval();
        } else {
            this->train();
        }

        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);
        x = pool->forward(x);
        x = x.flatten(1);
        x = torch::relu(fc1->forward(x));

        auto policy = fc2->forward(x);
        auto value = fc3->forward(x);

        policy = torch::softmax(policy, 1);
        value = torch::tanh(value);

        return {policy, value};
    }

private:
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    std::string device_;
};

TORCH_MODULE(TicTacToeModel);