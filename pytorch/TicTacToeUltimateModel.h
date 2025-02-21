#include <torch/torch.h>

class TicTacToeUltimateModel : public torch::nn::Module {
public:
    explicit TicTacToeUltimateModel(torch::Device device = torch::kCPU)
            : conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(5, 8, 3).stride(1).padding(1))),
              fc1(8 * 9 * 9, 64),
              fc2(64, 81),
              fc3(64, 1),
              device(device) {
        register_module("conv1", conv1);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        to(device);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, bool eval_mode = false) {
        if (eval_mode) {
            eval();
        } else {
            train();
        }

        x = torch::relu(conv1(x));
        x = x.flatten(1);
        x = torch::relu(fc1(x));
        auto policy = fc2(x);
        auto value = fc3(x);

        policy = torch::softmax(policy, 1);
        value = torch::tanh(value);

        return std::make_tuple(policy, value);
    }

private:
    torch::nn::Conv2d conv1;
    torch::nn::Linear fc1, fc2, fc3;
    torch::Device device;
};