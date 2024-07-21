#include <memory>
#include "Tensor4D.h"
#include "Sequential.h"
#include "ResBlock.h"
#include "FlattenLayer.h"
#include "LinearLayer.h"
#include "Tanh.h"

namespace nnm {

    class ResNet : public Layer<Tensor4D, Tensor4D> {
    private:
        std::unique_ptr<Sequential> startBlock;
        std::vector<std::unique_ptr<ResBlock>> backBone;
        std::unique_ptr<Sequential> policyHead;
        std::unique_ptr<Sequential> valueHead;

        size_t action_size;
        size_t row_count;
        size_t column_count;

    public:
        ResNet(size_t num_resBlocks, size_t num_hidden, size_t action_size, size_t row_count, size_t column_count)
                : action_size(action_size), row_count(row_count), column_count(column_count) {

            startBlock = std::make_unique<Sequential>();
            startBlock->add_layer(std::make_unique<ConvolutionalLayer>(3, num_hidden, 3, 1, 1));
            startBlock->add_layer(std::make_unique<BatchNorm2d>(num_hidden));
            startBlock->add_layer(std::make_unique<ReLULayer>());

            for (size_t i = 0; i < num_resBlocks; ++i) {
                backBone.push_back(std::make_unique<ResBlock>(num_hidden));
            }

            policyHead = std::make_unique<Sequential>();
            policyHead->add_layer(std::make_unique<ConvolutionalLayer>(num_hidden, 32, 3, 1, 1));
            policyHead->add_layer(std::make_unique<BatchNorm2d>(32));
            policyHead->add_layer(std::make_unique<ReLULayer>());
            policyHead->add_layer(std::make_unique<Flatten>());
            policyHead->add_layer(std::make_unique<LinearLayer>(32 * row_count * column_count, action_size));

            valueHead = std::make_unique<Sequential>();
            valueHead->add_layer(std::make_unique<ConvolutionalLayer>(num_hidden, 3, 3, 1, 1));
            valueHead->add_layer(std::make_unique<BatchNorm2d>(3));
            valueHead->add_layer(std::make_unique<ReLULayer>());
            valueHead->add_layer(std::make_unique<Flatten>());
            valueHead->add_layer(std::make_unique<LinearLayer>(3 * row_count * column_count, 1));
            valueHead->add_layer(std::make_unique<Tanh>());
        }

        std::pair<Tensor4D, Tensor4D> forward(const Tensor4D &input) {
            Tensor4D x = startBlock->forward(input);
            for (const auto &resBlock: backBone) {
                x = resBlock->forward(x);
            }
            Tensor4D policy = policyHead->forward(x);
            Tensor4D value = valueHead->forward(x);
            return {policy, value};
        }

        // Implement other necessary methods...
    };
}