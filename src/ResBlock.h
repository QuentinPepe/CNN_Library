#pragma once

#include <memory>
#include "Tensor4D.h"
#include "Sequential.h"
#include "ConvolutionalLayer.h"
#include "BatchNorm2d.h"
#include "ReLULayer.h"

namespace nnm {
    class ResBlock : public Layer<Tensor4D, Tensor4D> {
    private:
        std::unique_ptr<ConvolutionalLayer> conv1;
        std::unique_ptr<BatchNorm2d> bn1;
        std::unique_ptr<ConvolutionalLayer> conv2;
        std::unique_ptr<BatchNorm2d> bn2;
        std::unique_ptr<ReLULayer> relu;

    public:
        ResBlock(size_t num_hidden) {
            conv1 = std::make_unique<ConvolutionalLayer>(num_hidden, num_hidden, 3, 1, 1);
            bn1 = std::make_unique<BatchNorm2d>(num_hidden);
            conv2 = std::make_unique<ConvolutionalLayer>(num_hidden, num_hidden, 3, 1, 1);
            bn2 = std::make_unique<BatchNorm2d>(num_hidden);
            relu = std::make_unique<ReLULayer>();
        }

        Tensor4D forward(const Tensor4D &input) override {
            Tensor4D x = input;
            Tensor4D residual = x;
            x = relu->forward(bn1->forward(conv1->forward(x)));
            x = bn2->forward(conv2->forward(x));
            x = x + residual;  // Assuming Tensor4D supports element-wise addition
            x = relu->forward(x);
            return x;
        }

        std::string get_name() const override { return "ResBlock"; }

        size_t get_input_size() const override {
            return conv1->get_input_size();
        }

        size_t get_output_size() const override {
            return conv2->get_output_size();
        }

    };
}