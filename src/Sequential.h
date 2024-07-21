#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include "Layer.h"
#include "Tensor4D.h"

namespace nnm {

    class Sequential : public Layer<Tensor4D, Tensor4D> {
    private:
        std::vector<std::unique_ptr<Layer<Tensor4D, Tensor4D>>> layers;

    public:
        Sequential() = default;

        // Add a layer to the sequence
        void add_layer(std::unique_ptr<Layer<Tensor4D, Tensor4D>> layer) {
            if (layers.empty()) {
                layers.push_back(std::move(layer));
            } else {
                // Ensure the layer's input size matches the previous layer's output size
                size_t prev_output_size = layers.back()->get_output_size();
                if (prev_output_size != layer->get_input_size() || prev_output_size == 0 ||
                    layer->get_input_size() == 0) {
                    throw std::invalid_argument("Layer input size does not match the previous layer's output size.");
                }
                layers.push_back(std::move(layer));
            }
        }

        Tensor4D forward(const Tensor4D &input) override {
            Tensor4D output = input;
            for (const auto &layer: layers) {
                output = layer->forward(output);
                std::cout << layer->get_name() << " output size: " << output.getBatchSize() << " "
                          << output.getChannels() << " " << output.getHeight() << " " << output.getWidth() << std::endl;
                output.print();
            }
            return output;
        }

        std::string get_name() const override {
            return "Sequential";
        }

        size_t get_input_size() const override {
            if (layers.empty()) {
                throw std::runtime_error("Sequential model is empty.");
            }
            return layers.front()->get_input_size();
        }

        size_t get_output_size() const override {
            if (layers.empty()) {
                throw std::runtime_error("Sequential model is empty.");
            }
            return layers.back()->get_output_size();
        }


        std::string extra_repr() const {
            std::string repr = "Layers:\n";
            for (const auto &layer: layers) {
                repr += "  " + layer->get_name() + "\n";
            }
            return repr;
        }
    };

} // namespace nnm
