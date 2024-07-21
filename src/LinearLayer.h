#pragma once

#include "Layer.h"
#include "Matrix.h"
#include "Vector.h"
#include <random>

namespace nnm {

    class LinearLayer : public Layer<Vector, Vector> {
    private:
        Matrix weights;
        Vector bias;
        size_t in_features;
        size_t out_features;

    public:
        LinearLayer(size_t in_features, size_t out_features)
                : in_features(in_features), out_features(out_features),
                  weights(out_features, in_features), bias(out_features) {

            // Xavier/Glorot initialization
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0 / std::sqrt(in_features), 1.0 / std::sqrt(in_features));

            for (size_t i = 0; i < out_features; ++i) {
                for (size_t j = 0; j < in_features; ++j) {
                    weights(i, j) = dis(gen);
                }
                bias[i] = 0.0f; // Initialize bias to zero
            }
        }

        Vector forward(const Vector &input) override {
            if (input.size() != in_features) {
                throw std::invalid_argument("Input size does not match layer input features");
            }
            return weights * input + bias;
        }

        std::string get_name() const override {
            return "LinearLayer";
        }

        size_t get_input_size() const override {
            return in_features;
        }

        size_t get_output_size() const override {
            return out_features;
        }

        std::unique_ptr<Layer<Vector, Vector>> clone() const override {
            return std::make_unique<LinearLayer>(*this);
        }

        const Matrix &get_weights() const {
            return weights;
        }

        const Vector &get_bias() const {
            return bias;
        }

        void set_weights(const Matrix &new_weights) {
            if (new_weights.getRows() != out_features || new_weights.getCols() != in_features) {
                throw std::invalid_argument("New weights dimensions do not match layer dimensions");
            }
            weights = new_weights;
        }

        void set_bias(const Vector &new_bias) {
            if (new_bias.size() != out_features) {
                throw std::invalid_argument("New bias size does not match layer output features");
            }
            bias = new_bias;
        }
    };

} // namespace nnm