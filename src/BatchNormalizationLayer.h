// BatchNormalizationLayer.h
#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include "Vector.h"

namespace nnm {
    class BatchNormalizationLayer : public Layer<Tensor4D, Tensor4D> {
    private:
        size_t num_features;
        float epsilon;
        float momentum;
        Vector running_mean;
        Vector running_var;
        Vector gamma;
        Vector beta;

    public:
        BatchNormalizationLayer(size_t num_features, float epsilon = 1e-5, float momentum = 0.1)
                : num_features(num_features), epsilon(epsilon), momentum(momentum),
                  running_mean(num_features), running_var(num_features),
                  gamma(num_features, 1.0f), beta(num_features, 0.0f) {}

        Tensor4D forward(const Tensor4D &input) override {
            // Implémentez ici la logique de forward pass pour la normalisation par lots
            // Utilisez running_mean et running_var pour l'inférence
            // ...
            return input; // Placeholder, remplacez par la vraie implémentation
        }

        std::string get_name() const override { return "BatchNormalizationLayer"; }

        size_t get_input_size() const override { return num_features; }

        size_t get_output_size() const override { return num_features; }

        std::unique_ptr<Layer<Tensor4D, Tensor4D>> clone() const override {
            return std::make_unique<BatchNormalizationLayer>(*this);
        }
    };
}