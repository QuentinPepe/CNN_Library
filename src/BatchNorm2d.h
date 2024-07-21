#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include <vector>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>

namespace nnm {

    class BatchNorm2d : public Layer<Tensor4D, Tensor4D> {
    private:
        size_t num_features;
        double eps;
        std::optional<double> momentum;

        std::vector<float> weight;
        std::vector<float> bias;
        std::vector<float> running_mean;
        std::vector<float> running_var;

    public:
        BatchNorm2d(size_t num_features, double eps = 1e-5, std::optional<double> momentum = 0.1,
                    bool affine = true, bool track_running_stats = true)
                : num_features(num_features), eps(eps), momentum(momentum) {

            if (affine) {
                weight.resize(num_features, 1.0f);
                bias.resize(num_features, 0.0f);
            }

            if (track_running_stats) {
                running_mean.resize(num_features, 0.0f);
                running_var.resize(num_features, 1.0f);
            }
        }

        Tensor4D forward(const Tensor4D &input) override {
            if (input.getChannels() != num_features) {
                throw std::invalid_argument("Input channel dimension doesn't match num_features");
            }

            Tensor4D output(input.getBatchSize(), num_features, input.getHeight(), input.getWidth());

            for (size_t c = 0; c < num_features; ++c) {
                double mean = running_mean[c];
                double var = running_var[c];
                double gamma = weight[c];
                double beta = bias[c];
                double inv_std = 1.0 / std::sqrt(var + eps);

                // Première passe : calculer la moyenne et la variance exactes
                double sum = 0.0;
                double sq_sum = 0.0;
                size_t count = input.getBatchSize() * input.getHeight() * input.getWidth();

                for (size_t n = 0; n < input.getBatchSize(); ++n) {
                    for (size_t h = 0; h < input.getHeight(); ++h) {
                        for (size_t w = 0; w < input.getWidth(); ++w) {
                            double x = input(n, c, h, w);
                            sum += x;
                            sq_sum += x * x;
                        }
                    }
                }

                double batch_mean = sum / count;
                double batch_var = (sq_sum / count) - (batch_mean * batch_mean);

                // Deuxième passe : normaliser et appliquer gamma et beta
                for (size_t n = 0; n < input.getBatchSize(); ++n) {
                    for (size_t h = 0; h < input.getHeight(); ++h) {
                        for (size_t w = 0; w < input.getWidth(); ++w) {
                            double x = input(n, c, h, w);
                            double normalized = (x - batch_mean) / std::sqrt(batch_var + eps);
                            output(n, c, h, w) = static_cast<float>(gamma * normalized + beta);
                        }
                    }
                }
            }

            return output;
        }


        std::string get_name() const override {
            return "BatchNorm2d";
        }

        size_t get_input_size() const override {
            return num_features;
        }

        size_t get_output_size() const override {
            return num_features;
        }

        void set_parameters(const Tensor4D &weight, const Tensor4D &bias,
                            const Tensor4D &running_mean, const Tensor4D &running_var) {
            if (weight.getChannels() != num_features || bias.getChannels() != num_features ||
                running_mean.getChannels() != num_features || running_var.getChannels() != num_features) {
                throw std::invalid_argument("Parameter sizes do not match num_features");
            }

            for (size_t c = 0; c < num_features; ++c) {
                this->weight[c] = weight(0, c, 0, 0);
                this->bias[c] = bias(0, c, 0, 0);
                this->running_mean[c] = running_mean(0, c, 0, 0);
                this->running_var[c] = running_var(0, c, 0, 0);
            }
        }
    };

} // namespace nnm