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
        std::optional<float> momentum;

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

            int64_t n_batch = input.getBatchSize();
            int64_t n_channel = num_features;

            std::vector<double> inv_std(n_channel);
            std::vector<double> gamma(n_channel);
            std::vector<double> beta(n_channel);

            for (int64_t c = 0; c < n_channel; c++) {
                inv_std[c] = 1.0 / std::sqrt(static_cast<double>(running_var[c]) + static_cast<double>(eps));
                gamma[c] = weight.empty() ? 1.0 : static_cast<double>(weight[c]);
                beta[c] = bias.empty() ? 0.0 : static_cast<double>(bias[c]);
            }

            for (int64_t n = 0; n < n_batch; ++n) {
                for (int64_t c = 0; c < n_channel; ++c) {
                    double inv_std_gamma = inv_std[c] * gamma[c];

                    for (int64_t h = 0; h < input.getHeight(); ++h) {
                        for (int64_t w = 0; w < input.getWidth(); ++w) {
                            double x = static_cast<double>(input(n, c, h, w));
                            double normalized = (x - running_mean[c]) * inv_std_gamma + beta[c];
                            output(n, c, h, w) = static_cast<float>(normalized);
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
                            const Tensor4D &running_mean, const Tensor4D &running_var, float momentum = 0.1f
        ) {
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


            this->momentum = momentum;
        }

        void set_momentum(float new_momentum) {
            momentum = new_momentum;
        }

        std::optional<float> get_momentum() const {
            return momentum;
        }
    };

} // namespace nnm