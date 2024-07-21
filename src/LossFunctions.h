#pragma once

#include "Tensor4D.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nnm {

    class LossFunctions {
    public:
        struct SoftmaxLossResult {
            float loss;
            Tensor4D gradient;
        };

        static SoftmaxLossResult softmax_loss(const Tensor4D &x, const Tensor4D &y) {
            if (x.getBatchSize() != y.getBatchSize() || y.getChannels() != 1 || y.getHeight() != 1 ||
                y.getWidth() != 1) {
                throw std::invalid_argument("Dimensions of x and y must match, and y should be a 1D tensor");
            }

            size_t N = x.getBatchSize();
            size_t C = x.getChannels();

            // Calculate shifted logits
            Tensor4D shifted_logits(N, C, 1, 1);
            for (size_t i = 0; i < N; ++i) {
                float max_val = -std::numeric_limits<float>::max();
                for (size_t j = 0; j < C; ++j) {
                    max_val = std::max(max_val, x(i, j, 0, 0));
                }
                for (size_t j = 0; j < C; ++j) {
                    shifted_logits(i, j, 0, 0) = x(i, j, 0, 0) - max_val;
                }
            }

            // Calculate sum of exp(shifted_logits)
            Tensor4D z(N, 1, 1, 1);
            for (size_t i = 0; i < N; ++i) {
                float sum = 0.0f;
                for (size_t j = 0; j < C; ++j) {
                    sum += std::exp(shifted_logits(i, j, 0, 0));
                }
                z(i, 0, 0, 0) = sum;
            }

            // Calculate log probabilities and probabilities
            Tensor4D log_probs(N, C, 1, 1);
            Tensor4D probs(N, C, 1, 1);
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < C; ++j) {
                    log_probs(i, j, 0, 0) = shifted_logits(i, j, 0, 0) - std::log(z(i, 0, 0, 0));
                    probs(i, j, 0, 0) = std::exp(log_probs(i, j, 0, 0));
                }
            }

            // Calculate loss
            float loss = 0.0f;
            for (size_t i = 0; i < N; ++i) {
                size_t label = static_cast<size_t>(y(i, 0, 0, 0));
                if (label < 0 || label >= C) {
                    throw std::out_of_range("Label must be between 0 and C-1");
                }
                loss -= log_probs(i, label, 0, 0);
            }
            loss /= N;

            // Calculate gradient
            Tensor4D dx = probs;
            for (size_t i = 0; i < N; ++i) {
                size_t label = static_cast<size_t>(y(i, 0, 0, 0));
                dx(i, label, 0, 0) -= 1.0f;
            }
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < C; ++j) {
                    dx(i, j, 0, 0) /= N;
                }
            }

            return {loss, dx};
        }
    };

} // namespace nnm