#pragma once

#include "Matrix.h"
#include "Vector.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nnm {

    class LossFunctions {
    public:
        struct SoftmaxLossResult {
            float loss;
            Matrix gradient;
        };

        static SoftmaxLossResult softmax_loss(const Matrix &x, const Vector &y) {
            if (x.getRows() != y.size()) {
                throw std::invalid_argument("Number of samples in x and y must match");
            }

            size_t N = x.getRows();
            size_t C = x.getCols();

            // Calculate shifted logits
            Matrix shifted_logits(N, C);
            for (size_t i = 0; i < N; ++i) {
                float max_val = -std::numeric_limits<float>::max();
                for (size_t j = 0; j < C; ++j) {
                    max_val = std::max(max_val, x(i, j));
                }
                for (size_t j = 0; j < C; ++j) {
                    shifted_logits(i, j) = x(i, j) - max_val;
                }
            }

            // Calculate sum of exp(shifted_logits)
            Vector z(N);
            for (size_t i = 0; i < N; ++i) {
                float sum = 0.0f;
                for (size_t j = 0; j < C; ++j) {
                    sum += std::exp(shifted_logits(i, j));
                }
                z[i] = sum;
            }

            // Calculate log probabilities and probabilities
            Matrix log_probs(N, C);
            Matrix probs(N, C);
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < C; ++j) {
                    log_probs(i, j) = shifted_logits(i, j) - std::log(z[i]);
                    probs(i, j) = std::exp(log_probs(i, j));
                }
            }

            // Calculate loss
            float loss = 0.0f;
            for (size_t i = 0; i < N; ++i) {
                if (y[i] < 0 || y[i] >= C) {
                    throw std::out_of_range("Label must be between 0 and C-1");
                }
                loss -= log_probs(i, static_cast<size_t>(y[i]));
            }
            loss /= N;

            // Calculate gradient
            Matrix dx = probs;
            for (size_t i = 0; i < N; ++i) {
                dx(i, static_cast<size_t>(y[i])) -= 1.0f;
            }
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < C; ++j) {
                    dx(i, j) /= N;
                }
            }

            return {loss, dx};
        }

        // You can add other loss functions here as needed
    };

} // namespace nnm