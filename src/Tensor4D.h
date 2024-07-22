#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <cmath>
#include <numeric>
#include "Matrix.h"

namespace nnm {
    class Tensor4D {
    private:
        std::vector<float> data;
        size_t batch_size, channels, height, width;

    public:
        Tensor4D(size_t batch_size, size_t channels, size_t height, size_t width)
                : batch_size(batch_size), channels(channels), height(height), width(width),
                  data(batch_size * channels * height * width, 0.0f) {}

        Tensor4D(size_t batch_size, size_t channels, size_t height, size_t width, float value)
                : batch_size(batch_size), channels(channels), height(height), width(width),
                  data(batch_size * channels * height * width, value) {}

        Tensor4D(const std::vector<std::vector<std::vector<std::vector<float>>>> &values) {
            size_t batch_size = values.size();
            size_t channels = values[0].size();
            size_t height = values[0][0].size();
            size_t width = values[0][0][0].size();

            data.resize(batch_size * channels * height * width);
            for (size_t n = 0; n < batch_size; ++n) {
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t h = 0; h < height; ++h) {
                        for (size_t w = 0; w < width; ++w) {
                            (*this)(n, c, h, w) = values[n][c][h][w];
                        }
                    }
                }
            }

            this->batch_size = batch_size;
            this->channels = channels;
            this->height = height;
            this->width = width;
        }

        Tensor4D(
                std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<float>>>> values)
                : batch_size(values.size()),
                  channels(values.begin()->size()),
                  height(values.begin()->begin()->size()),
                  width(values.begin()->begin()->begin()->size()) {
            data.resize(batch_size * channels * height * width);
            size_t n = 0;
            for (const auto &batch: values) {
                size_t c = 0;
                for (const auto &channel: batch) {
                    size_t h = 0;
                    for (const auto &row: channel) {
                        size_t w = 0;
                        for (float val: row) {
                            (*this)(n, c, h, w) = val;
                            ++w;
                        }
                        ++h;
                    }
                    ++c;
                }
                ++n;
            }
        }

        Tensor4D(std::initializer_list<float> values)
                : batch_size(1), channels(values.size()), height(1), width(1) {
            data.assign(values.begin(), values.end());
        }


        float &operator()(size_t n, size_t c, size_t h, size_t w) {
            return data[(n * channels * height * width) + (c * height * width) + (h * width) + w];
        }

        const float &operator()(size_t n, size_t c, size_t h, size_t w) const {
            return data[(n * channels * height * width) + (c * height * width) + (h * width) + w];
        }

        Tensor4D operator+(const Tensor4D &other) const {
            if (batch_size != other.batch_size || channels != other.channels ||
                height != other.height || width != other.width) {
                throw std::invalid_argument("Tensor4D.cpp dimensions do not match for addition");
            }
            Tensor4D result(batch_size, channels, height, width);
            for (size_t i = 0; i < data.size(); ++i) {
                result.data[i] = data[i] + other.data[i];
            }

            return result;
        }

        Tensor4D operator-(const Tensor4D &other) const {
            if (batch_size != other.batch_size || channels != other.channels ||
                height != other.height || width != other.width) {
                throw std::invalid_argument("Tensor4D.cpp dimensions do not match for subtraction");
            }
            Tensor4D result(batch_size, channels, height, width);
            for (size_t i = 0; i < data.size(); ++i) {
                result.data[i] = data[i] - other.data[i];
            }
            return result;
        }

        Tensor4D elementWiseMul(const Tensor4D &other) const {
            if (batch_size != other.batch_size || channels != other.channels ||
                height != other.height || width != other.width) {
                throw std::invalid_argument("Tensor4D.cpp dimensions do not match for element-wise multiplication");
            }
            Tensor4D result(batch_size, channels, height, width);
            for (size_t i = 0; i < data.size(); ++i) {
                result.data[i] = data[i] * other.data[i];
            }
            return result;
        }

        float sum() const {
            return std::accumulate(data.begin(), data.end(), 0.0f);
        }

        float max() const {
            return *std::max_element(data.begin(), data.end());
        }

        float mean() const {
            return std::accumulate(data.begin(), data.end(), 0.0f) / static_cast<float>(data.size());
        }

        void fill(float value) {
            std::fill(data.begin(), data.end(), value);
        }

        void print() const {
            for (size_t n = 0; n < batch_size; ++n) {
                std::cout << "Batch " << n << ":\n";
                for (size_t c = 0; c < channels; ++c) {
                    std::cout << "Channel " << c << ":\n";
                    for (size_t h = 0; h < height; ++h) {
                        for (size_t w = 0; w < width; ++w) {
                            std::cout << (*this)(n, c, h, w) << " ";
                        }
                        std::cout << "\n";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
        }

        Tensor4D pad(size_t pad_h, size_t pad_w) const {
            Tensor4D padded(batch_size, channels, height + 2 * pad_h, width + 2 * pad_w);
            for (size_t n = 0; n < batch_size; ++n) {
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t h = 0; h < height; ++h) {
                        for (size_t w = 0; w < width; ++w) {
                            padded(n, c, h + pad_h, w + pad_w) = (*this)(n, c, h, w);
                        }
                    }
                }
            }
            return padded;
        }

        Tensor4D subTensor(size_t start_n, size_t start_c, size_t start_h, size_t start_w,
                           size_t sub_batch, size_t sub_channels, size_t sub_height, size_t sub_width) const {
            Tensor4D sub(sub_batch, sub_channels, sub_height, sub_width);
            for (size_t n = 0; n < sub_batch; ++n) {
                for (size_t c = 0; c < sub_channels; ++c) {
                    for (size_t h = 0; h < sub_height; ++h) {
                        for (size_t w = 0; w < sub_width; ++w) {
                            sub(n, c, h, w) = (*this)(start_n + n, start_c + c, start_h + h, start_w + w);
                        }
                    }
                }
            }
            return sub;
        }

        bool operator==(const Tensor4D &other) const {
            if (getBatchSize() != other.getBatchSize() || getChannels() != other.getChannels() ||
                getHeight() != other.getHeight() || getWidth() != other.getWidth()) {
                return false;
            }
            for (size_t n = 0; n < getBatchSize(); ++n) {
                for (size_t c = 0; c < getChannels(); ++c) {
                    for (size_t h = 0; h < getHeight(); ++h) {
                        for (size_t w = 0; w < getWidth(); ++w) {
                            if ((*this)(n, c, h, w) != other(n, c, h, w)) {
                                return false;
                            }
                        }
                    }
                }
            }
            return true;
        }

        Matrix channelToMatrix(const Tensor4D &tensor, size_t batch_index, size_t channel_index) {
            size_t height = tensor.getHeight();
            size_t width = tensor.getWidth();
            Matrix result(height, width);

            size_t base_offset = (batch_index * tensor.getChannels() * height * width) +
                                 (channel_index * height * width);

            const float *tensor_data = tensor.getData().data() + base_offset;

            float *matrix_data = result.getData().data();

            size_t num_elements = height * width;

            size_t i = 0;
            for (; i + 7 < num_elements; i += 8) {
                __m256 tensor_values = _mm256_loadu_ps(tensor_data + i);
                _mm256_storeu_ps(matrix_data + i, tensor_values);
            }

            for (; i < num_elements; ++i) {
                matrix_data[i] = tensor_data[i];
            }

            return result;
        }


        [[nodiscard]] Tensor4D
        pad(const std::vector<std::pair<size_t, size_t>> &padding, float constant_value = 0.0f) const {
            if (padding.size() != 4) {
                throw std::invalid_argument("Padding should be specified for all 4 dimensions");
            }

            size_t new_batch = batch_size + padding[0].first + padding[0].second;
            size_t new_channels = channels + padding[1].first + padding[1].second;
            size_t new_height = height + padding[2].first + padding[2].second;
            size_t new_width = width + padding[3].first + padding[3].second;

            Tensor4D padded(new_batch, new_channels, new_height, new_width);
            padded.fill(constant_value);

            for (size_t n = 0; n < batch_size; ++n) {
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t h = 0; h < height; ++h) {
                        for (size_t w = 0; w < width; ++w) {
                            padded(n + padding[0].first,
                                   c + padding[1].first,
                                   h + padding[2].first,
                                   w + padding[3].first) = (*this)(n, c, h, w);
                        }
                    }
                }
            }

            return padded;
        }

        const float &operator()(size_t i, size_t j) const {
            return this->operator()(0, i, j, 0);
        }

        float &operator()(size_t i, size_t j) {
            return this->operator()(0, i, j, 0);
        }

        Tensor4D operator*(float scalar) const {
            Tensor4D result(batch_size, channels, height, width);
            for (size_t i = 0; i < data.size(); ++i) {
                result.data[i] = data[i] * scalar;
            }
            return result;
        }

        friend Tensor4D operator*(float scalar, const Tensor4D &tensor) {
            return tensor * scalar;
        }

        Tensor4D operator/(float scalar) const {
            if (scalar == 0.0f) {
                throw std::invalid_argument("Division by zero");
            }
            return (*this) * (1.0f / scalar);
        }


        size_t getBatchSize() const { return batch_size; }

        size_t getChannels() const { return channels; }

        size_t getHeight() const { return height; }

        size_t getWidth() const { return width; }

        const std::vector<float> &getData() const { return data; }

        std::vector<float> &getData() { return data; }

        Tensor4D(int batch_size, int channels, int height, int width, const std::vector<float> &data)
                : batch_size(batch_size), channels(channels), height(height), width(width), data(data) {}
    };

    Tensor4D create_tensor(const std::vector<std::vector<std::vector<std::vector<float>>>> &values);


} // namespace nnm

