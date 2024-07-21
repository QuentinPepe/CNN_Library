#include "Tensor4D.h"

nnm::Tensor4D nnm::create_tensor(const std::vector<std::vector<std::vector<std::vector<float>>>> &values) {
    size_t batch_size = values.size();
    size_t channels = values[0].size();
    size_t height = values[0][0].size();
    size_t width = values[0][0][0].size();

    Tensor4D tensor(batch_size, channels, height, width);

    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    tensor(n, c, h, w) = values[n][c][h][w];
                }
            }
        }
    }

    return tensor;
}