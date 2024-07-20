#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>

using namespace std;

// Define a function for the naive forward pass of a convolutional layer
tuple<vector<vector<vector<vector<float>>>>, tuple<vector<vector<vector<vector<float>>>>, vector<vector<
        vector<vector<float>>>>, vector<float>, tuple<int, int>>>

cnn_forward_naive(
        const vector<vector<vector<vector<float>>

        >> &x,
        const vector<vector<vector<vector<float>>>> &w,
        const vector<float> &b,
        const tuple<int, int> &cnn_params
) {
    int stride, pad;
    tie(stride, pad
    ) =
            cnn_params;

// Get dimensions
    int N = x.size();
    int C = x[0].size();
    int H = x[0][0].size();
    int W = x[0][0][0].size();

    int F = w.size();
    int HH = w[0].size();
    int WW = w[0][0].size();

// Output dimensions
    int height_out = 1 + (H + 2 * pad - HH) / stride;
    int width_out = 1 + (W + 2 * pad - WW) / stride;

// Initialize output feature maps
    vector<vector<vector<vector<float>>>>
            feature_maps(N, vector<vector<vector<float>>
    >(F,
      vector<vector<float>>(height_out, vector<float>(width_out, 0)
      )));

// Zero padding
    vector<vector<vector<vector<float>>>>
            x_padded(N, vector<vector<vector<float>>
    >(C,
      vector<vector<float>>(H
                            + 2 * pad,
                            vector<float>(W
                                          + 2 * pad, 0))));

    for (
            int n = 0;
            n < N;
            ++n) {
        for (
                int c = 0;
                c < C;
                ++c) {
            for (
                    int i = 0;
                    i < H;
                    ++i) {
                for (
                        int j = 0;
                        j < W;
                        ++j) {
                    x_padded[n][c][i + pad][j + pad] = x[n][c][i][j];
                }
            }
        }
    }

// Convolution operation
    for (
            int n = 0;
            n < N;
            ++n) {
        for (
                int f = 0;
                f < F;
                ++f) {
            for (
                    int i = 0;
                    i < height_out;
                    ++i) {
                for (
                        int j = 0;
                        j < width_out;
                        ++j) {
                    float sum = 0;
                    for (
                            int c = 0;
                            c < C;
                            ++c) {
                        for (
                                int di = 0;
                                di < HH;
                                ++di) {
                            for (
                                    int dj = 0;
                                    dj < WW;
                                    ++dj) {
                                sum += x_padded[n][c][
                                               i * stride
                                               + di][
                                               j * stride
                                               + dj] * w[f][c][di][dj];
                            }
                        }
                    }
                    feature_maps[n][f][i][j] = sum + b[f];
                }
            }
        }
    }

// Return the result
    return
            make_tuple(feature_maps, make_tuple(x, w, b, cnn_params)
            );
}

// Define a function to calculate absolute error
float absolute_error(const vector<vector<vector<vector<float>>

>> &x, const vector<vector<vector<vector<float>>>> &y) {
    float error = 0;
    int N = x.size();
    int F = x[0].size();
    int H = x[0][0].size();
    int W = x[0][0][0].size();

    for (
            int n = 0;
            n < N;
            ++n) {
        for (
                int f = 0;
                f < F;
                ++f) {
            for (
                    int i = 0;
                    i < H;
                    ++i) {
                for (
                        int j = 0;
                        j < W;
                        ++j) {
                    error +=
                            fabs(x[n][f][i][j]
                                 - y[n][f][i][j]);
                }
            }
        }
    }

    return
            error;
}

int main() {
    // Define shapes
    int N = 1;  // Number of images
    int C = 3;  // Number of channels
    int H = 4;  // Height of image
    int W = 4;  // Width of image

    int F = 3;  // Number of filters
    int HH = 4; // Filter height
    int WW = 4; // Filter width

    // Initialize data
    vector<vector<vector<vector<float>>>> x(N, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(
            W))));
    vector<vector<vector<vector<float>>>> w(F, vector<vector<vector<float>>>(C, vector<vector<float>>(HH, vector<float>(
            WW))));
    vector<float> b(F);

    // Fill input data, weights, and biases
    float val = 0;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    x[n][c][i][j] = val++;
                }
            }
        }
    }

    val = -1.0;
    for (int f = 0; f < F; ++f) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < HH; ++i) {
                for (int j = 0; j < WW; ++j) {
                    w[f][c][i][j] = val;
                    val += 0.1;
                }
            }
        }
    }

    val = -1.0;
    for (int f = 0; f < F; ++f) {
        b[f] = val;
        val += 0.5;
    }

    // Convolution parameters
    tuple<int, int> cnn_params = make_tuple(2, 1);  // stride=2, pad=1

    // Calculate output
    auto [feature_maps, _] = cnn_forward_naive(x, w, b, cnn_params);

    // Define true output (for testing)
    vector<vector<vector<vector<float>>>> correct_out(N, vector<vector<vector<float>>>(F, vector<vector<float>>(2,
                                                                                                                vector<float>(
                                                                                                                        2))));
    // Populate correct_out with known values
    // In practice, these values would be derived from correct calculations or another trusted source

    // Calculate and print absolute error
    float error = absolute_error(correct_out, feature_maps);
    cout << "Absolute error: " << error << endl;

    // Print output feature maps for verification
    for (const auto &fm: feature_maps) {
        for (const auto &f: fm) {
            for (const auto &row: f) {
                for (float val: row) {
                    cout << val << ' ';
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }

    return 0;
}
