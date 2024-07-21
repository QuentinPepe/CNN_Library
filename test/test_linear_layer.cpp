#include "gtest/gtest.h"
#include "LinearLayer.h"

TEST(LinearLayerTest, ForwardTest) {
    nnm::LinearLayer layer(4, 3);

    // Initialiser les poids et les biais avec les valeurs fournies
    nnm::Matrix weights(3, 4);
    weights(0, 0) = 0.5323059;
    weights(0, 1) = 0.49028996;
    weights(0, 2) = 0.71058714;
    weights(0, 3) = 0.33581635;
    weights(1, 0) = -0.3091866;
    weights(1, 1) = -0.25871873;
    weights(1, 2) = 0.273515;
    weights(1, 3) = 0.76105624;
    weights(2, 0) = 0.25172326;
    weights(2, 1) = -0.4380475;
    weights(2, 2) = -0.4352071;
    weights(2, 3) = -0.87533426;

    nnm::Vector bias(3);
    bias[0] = 0.0f;
    bias[1] = 0.0f;
    bias[2] = 0.0f;

    layer.set_weights(weights);
    layer.set_bias(bias);

    // Définir l'entrée
    nnm::Vector input(4);
    input[0] = 1.0f;
    input[1] = 2.0f;
    input[2] = 3.0f;
    input[3] = 4.0f;

    // Calculer la sortie attendue
    nnm::Vector expected_output(3);
    expected_output[0] = 4.987913;
    expected_output[1] = 3.038146;
    expected_output[2] = -5.4313297;

    // Calculer la sortie réelle
    nnm::Vector output = layer.forward(input);

    // Vérifier que la sortie réelle correspond à la sortie attendue
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(output[i], expected_output[i], 1e-5);
    }
}

