#pragma once

#include <torch/nn/module.h>
#include <utility>
#include <random>
#include "Node.h"
#include "TicTacToeModel.h"

constexpr float C = 2;

void backpropagate(Node *node, float value) {
    Node *n = node;
    while (n) {
        n->_reward += value;
        n->_nSims += 1;
        n = n->_parent;
        value *= -1;
    }
}

double ucb1(double cReward, double cNsims, int pNsims, float prior) {
    double exploitation = 0;
    if (cNsims != 0) {
        exploitation = cReward / cNsims;
    }
    const double exploration = sqrtf(pNsims) / (cNsims + 1);

    return exploitation + C * exploration * prior;
}

Node *selectUcb(const Node *n) {
    int bestI = -1;
    double bestScore = -1.0;
    for (int i = 0; i < n->_nMoves; i++) {
        const auto &c = n->_children[i];
        const double s = ucb1(c->_reward, c->_nSims, n->_nSims, c->_prior);
        if (s > bestScore) {
            bestScore = s;
            bestI = i;
        }
    }
    assert(bestI > -1);
    return n->_children[bestI].get();
}

class MCTSLearn {
private:
    float dirichlet_epsilon;
    float dirichlet_alpha;

public:
    MCTSLearn(float dirichlet_epsilon, float dirichlet_alpha) : dirichlet_epsilon(dirichlet_epsilon),
                                                                dirichlet_alpha(dirichlet_alpha) {}

    std::vector<float> search(const TicTacToe &game, TicTacToeModel &model, int num_searches) {
        Node root(game);
        for (int i = 0; i < num_searches; i++) {
            Node *node = selectAndExpand(&root, model);

            auto [value, terminated] = node->_game.getValueAndTerminated();

            if (!terminated) {
                auto [_, v] = model(node->_game.getTorchEncodedState().to(torch::kCUDA).unsqueeze(0));
                value = -v.item<float>();
            }
            backpropagate(node, value);
        }

        std::vector<float> action_probs(9, 0.0f);
        const auto legal_moves = root._game.getLegalMoves();
        float sum = 0.0f;
        for (const auto &child: root._children) {
            int move = child->_move;
            if (legal_moves[move]) {
                action_probs[move] = child->_nSims;
                sum += action_probs[move];
            }
        }

        for (int i = 0; i < 9; ++i) {
            action_probs[i] /= sum;
        }

        return action_probs;
    }


    Node *selectAndExpand(Node *root, TicTacToeModel &model) const {
        Node *n = root;
        while (true) {
            // return node if game terminated
            if (not n->_game.isRunning())
                return n;
            // expand if new child found
            if (n->_children.size() == 0) {

                auto [policy, _] = model(n->_game.getTorchEncodedState().to(torch::kCUDA).unsqueeze(0));
                const auto legal_moves = n->_game.getLegalMoves();

                std::random_device rd;
                std::mt19937 gen(rd());
                std::vector<float> alpha_vec(n->_game.getActionSize(), dirichlet_alpha);
                std::gamma_distribution<float> gamma(dirichlet_alpha, 1.0f);

                at::Tensor noise = torch::zeros_like(policy);
                float noise_sum = 0.0f;
                for (int i = 0; i < n->_game.getActionSize(); ++i) {
                    float sample = gamma(gen);
                    noise[0][i] = sample;
                    noise_sum += sample;
                }
                noise /= noise_sum;

                policy = (1 - dirichlet_epsilon) * policy + dirichlet_epsilon * noise;

                for (int i = 0; i < 9; ++i) {
                    if (!legal_moves[i]) {
                        policy[0][i] = 0;
                    }
                }
                policy /= policy.sum().item<float>();

                for (uint8_t i = 0; i < 9; ++i) {
                    if (legal_moves[i]) {
                        n->_children.push_back(std::make_unique<Node>(n->_game, n, i, policy[0][i].item<float>()));
                    }
                }
            }
            // select child node using UCB
            n = selectUcb(n);
        }
    }

};