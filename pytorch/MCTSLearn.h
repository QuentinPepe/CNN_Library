#pragma once

#include <torch/nn/module.h>
#include <utility>
#include <random>
#include "Node.h"
#include "TicTacToeModel.h"

constexpr float C = 1.414;

class MCTSLearn {
private:
    float dirichlet_epsilon;
    float dirichlet_alpha;

    Node *select(Node *node) {
        Node *best_child = nullptr;
        float best_ucb = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < node->getChildCount(); ++i) {
            Node *child = node->getChild(i);
            float ucb = computeUCB(node, child);
            if (ucb > best_ucb) {
                best_child = child;
                best_ucb = ucb;
            }
        }

        return best_child;
    }

    float computeUCB(Node *parent, Node *child) {
        float q_value = 0.0f;
        if (child->getVisits() != 0) {
            q_value = 1 - ((child->getReward() / child->getVisits()) + 1) / 2;
        }
        return q_value + C * (sqrtf(parent->getVisits()) / (child->getVisits() + 1)) * child->getProbability();
    }

public:
    MCTSLearn(float dirichlet_epsilon, float dirichlet_alpha) : dirichlet_epsilon(dirichlet_epsilon),
                                                                dirichlet_alpha(dirichlet_alpha) {}

    std::vector<float> search(TicTacToe *game, TicTacToeModel &model, int num_searches) {
        Node root(game, nullptr, 9);
        at::Tensor state = game->getTorchEncodedState().to(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        auto [policy, _] = model(state.unsqueeze(0));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::vector<float> alpha_vec(game->getActionSize(), dirichlet_alpha);
        std::gamma_distribution<float> gamma(dirichlet_alpha, 1.0f);

        at::Tensor noise = torch::zeros_like(policy);
        float noise_sum = 0.0f;
        for (int i = 0; i < game->getActionSize(); ++i) {
            float sample = gamma(gen);
            noise[0][i] = sample;
            noise_sum += sample;
        }
        noise /= noise_sum;

        policy = (1 - dirichlet_epsilon) * policy + dirichlet_epsilon * noise;

        const auto valid_moves = game->getLegalMoves();

        for (int i = 0; i < game->getActionSize(); ++i) {
            if (!valid_moves[i]) {
                policy[0][i] = 0.0f;
            }
        }
        policy /= policy.sum();

        for (int action = 0; action < game->getActionSize(); ++action) {
            if (valid_moves[action]) {
                TicTacToe *child_game = game->clone();
                child_game->makeMove(action);
                root.addChild(child_game, action, policy[0][action].item<float>());
            }
        }

        for (int search = 0; search < num_searches; ++search) {
            Node *node = &root;
            while (node->getChild(0) != nullptr) {
                node = select(node);
            }

            float value;
            bool is_terminal;
            std::tie(value, is_terminal) = node->getGame()->getValueAndTerminated();
            value = -value;

            if (!is_terminal) {
                at::Tensor child_state = node->getGame()->getTorchEncodedState().to(
                        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
                auto [child_policy, child_value] = model(child_state.unsqueeze(0));

                const auto child_valid_moves = node->getGame()->getLegalMoves();
                for (int i = 0; i < game->getActionSize(); ++i) {
                    if (!child_valid_moves[i]) {
                        child_policy[0][i] = 0.0f;
                    }
                }

                child_policy /= child_policy.sum();

                value = child_value.item<float>();

                for (int action = 0; action < game->getActionSize(); ++action) {
                    if (child_valid_moves[action]) {
                        TicTacToe *grandchild_game = node->getGame()->clone();
                        grandchild_game->makeMove(action);
                        node->addChild(grandchild_game, action, child_policy[0][action].item<float>());
                    }
                }
            }

            while (node != nullptr) {
                node->updateStats(-value);
                node = node->getParent();
                value = -value;
            }
        }

        std::vector<float> action_probs(game->getActionSize(), 0.0f);
        float total_visits = 0.0f;
        for (int i = 0; i < root.getChildCount(); ++i) {
            Node *child = root.getChild(i);
            action_probs[child->getMove()] = child->getVisits();
            total_visits += child->getVisits();
        }
        for (auto &prob: action_probs) {
            prob /= total_visits;
        }

        return action_probs;
    }

};