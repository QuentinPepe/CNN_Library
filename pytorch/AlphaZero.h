#pragma once

#include <torch/torch.h>
#include <random>
#include <utility>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include "TicTacToe.h"
#include "TicTacToeModel.h"
#include "MCTSLearn.h"

class AlphaZero {
private:
    TicTacToeModel _model;
    torch::optim::Optimizer &optimizer;
    TicTacToe game;
    std::map<std::string, float> args;
    MCTSLearn mcts;

    std::vector<std::tuple<at::Tensor, std::vector<float>, float>> selfPlay() {
        std::vector<std::tuple<at::Tensor, std::vector<float>, Player>> memory;
        Player player = Player::x;
        TicTacToe state = game;

        while (true) {
            TicTacToe neutral_state = state;
            std::vector<float> action_probs = mcts.search(neutral_state, _model, args["num_searches"]);
            memory.emplace_back(neutral_state.getTorchEncodedState(), action_probs, player);

            std::array<float, 9> legal_temperature_probs = {0};
            for (int move = 0; move < 9; ++move) {
                legal_temperature_probs[move] = std::pow(action_probs[move], 1.0f / args["temperature"]);
            }

            float sum = std::accumulate(legal_temperature_probs.begin(), legal_temperature_probs.end(), 0.0f);
            for (float &prob: legal_temperature_probs) {
                prob /= sum;
            }

            std::discrete_distribution<> dist(legal_temperature_probs.begin(), legal_temperature_probs.end());
            std::mt19937 gen(std::random_device{}());
            int action = dist(gen);

            state.makeMove(action);

            auto [value, is_terminal] = state.getValueAndTerminated();

            if (is_terminal) {
                //state.printBoard();
                std::vector<std::tuple<at::Tensor, std::vector<float>, float>> returnMemory;
                for (const auto &[hist_neutral_state, hist_action_probs, hist_player]: memory) {
                    float hist_outcome = (hist_player == player) ? value : -value;
                    /*
                    if (hist_player == Player::x) {
                        std::cout << "Player X" << std::endl;
                    } else {
                        std::cout << "Player O" << std::endl;
                    }
                    std::cout << hist_outcome << std::endl;
                    std::cout << hist_neutral_state << std::endl;
                     */
                    returnMemory.emplace_back(hist_neutral_state, hist_action_probs, hist_outcome);
                }
                return returnMemory;
            }

            player = (player == Player::x) ? Player::o : Player::x;
        }
    }

    void train(const std::vector<std::tuple<at::Tensor, std::vector<float>, float>> &memory) {
        std::vector<size_t> indices(memory.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));

        auto device = _model->parameters()[0].device();

        for (size_t batchIdx = 0; batchIdx < memory.size(); batchIdx += args["batch_size"]) {
            std::vector<at::Tensor> states, policy_targets, value_targets;

            for (size_t i = batchIdx;
                 i < std::min(memory.size(), batchIdx + static_cast<size_t>(args["batch_size"])); ++i) {
                const auto &[state, policy, value] = memory[indices[i]];
                states.push_back(state.to(device));
                policy_targets.push_back(torch::tensor(policy).to(device));
                value_targets.push_back(torch::tensor(value).to(device));
            }

            auto state_batch = torch::stack(states);
            auto policy_target_batch = torch::stack(policy_targets);
            auto value_target_batch = torch::stack(value_targets).view({-1, 1});

            auto [out_policy, out_value] = _model(state_batch);

            auto policy_loss = torch::nn::functional::cross_entropy(out_policy, policy_target_batch);
            auto value_loss = torch::nn::functional::mse_loss(out_value, value_target_batch);
            auto loss = policy_loss + value_loss;

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
    }

public:
    AlphaZero(TicTacToeModel &model, torch::optim::Optimizer &optimizer,
              TicTacToe game, std::map<std::string, float> args)
            : _model(std::move(model)), optimizer(optimizer), game(game), args(args),
              mcts(args["dirichlet_epsilon"], args["dirichlet_alpha"]) {}

    void learn() {
        for (int iteration = 0; iteration < args["num_iterations"]; ++iteration) {
            std::cout << "Iteration " << iteration + 1 << "/" << args["num_iterations"] << std::flush << std::endl;

            std::vector<std::tuple<at::Tensor, std::vector<float>, float>> memory;

            _model->eval();
            for (int selfPlay_iteration = 0;
                 selfPlay_iteration < args["num_selfPlay_iterations"]; ++selfPlay_iteration) {
                /* std::cout << "Self-play iteration " << selfPlay_iteration + 1 << "/" << args["num_selfPlay_iterations"]
                          << std::flush << std::endl; */
                auto new_memory = selfPlay();
                memory.insert(memory.end(), new_memory.begin(), new_memory.end());
            }

            std::cout << "Self-play completed. Samples: " << memory.size() << std::flush << std::endl;

            _model->train();
            for (int epoch = 0; epoch < args["num_epochs"]; ++epoch) {
                train(memory);
            }

            std::cout << "Saving _model for iteration " << iteration + 1 << std::flush << std::endl;
            torch::save(_model, "model_" + std::to_string(iteration) + ".pt");
            torch::save(optimizer, "optimizer_" + std::to_string(iteration) + ".pt");
        }

        std::cout << "Training completed." << std::flush << std::endl;
    }
};