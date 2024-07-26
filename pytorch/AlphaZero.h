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

    std::vector<std::tuple<at::Tensor, std::vector<float>, float>> selfPlayParallel(int num_games) {
        std::vector<std::thread> threads;
        std::vector<std::vector<std::tuple<at::Tensor, std::vector<float>, float>>> thread_memories(num_games);

        for (int i = 0; i < num_games; ++i) {
            threads.emplace_back([this, i, &thread_memories]() {
                thread_memories[i] = this->selfPlaySingle();
            });
        }

        for (auto &thread: threads) {
            thread.join();
        }

        std::vector<std::tuple<at::Tensor, std::vector<float>, float>> combined_memory;
        for (const auto &memory: thread_memories) {
            combined_memory.insert(combined_memory.end(), memory.begin(), memory.end());
        }

        return combined_memory;
    }

    int determineOptimalThreadCount() {
        unsigned int hardware_threads = std::thread::hardware_concurrency();

        if (hardware_threads == 0) {
            hardware_threads = 2;
        }

        float thread_factor = args.count("thread_factor") ? args["thread_factor"] : 0.75f;

        return std::max(1, static_cast<int>(hardware_threads * thread_factor));
    }


    std::vector<std::tuple<at::Tensor, std::vector<float>, float>> selfPlaySingle() {
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
                std::vector<std::tuple<at::Tensor, std::vector<float>, float>> returnMemory;
                for (const auto &[hist_neutral_state, hist_action_probs, hist_player]: memory) {
                    float hist_outcome = (hist_player == player) ? value : -value;
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
            int num_threads = determineOptimalThreadCount();
            int games_per_thread = std::max(1, static_cast<int>(args["num_selfPlay_iterations"]) / num_threads);
            int total_games = num_threads * games_per_thread;

            std::cout << "Starting " << total_games << " self-play games on " << num_threads << " threads..."
                      << std::endl;

            for (int i = 0; i < total_games; i += num_threads) {
                int games_this_batch = std::min(num_threads, total_games - i);
                auto batch_memory = selfPlayParallel(games_this_batch);
                memory.insert(memory.end(), batch_memory.begin(), batch_memory.end());
                std::cout << "Completed " << i + games_this_batch << "/" << total_games << " games" << std::endl;
            }

            std::cout << "Self-play completed. Total samples: " << memory.size() << std::flush << std::endl;

            _model->train();
            for (int epoch = 0; epoch < args["num_epochs"]; ++epoch) {
                train(memory);
            }

            std::cout << "Saving model for iteration " << iteration + 1 << std::flush << std::endl;
            torch::save(_model, "model_" + std::to_string(iteration) + ".pt");
            torch::save(optimizer, "optimizer_" + std::to_string(iteration) + ".pt");
        }

        std::cout << "Training completed." << std::flush << std::endl;
    }
};