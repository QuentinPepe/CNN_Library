#include <torch/torch.h>
#include <iostream>
#include <map>
#include "TicTacToe.h"
#include "TicTacToeModel.h"
#include "MCTSLearn.h"
#include "AlphaZero.h"


int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (torch::cuda::is_available() ? "CUDA" : "CPU") << std::endl;

    TicTacToe game;
    TicTacToeModel model;
    model->to(device);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

    std::map<std::string, float> args = {
            {"num_iterations",          50},
            {"num_selfPlay_iterations", 500},
            {"num_epochs",              100},
            {"batch_size",              64},
            {"num_searches",            150},
            {"dirichlet_epsilon",       0.25},
            {"dirichlet_alpha",         0.3},
            {"temperature",             1.25}
    };

    AlphaZero alphaZero(model, optimizer, game, args);

    std::cout << "Starting training..." << std::endl;
    alphaZero.learn();
    std::cout << "Training completed." << std::endl;

    torch::save(model, "final_model.pt");
    std::cout << "Final _model saved." << std::endl;

    return 0;
}


int mainaaa() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (torch::cuda::is_available() ? "CUDA" : "CPU") << std::endl;

    TicTacToeModel model;
    model->to(device);

    torch::load(model, "model_3.pt");
    model->eval();

    TicTacToe game;
    std::map<std::string, float> args = {
            {"num_searches",      1500},
            {"dirichlet_epsilon", 0},
            {"dirichlet_alpha",   0.1},
            {"temperature",       1.0},
            {"thread_factor",     0.75}
    };
    MCTSLearn mcts(args["dirichlet_epsilon"], args["dirichlet_alpha"]);

    std::cout << "Welcome to Tic-Tac-Toe!" << std::endl;

    char choice;
    bool playerFirst;
    do {
        std::cout << "Do you want to play first? (y/n): ";
        std::cin >> choice;
    } while (choice != 'y' && choice != 'n');
    playerFirst = (choice == 'y');

    if (playerFirst) {
        std::cout << "You are X, the AI is O." << std::endl;
    } else {
        std::cout << "You are O, the AI is X." << std::endl;
    }

    while (true) {
        std::cout << "Current board:" << std::endl;
        game.printBoard();

        if (playerFirst) {
            int move;
            while (true) {
                std::cout << "Enter your move (0-8): ";
                std::cin >> move;
                auto legal_moves = game.getLegalMoves();
                if (move >= 0 && move < 9 && legal_moves[move]) {
                    break;
                }
                std::cout << "Invalid move. Try again." << std::endl;
            }
            game.makeMove(move);
        } else {
            std::vector<float> action_probs = mcts.search(game, model, args["num_searches"]);
            std::cout << "Action probabilities:" << std::endl;
            for (int i = 0; i < 9; ++i) {
                std::cout << i << ": " << action_probs[i] << " ";
            }
            std::cout << std::endl;

            std::discrete_distribution<> dist(action_probs.begin(), action_probs.end());
            std::mt19937 gen(std::random_device{}());
            int ai_move = dist(gen);

            auto [policy, value] = model(game.getTorchEncodedState().to(torch::kCUDA).unsqueeze(0));
            std::cout << "AI policy:" << std::endl;
            std::cout << policy << std::endl;

            game.makeMove(ai_move);
            std::cout << "AI chose move: " << ai_move << std::endl;
        }

        auto [value, is_terminal] = game.getValueAndTerminated();
        if (is_terminal) {
            std::cout << "Final board:" << std::endl;
            game.printBoard();
            if (value != 0) {
                std::cout << (!playerFirst ? "AI wins!" : "You win!") << std::endl;
            } else {
                std::cout << "It's a draw!" << std::endl;
            }
            break;
        }

        playerFirst = !playerFirst;
    }

    return 0;
}

