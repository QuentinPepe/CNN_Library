#include "TicTacToeModel.h"
#include "Tensor4D.h"
#include <iostream>
#include <iomanip>

// Function to print the Tic-Tac-Toe board
void printBoard(const nnm::Tensor4D &board) {
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            char symbol = ' ';
            if (board(0, 0, i, j) == 1) symbol = 'X';
            else if (board(0, 1, i, j) == 1) symbol = 'O';
            std::cout << " " << symbol << " ";
            if (j < 2) std::cout << "|";
        }
        std::cout << std::endl;
        if (i < 2) std::cout << "---+---+---" << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // Create an instance of the TicTacToeModel
    nnm::TicTacToeModel model;

    // Create a sample Tic-Tac-Toe board state
    // 0 = empty, 1 = X, 2 = O
    nnm::Tensor4D board(1, 3, 3, 3);

    // Set up the board:
    // X |   | O
    // --+---+--
    //   | X |
    // --+---+--
    // O |   |

    board(0, 0, 0, 0) = 1;
    board(0, 0, 1, 1) = 1;

    board(0, 1, 0, 2) = 1;
    board(0, 1, 2, 0) = 1;

    board(0, 2, 0, 1) = 1;
    board(0, 2, 1, 0) = 1;
    board(0, 2, 1, 2) = 1;
    board(0, 2, 2, 1) = 1;

    // Print the board
    std::cout << "Current Tic-Tac-Toe board:" << std::endl;
    printBoard(board);

    // Evaluate the board using the model
    auto [policy, value] = model.forward(board);

    // Print the policy (move probabilities)
    std::cout << "Move probabilities:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            std::cout << std::fixed << std::setprecision(4) << policy(0, i * 3 + j, 0, 0) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Print the value (game outcome prediction)
    std::cout << "Predicted game outcome: " << value(0, 0, 0, 0) << std::endl;
    std::cout << "(-1 favors O, +1 favors X)" << std::endl;

    return 0;
}