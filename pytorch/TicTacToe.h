#pragma once

#include <cstdint>
#include <ATen/Tensor.h>
#include <ATen/ops/zeros.h>
#include "Tensor4D.h"

constexpr uint16_t BOARD_MASK = 511;

enum class Player : uint8_t {
    x = 0,
    o = 1
};

enum class GameState : uint8_t {
    x_win = 0,
    o_win = 1,
    draw = 2,
    running = 3
};

enum class Cell : uint8_t {
    x,
    o,
    empty
};


struct TicTacToe {

private:
    uint16_t x_board;
    uint16_t o_board;
    GameState game_state;
    Cell currentPlayer;

    static bool checkWin(uint16_t board) {
        uint16_t horizontal = 0b000000111;
        for (int i = 0; i < 3; i++) {
            if ((board & horizontal) == horizontal) return true;
            horizontal <<= 3;
        }

        uint16_t vertical = 0b000000001 | (0b000000001 << 3) | (0b000000001 << 6);
        for (int i = 0; i < 3; i++) {
            if ((board & vertical) == vertical) return true;
            vertical <<= 1;
        }

        uint16_t diagonal1 = 0b100010001;
        uint16_t diagonal2 = 0b001010100;
        if ((board & diagonal1) == diagonal1 || (board & diagonal2) == diagonal2) return true;

        return false;
    }

public:
    TicTacToe() : x_board(0), o_board(0), game_state(GameState::running), currentPlayer(Cell::x) {}

    TicTacToe(const TicTacToe &other) : x_board(other.x_board), o_board(other.o_board), game_state(other.game_state),
                                        currentPlayer(other.currentPlayer) {}

    void makeMove(uint8_t move) {
        uint16_t bit = 1 << move;
        if (currentPlayer == Cell::x) {
            x_board |= bit;
            game_state = checkWin(x_board) ? GameState::x_win : GameState::running;
            currentPlayer = Cell::o;
        } else if (currentPlayer == Cell::o) {
            o_board |= bit;
            game_state = checkWin(o_board) ? GameState::o_win : GameState::running;
            currentPlayer = Cell::x;
        }

        if (game_state == GameState::running && (x_board | o_board) == BOARD_MASK) {
            game_state = GameState::draw;
        }
    }

    [[nodiscard]] std::array<uint8_t, 9> getLegalMoves() const {
        uint16_t board = ~(x_board | o_board) & BOARD_MASK;
        std::array<uint8_t, 9> legal_moves{};
        while (board) {
            uint8_t move = __builtin_ctz(board);
            board &= board - 1;
            legal_moves[move] = 1;
        }
        return legal_moves;
    }

    void printLegalMoves() const {
        auto legal_moves = getLegalMoves();
        std::cout << "Legal moves: ";
        for (int i = 0; i < 9; i++) {
            if (legal_moves[i]) {
                std::cout << i << " ";
            } else {
                std::cout << ". ";
            }
        }
        std::cout << std::endl;
    }

    [[nodiscard]] nnm::Tensor4D getEncodedState() const {
        nnm::Tensor4D tensor(1, 3, 3, 3);

        int xIndex = currentPlayer == Cell::x ? 0 : 1;
        int oIndex = abs(1 - xIndex);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (x_board & (1 << (i * 3 + j))) {
                    tensor(0, xIndex, i, j) = 1;
                } else if (o_board & (1 << (i * 3 + j))) {
                    tensor(0, oIndex, i, j) = 1;
                } else {
                    tensor(0, 2, i, j) = 1;
                }
            }
        }
        return tensor;
    }

    [[nodiscard]] at::Tensor getTorchEncodedState() const {
        at::Tensor tensor = at::zeros({3, 3, 3}, at::kFloat);

        int xIndex = currentPlayer == Cell::x ? 0 : 1;
        int oIndex = abs(1 - xIndex);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (x_board & (1 << (i * 3 + j))) {
                    tensor[xIndex][i][j] = 1;
                } else if (o_board & (1 << (i * 3 + j))) {
                    tensor[oIndex][i][j] = 1;
                } else {
                    tensor[2][i][j] = 1;
                }
            }
        }
        return tensor;
    }


    [[nodiscard]] GameState getGameState() const {
        return game_state;
    }

    [[nodiscard]] int getActionSize() const {
        return 9;
    }

    [[nodiscard]] std::pair<float, bool> getValueAndTerminated() const {
        switch (game_state) {
            case GameState::x_win:
                return {currentPlayer == Cell::o ? 1.0f : -1.0f, true};
            case GameState::o_win:
                return {currentPlayer == Cell::x ? 1.0f : -1.0f, true};
            case GameState::draw:
                return {0.0f, true};
            case GameState::running:
                return {0.0f, false};
            default:
                throw std::runtime_error("Invalid game state");
        }
    }

    void printBoard() const {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (x_board & (1 << (i * 3 + j))) {
                    std::cout << "X ";
                } else if (o_board & (1 << (i * 3 + j))) {
                    std::cout << "O ";
                } else {
                    std::cout << ". ";
                }
            }
            std::cout << std::endl;
        }
    }

    TicTacToe *clone() {
        auto *game = new TicTacToe();
        game->x_board = x_board;
        game->o_board = o_board;
        game->game_state = game_state;
        game->currentPlayer = currentPlayer;
        return game;
    }
};