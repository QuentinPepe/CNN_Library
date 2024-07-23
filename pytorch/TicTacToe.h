#include <cstdint>
#include "Tensor4D.h"

constexpr uint8_t BOARD_MASK = 0b111111111;

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
    uint16_t x_board;
    uint16_t o_board;
    GameState game_state;
    Cell currentPlayer;

    TicTacToe() : x_board(0), o_board(0), game_state(GameState::running), currentPlayer(Cell::x) {}

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
    }

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

    nnm::Tensor4D getInitialState() {
        nnm::Tensor4D tensor(1, 3, 3, 3);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                tensor(0, 2, i, j) = 1;
            }
        }
    }

    [[nodiscard]] std::vector<uint8_t> getLegalMoves() const {
        uint16_t board = ~(x_board | o_board) & BOARD_MASK;
        std::vector<uint8_t> legal_moves;
        while (board) {
            uint8_t move = __builtin_ctz(board);
            board &= board - 1;
            legal_moves.push_back(move);
        }
        return legal_moves;
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

    [[nodiscard]] GameState getGameState() const {
        return game_state;
    }

};