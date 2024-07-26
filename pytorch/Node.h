#pragma once

#include "TicTacToe.h"

constexpr int maxChildrenNumber = 9;

struct Node {
    double _reward;
    int _nSims;
    TicTacToe _game;
    int _nMoves;
    float _prior;
    Player _player;  // before playing the move
    Node *_parent;
    int _move;
    std::vector<std::unique_ptr<Node>> _children;

    Node(const Node &) = delete;

    explicit Node(const TicTacToe &game) : _reward(0), _nSims(0), _game(game), _nMoves(game.CountLegalMoves()),
                                           _player(game.getCurrentPlayer()), _parent(nullptr), _prior(0), _move(-1) {
        _children.reserve(maxChildrenNumber);

    }

    Node(const TicTacToe &game, Node *parent, uint8_t move, float prior) : _reward(0), _nSims(0), _game(game),
                                                                           _parent(parent), _prior(prior), _move(move) {
        _player = game.getCurrentPlayer();
        _game.makeMove(move);
        _nMoves = _game.CountLegalMoves();
        _children.reserve(_nMoves);
    }
};
