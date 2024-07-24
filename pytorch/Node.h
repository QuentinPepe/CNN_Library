#pragma once

#include <cstdlib>
#include <new>
#include "TicTacToe.h"

constexpr int maxChildrenNumber = 9;

struct Node {
    Node(TicTacToe *game, Node *parent, uint8_t move, float probability = 0.0f)
            : _game(game), _parent(parent), _children_memory(nullptr),
              _max_children(maxChildrenNumber), _child_count(0), _move(move), _reward(0), _visits(1),
              _probability(probability) {
        _children_memory = static_cast<char *>(std::malloc(sizeof(Node) * _max_children));
        if (!_children_memory) {
            throw std::bad_alloc();
        }
    }

    ~Node() {
        /*
        delete _game;
        for (int i = 0; i < _child_count; ++i) {
            getChild(i)->~Node();
        }
        std::free(_children_memory);
         */
    }

    Node *addChild(TicTacToe *childGame, uint8_t move, float probability) {
        if (_child_count >= _max_children) {
            throw std::runtime_error("Max children reached");
        }

        Node *new_child = new(_children_memory + _child_count * sizeof(Node)) Node(childGame, this, move, probability);
        _child_count++;

        return new_child;
    }

    [[nodiscard]] Node *getChild(int index) const {
        if (index < 0 || index >= _child_count) {
            return nullptr;
        }
        return reinterpret_cast<Node *>(_children_memory + index * sizeof(Node));
    }

    TicTacToe *getGame() {
        return _game;
    }

    void updateStats(float amount) {
        _reward += amount;
        _visits++;
    }

    Node *getParent() {
        return _parent;
    }

    int getChildCount() const {
        return _child_count;
    }

    uint8_t getMove() const {
        return _move;
    }

    int getVisits() const {
        return _visits;
    }

    int getReward() const {
        return _reward;
    }

    float getProbability() const {
        return _probability;
    }

private:
    TicTacToe *_game;
    Node *_parent;

    char *_children_memory;
    int _max_children;
    int _child_count;

    uint8_t _move;
    int _visits;
    float _reward;
    float _probability;
};