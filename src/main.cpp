#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>
#include <memory>

using namespace std;

enum Action {
    UP, RIGHT, DOWN, LEFT
};

class MiniGame {
public:
    virtual void reset(mt19937 &rng) = 0;

    virtual string getGPU() = 0;

    virtual vector<int> getRegisters() = 0;

    virtual void tick(const vector<Action> &actions) = 0;

    virtual bool isGameOver() = 0;

    virtual vector<int> getRankings() = 0;

    virtual string getName() = 0;
};

class HurdleRace : public MiniGame {
private:
    string map;
    vector<int> positions;
    vector<int> stunTimers;
    vector<bool> dead;
    vector<bool> jumped;
    vector<int> finished;
    int rank;

    const int STUN_DURATION = 2;

public:
    HurdleRace() : positions(3), stunTimers(3), dead(3, false), jumped(3, false), finished(3, -1), rank(0) {}

    void reset(mt19937 &rng) override {
        uniform_int_distribution<> distStart(3, 7);
        uniform_int_distribution<> distHurdles(3, 6);
        uniform_int_distribution<> distBool(0, 1);

        int startStretch = distStart(rng);
        int hurdles = distHurdles(rng);
        int length = 30;

        map.clear();
        map.append(startStretch, '.');
        for (int i = 0; i < hurdles; ++i) {
            map += distBool(rng) ? "#...." : "#...";
        }
        map.append(length - map.length(), '.');
        map.back() = '.';

        fill(positions.begin(), positions.end(), 0);
        fill(stunTimers.begin(), stunTimers.end(), 0);
        fill(finished.begin(), finished.end(), -1);
        fill(dead.begin(), dead.end(), false);
        fill(jumped.begin(), jumped.end(), false);
        rank = 0;
    }

    string getGPU() override { return map; }

    vector<int> getRegisters() override {
        vector<int> registers(8, -1);
        copy(positions.begin(), positions.end(), registers.begin());
        copy(stunTimers.begin(), stunTimers.end(), registers.begin() + 3);
        return registers;
    }

    void tick(const vector<Action> &actions) override {
        int maxX = map.length() - 1;
        int countFinishes = 0;

        for (int i = 0; i < 3; ++i) {
            jumped[i] = false;

            if (actions[i] == Action::UP) stunTimers[i] = max(0, stunTimers[i] - 1);
            if (stunTimers[i] > 0 || finished[i] > -1) continue;

            int moveBy = 0;
            bool jump = false;

            switch (actions[i]) {
                case Action::DOWN:
                    moveBy = 2;
                    break;
                case Action::LEFT:
                    moveBy = 1;
                    break;
                case Action::RIGHT:
                    moveBy = 3;
                    break;
                case Action::UP:
                    moveBy = 2;
                    jump = true;
                    jumped[i] = true;
                    break;
            }

            for (int x = 0; x < moveBy; ++x) {
                positions[i] = min(maxX, positions[i] + 1);
                if (map[positions[i]] == '#' && !jump) {
                    stunTimers[i] = STUN_DURATION;
                    break;
                }
                if (positions[i] == maxX && finished[i] == -1) {
                    finished[i] = rank;
                    countFinishes++;
                    break;
                }
                jump = false;
            }
        }
        rank += countFinishes;
    }

    bool isGameOver() override {
        int count = 0;
        for (int i = 0; i < 3; ++i) {
            if (finished[i] > -1) return true;
            if (finished[i] > -1 || dead[i]) count++;
        }
        return count >= 2;
    }

    vector<int> getRankings() override {
        vector<int> rankings(3);
        for (int i = 0; i < 3; ++i) {
            rankings[i] = (finished[i] == -1) ? rank : finished[i];
        }
        return rankings;
    }

    string getName() override { return "Hurdle Race"; }
};

class Archery : public MiniGame {
private:
    vector<vector<int>> cursors;
    vector<int> wind;
    vector<bool> dead;

public:
    Archery() : cursors(3, vector<int>(2)), dead(3, false) {}

    void reset(mt19937 &rng) override {
        uniform_int_distribution<> distPos(5, 9);
        uniform_int_distribution<> distSign(0, 1);
        uniform_int_distribution<> distRounds(12, 15);
        uniform_int_distribution<> distWind(0, 9);

        int x = distPos(rng) * (distSign(rng) ? 1 : -1);
        int y = distPos(rng) * (distSign(rng) ? 1 : -1);
        for (auto &cursor: cursors) {
            cursor[0] = x;
            cursor[1] = y;
        }

        wind.clear();
        int rounds = distRounds(rng);
        for (int i = 0; i < rounds; ++i) {
            wind.push_back(distWind(rng));
        }

        fill(dead.begin(), dead.end(), false);
    }

    string getGPU() override {
        string gpu;
        for (int w: wind) gpu += to_string(w);
        return gpu;
    }

    vector<int> getRegisters() override {
        vector<int> registers(8, -1);
        for (int i = 0; i < 3; ++i) {
            registers[i * 2] = cursors[i][0];
            registers[i * 2 + 1] = cursors[i][1];
        }
        return registers;
    }

    void tick(const vector<Action> &actions) override {
        int offset = wind[0];
        for (int i = 0; i < 3; ++i) {
            if (actions[i] == Action::UP) {
                dead[i] = true;
                continue;
            }

            int dx = 0, dy = 0;
            switch (actions[i]) {
                case Action::DOWN:
                    dy = offset;
                    break;
                case Action::LEFT:
                    dx = -offset;
                    break;
                case Action::RIGHT:
                    dx = offset;
                    break;
                case Action::UP:
                    dy = -offset;
                    break;
            }

            cursors[i][0] = clamp(cursors[i][0] + dx, -20, 20);
            cursors[i][1] = clamp(cursors[i][1] + dy, -20, 20);
        }
        wind.erase(wind.begin());
    }

    bool isGameOver() override { return wind.empty(); }

    vector<int> getRankings() override {
        vector<pair<int, double>> scores;
        for (int i = 0; i < 3; ++i) {
            double distance = sqrt(pow(cursors[i][0], 2) + pow(cursors[i][1], 2));
            scores.emplace_back(i, dead[i] ? numeric_limits<double>::max() : distance);
        }
        sort(scores.begin(), scores.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

        vector<int> rankings(3);
        for (int i = 0; i < 3; ++i) {
            rankings[scores[i].first] = i;
        }
        return rankings;
    }

    string getName() override { return "Archery"; }
};

class Diving : public MiniGame {
private:
    string goal;
    vector<int> points;
    vector<int> combo;
    vector<bool> dead;
    int turnsRemaining;

public:
    Diving() : points(3, 0), combo(3, 0), dead(3, false), turnsRemaining(0) {}

    void reset(mt19937 &rng) override {
        uniform_int_distribution<> distLength(12, 15);
        uniform_int_distribution<> distAction(0, 3);

        int length = distLength(rng);
        goal.clear();
        for (int i = 0; i < length; ++i) {
            goal += "UDLR"[distAction(rng)];
        }

        fill(points.begin(), points.end(), 0);
        fill(combo.begin(), combo.end(), 0);
        fill(dead.begin(), dead.end(), false);
        turnsRemaining = length + 1;
    }

    string getGPU() override { return goal; }

    vector<int> getRegisters() override {
        vector<int> registers(8, -1);
        copy(points.begin(), points.end(), registers.begin());
        copy(combo.begin(), combo.end(), registers.begin() + 3);
        return registers;
    }

    void tick(const vector<Action> &actions) override {
        for (int i = 0; i < 3; ++i) {
            if (actions[i] == Action::UP) {
                dead[i] = true;
                continue;
            }

            if (goal[0] == "UDLR"[static_cast<int>(actions[i])]) {
                combo[i]++;
                points[i] += combo[i];
            } else {
                combo[i] = 0;
            }
        }
        goal.erase(0, 1);
        turnsRemaining--;
    }

    bool isGameOver() override { return goal.empty(); }

    vector<int> getRankings() override {
        vector<pair<int, int>> scores;
        for (int i = 0; i < 3; ++i) {
            scores.emplace_back(i, dead[i] ? -1 : points[i]);
        }
        sort(scores.begin(), scores.end(), [](const auto &a, const auto &b) { return a.second > b.second; });

        vector<int> rankings(3);
        for (int i = 0; i < 3; ++i) {
            rankings[scores[i].first] = i;
        }
        return rankings;
    }

    string getName() override { return "Diving"; }
};

class RollerSpeedSkating : public MiniGame {
private:
    vector<int> positions;
    vector<int> risk;
    vector<bool> dead;
    vector<Action> directions;
    int length, timer;

public:
    RollerSpeedSkating() : positions(3, 0), risk(3, 0), dead(3, false), length(10), timer(15) {
        directions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
    }

    void reset(mt19937 &rng) override {
        fill(positions.begin(), positions.end(), 0);
        fill(risk.begin(), risk.end(), 0);
        fill(dead.begin(), dead.end(), false);
        shuffle(directions.begin(), directions.end(), rng);
        timer = 15;
    }

    string getGPU() override {
        string gpu;
        for (Action dir: directions) {
            gpu += "UDLR"[static_cast<int>(dir)];
        }
        return gpu;
    }

    vector<int> getRegisters() override {
        vector<int> registers(8, -1);
        copy(positions.begin(), positions.end(), registers.begin());
        copy(risk.begin(), risk.end(), registers.begin() + 3);
        registers[6] = timer;
        return registers;
    }

    void tick(const vector<Action> &actions) override {
        for (int i = 0; i < 3; ++i) {
            if (actions[i] == Action::UP) {
                dead[i] = true;
                continue;
            }

            if (risk[i] < 0) {
                risk[i]++;
                continue;
            }

            int idx = find(directions.begin(), directions.end(), actions[i]) - directions.begin();
            int dx = (idx == 0) ? 1 : (idx == 3) ? 3 : 2;

            positions[i] += dx;
            int riskValue = -1 + idx;
            risk[i] = max(0, risk[i] + riskValue);
        }

        for (int i = 0; i < 3; ++i) {
            if (risk[i] < 0) continue;

            bool clash = false;
            for (int k = 0; k < 3; ++k) {
                if (k == i) continue;
                if (positions[k] % length == positions[i] % length) {
                    clash = true;
                    break;
                }
            }
            if (clash) risk[i] += 2;

            if (risk[i] >= 5) risk[i] = -2; // stun
        }

        shuffle(directions.begin(), directions.end(), mt19937(time(0)));
        timer--;
    }

    bool isGameOver() override { return timer <= 0; }

    vector<int> getRankings() override {
        vector<pair<int, int>> scores;
        for (int i = 0; i < 3; ++i) {
            scores.emplace_back(i, dead[i] ? -1 : positions[i]);
        }
        sort(scores.begin(), scores.end(), [](const auto &a, const auto &b) { return a.second > b.second; });

        vector<int> rankings(3);
        for (int i = 0; i < 3; ++i) {
            rankings[scores[i].first] = i;
        }
        return rankings;
    }

    string getName() override { return "Roller Speed Skating"; }
};

class Game {
private:
    int playerIdx;
    vector<unique_ptr<MiniGame>> games;
    vector<vector<int>> medals;
    mt19937 rng;

public:
    Game(int playerIdx) : playerIdx(playerIdx), medals(4, vector<int>(3, 0)), rng(time(0)) {
        games.push_back(make_unique<HurdleRace>());
        games.push_back(make_unique<Archery>());
        games.push_back(make_unique<Diving>());
        games.push_back(make_unique<RollerSpeedSkating>());

        for (auto &game: games) {
            game->reset(rng);
        }
    }

    void run() {
        while (true) {
            vector<string> scoreInfo(3);
            for (int i = 0; i < 3; i++) {
                int score = 1;
                for (int j = 0; j < 4; j++) {
                    score *= medals[j][1] + medals[j][0] * 3;
                }
                scoreInfo[i] = to_string(score);
                for (int j = 0; j < 4; j++) {
                    scoreInfo[i] += " " + to_string(medals[j][0]) + " " + to_string(medals[j][1]) + " " +
                                    to_string(medals[j][2]);
                }
            }

            for (const auto &info: scoreInfo) {
                cout << info << endl;
            }

            bool allGameOver = true;
            for (const auto &game: games) {
                cout << game->getGPU();
                auto registers = game->getRegisters();
                for (int reg: registers) {
                    cout << " " << reg;
                }
                cout << endl;

                if (!game->isGameOver()) {
                    allGameOver = false;
                }
            }

            if (allGameOver) {
                break;
            }

            string action;
            cin >> action;

            vector<Action> actions(3);
            if (action == "UP") actions[playerIdx] = Action::UP;
            else if (action == "DOWN") actions[playerIdx] = Action::DOWN;
            else if (action == "LEFT") actions[playerIdx] = Action::LEFT;
            else if (action == "RIGHT") actions[playerIdx] = Action::RIGHT;

            for (int i = 0; i < 3; i++) {
                if (i != playerIdx) {
                    actions[i] = static_cast<Action>(rand() % 4);
                }
            }

            for (auto &game: games) {
                if (!game->isGameOver()) {
                    game->tick(actions);
                    if (game->isGameOver()) {
                        auto rankings = game->getRankings();
                        for (int i = 0; i < 3; i++) {
                            medals[&game - &games[0]][rankings[i]]++;
                        }
                    }
                }
            }
        }
    }
};

int main() {
    int playerIdx, nbGames;
    cin >> playerIdx >> nbGames;

    Game game(playerIdx);
    game.run();

    return 0;
}