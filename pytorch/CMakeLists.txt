add_executable(PyTorch main.cpp
        TicTacToe.h
        MCTS.h)

target_link_libraries(PyTorch
        PRIVATE
        GameLib
)

target_include_directories(PyTorch
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)