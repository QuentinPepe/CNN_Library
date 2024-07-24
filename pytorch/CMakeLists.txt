add_executable(PyTorch main.cpp
        TicTacToe.h
        MCTSLearn.h
        Node.h
        TicTacToeUltimateModel.h
        TicTacToeModel.h
        AlphaZero.h)

target_link_libraries(PyTorch
        PRIVATE
        GameLib
)

target_include_directories(PyTorch
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)