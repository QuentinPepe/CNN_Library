add_library(GameLib
        Vector.h
        ConvolutionalLayer.cpp
        ReLULayer.cpp
        ReLULayer.h
        PoolingLayer.cpp
        PoolingLayer.h
        FullyConnectedLayer.cpp
        FullyConnectedLayer.h
)

add_executable(CNN main.cpp
        Layer.h
        Matrix.h
        ConvolutionalLayer.h
)

target_link_libraries(CNN PRIVATE GameLib)

target_include_directories(GameLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(CNN PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})