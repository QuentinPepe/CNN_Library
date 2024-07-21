add_library(GameLib
        Vector.h
        ReLULayer.h
        LinearLayer.h
        MaxPoolingLayer.h
        FullyConnectedLayer.h
        LossFunctions.h
        BatchNorm2d.h
        Layer.h
        Matrix.h
        ConvolutionalLayer.h
        Tensor4D.h
        FlattenLayer.h
        Tanh.h
        Sequential.h
        Tensor4D.cpp
)

add_executable(CNN main.cpp
)

target_link_libraries(CNN PRIVATE GameLib)

target_include_directories(GameLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(CNN PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})