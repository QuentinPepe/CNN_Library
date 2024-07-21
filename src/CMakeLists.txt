add_library(GameLib
        Vector.h
        ConvolutionalLayer.cpp
        BatchNormalizationLayer.h
        ReLULayer.h
        LinearLayer.h
        MaxPoolingLayer.h
        FullyConnectedLayer.h
        LossFunctions.h
        BatchNorm2d.h
)

add_executable(CNN main.cpp
        Layer.h
        Matrix.h
        ConvolutionalLayer.h
        Tensor4D.h
)

target_link_libraries(CNN PRIVATE GameLib)

target_include_directories(GameLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(CNN PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})