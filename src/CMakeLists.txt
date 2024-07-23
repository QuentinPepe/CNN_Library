add_library(GameLib
        Vector.h
        ReLULayer.h
        LinearLayer.h
        MaxPoolingLayer.h
        LossFunctions.h
        BatchNorm2d.h
        Layer.h
        Matrix.h
        ConvolutionalLayer.h
        Tensor4D.h
        FlattenLayer.h
        Tanh.h
        Sequential.h
        Foo.cpp
        ResNet.h
        ResBlock.h
        TicTacToeModel.h
        SoftMaxLayer.h
)

add_executable(CNN main.cpp
)

set_target_properties(GameLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(CNN PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(GameLib PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(CNN PRIVATE GameLib ${TORCH_LIBRARIES})

target_include_directories(GameLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${TORCH_INCLUDE_DIRS})
target_include_directories(CNN PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TORCH_INCLUDE_DIRS})
