    cmake_minimum_required(VERSION 3.10)
project(CNN VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -fopenmp")

# Find SFML
# find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

include(FetchContent)
FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2)
FetchContent_MakeAvailable(json)

add_subdirectory(src)

option(BUILD_TESTS "Build the tests" ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()