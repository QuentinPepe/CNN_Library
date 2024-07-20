# Find Google Test
find_package(GTest)

if (GTEST_FOUND)
    # Create the test executable
    add_executable(runTests main.cpp
            test_matrix.cpp
            test_vector.cpp
            test_convolutional_layer.cpp
            test_matrix_convolutional.cpp
    )

    # Link the test executable with Google Test and our Game library
    target_link_libraries(runTests
            PRIVATE
            GTest::GTest
            GTest::Main
            GameLib
    )

    # Make sure the compiler can find include files for our GameLib library
    target_include_directories(runTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

    # Discover tests
    include(GoogleTest)
    gtest_discover_tests(runTests)
else ()
    message(WARNING "Google Test not found. Tests will not be built.")
endif ()