# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.10)

# Project's name and language
project(Project2_YZ CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the output directories for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Can automatically find and configure OpenCV or other libraries if needed
find_package(OpenCV REQUIRED)

# Add both main.cpp and matchings.cpp to the executable
add_executable(Project2_YZ ./src/extractImgFeature2csv.cpp ./src/matchings.cpp)
# test
# add_executable(test ./src/test.cpp ./src/csv_util.cpp)


# Link OpenCV libraries
target_link_libraries(Project2_YZ ${OpenCV_LIBS})
# test
# target_link_libraries(test ${OpenCV_LIBS})

# Ensure the OpenCV include directories are available to all targets
include_directories(${OpenCV_INCLUDE_DIRS})