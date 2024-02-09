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

# Add the first executable that uses extractFeature2csv.cpp and other necessary source files
add_executable(extractFeature ./src/extractFeature2csv.cpp ./src/matchings.cpp ./src/csv_util.cpp)

# Link OpenCV libraries
target_link_libraries(extractFeature ${OpenCV_LIBS})

# Add the second executable that uses matchings.cpp and other necessary source files
#add_executable(matching ./src/matchings.cpp ./src/csv2matching.cpp ./src/csv_util.cpp)

# Link OpenCV libraries with the second executable
#target_link_libraries(matching ${OpenCV_LIBS})

# Ensure the OpenCV include directories are available to all targets
include_directories(${OpenCV_INCLUDE_DIRS})