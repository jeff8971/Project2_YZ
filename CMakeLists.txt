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
add_executable(extractFeature ./src/extractFeature2csv.cpp ./src/matchings.cpp ./src/csv_util.cpp ./src/faceDetect.cpp)

# Link OpenCV libraries
target_link_libraries(extractFeature ${OpenCV_LIBS})

# Add the second executable that uses matchings.cpp and other necessary source files
add_executable(matching ./src/matchings.cpp ./src/csv2matching.cpp ./src/csv_util.cpp ./src/faceDetect.cpp)

# Link OpenCV libraries with the second executable
target_link_libraries(matching ${OpenCV_LIBS})

# Add the third executable that uses dnn_embedding.cpp and other necessary source files
add_executable(dnn_embedding ./src/dnn_embedding.cpp ./src/matchings.cpp ./src/csv_util.cpp)

# Link OpenCV libraries with the third executable
target_link_libraries(dnn_embedding ${OpenCV_LIBS})

# Add the fourth executable that uses faceDetect.cpp and other necessary source files
add_executable(faceDetecting ./src/faceDetecting.cpp)

# Link OpenCV libraries with the fourth executable
target_link_libraries(faceDetecting ${OpenCV_LIBS})

# Ensure the OpenCV include directories are available to all targets
include_directories(${OpenCV_INCLUDE_DIRS})