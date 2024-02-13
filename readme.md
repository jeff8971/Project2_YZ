# Project2_YZ: CS5330 Project 2 - Content-based Image Retrieval

[Project Repository](https://github.com/jeff8971/Project2_YZ)

## Overview
Project2_YZ is developed as part of the CS5330 Computer Vision course, focusing on Content-based Image Retrieval (CBIR). The project aims to implement a system capable of retrieving images from a database by similarity to a query image. This involves feature extraction, feature matching, and utilizing deep neural network embeddings to facilitate the retrieval process.

## System Environment
- **IDE**: Visual Studio Code or any preferred C++ IDE
- **Compiler**: C++ Compiler supporting C++11 standard
- **System Configuration**: Compatible with various operating systems, including macOS, Linux, and Windows.
- **Dependencies**: Requires OpenCV for image processing and feature extraction functionalities.

## Project Structure
- `src/`: Source files implementing the core functionality of the project.
  - `extractFeature2csv.cpp`: Extracts features from images and saves them in CSV format.
  - `matchings.cpp`: Implements the feature matching logic.
  - `csv_util.cpp`: Utilities for handling CSV files.
  - `csv2matching.cpp`: Converts CSV data to matching pairs.
  - `dnn_embedding.cpp`: Utilizes deep neural network embeddings for image retrieval.
- `include/`: Contains header files for the project.
- `bin/`: Executable files generated after building the project are stored here.
- `CMakeLists.txt`: Configuration file for building the project using CMake.

## Features
- **Feature Extraction and Saving**: Extract features from images and save them in a structured CSV format for further processing.
- **Feature Matching**: Match features between different images to find similarities.
- **Deep Neural Network Embeddings**: Utilize advanced DNN embeddings to improve the accuracy and efficiency of image retrieval.
- **CSV Utilities**: Tools for efficiently handling CSV files containing feature data.

## Getting Started
### Prerequisites
- Install [OpenCV](https://opencv.org/releases/) library.
- Install CMake.
- Ensure C++11 support in your compiler.

### Installation
1. Clone the repository:
```git clone https://github.com/jeff8971/Project2_YZ.git```
2. Navigate to the project directory:```cd Project2_YZ```
3. Build the project using CMake:
```
cd build
cmake ..
make
```

### Running the Application
After building, the project generates three executables for different tasks within the `bin/` directory:
- `./extractFeature`: For extracting features from images and saving them to CSV.
- `./matching`: For performing feature matching between images.
- `./dnn_embedding`: For applying deep neural network embeddings for image retrieval.


### Using `extractFeature`

The `extractFeature` executable is a core component of the Project2_YZ, designed for extracting features from a set of images using various methods. This tool supports multiple feature extraction techniques, allowing for flexibility in how images are processed and analyzed for retrieval.

#### Usage
To use `extractFeature`, navigate to the `bin/` directory after building the project and run the following command:
```./extractFeature <method> <directory_of_images>```

- `<method>`: Specifies the feature extraction method to use.
- `<directory_of_images>`: The path to the directory containing the images from which features will be extracted.

#### Methods
The following methods can be specified for the `<method>` parameter:

- `b`: Baseline method for basic feature extraction.
- `h2`: RG 2D Histogram method for extracting features based on a 2-dimensional histogram of the RG color space.
- `h3`: RGB 3D Histogram method for extracting features using a 3-dimensional histogram of the RGB color space.
- `m`: Multi-histogram method that combines multiple histograms for feature extraction.
- `tc`: Texture and Color method that analyzes both texture and color characteristics of the images.
- `glcm`: GLCM (Gray Level Co-occurrence Matrix) filter for texture feature extraction.
- `l`: Laws' Histogram method for texture analysis based on Laws' texture energy measures.
- `gabor`: Gabor Histogram method for extracting features using Gabor filters.
- `custom`: custom method that integrates Gabor features, edge histograms, Laws' texture features, and color texture features for the purpose of feature extraction.

### Example
To extract features using the RGB 3D Histogram method from images in the `images/` directory, you would run:
`./extractFeature h3 path_of_directory_of_images/`

This command processes all images in the specified directory using the chosen feature extraction method and outputs the results accordingly.


### Using `matching`

The `matching` executable is another vital tool in the Project2_YZ, designed for matching a target image against a dataset of images using various feature comparison methods. This functionality is crucial for the content-based image retrieval process, allowing for the identification of similar images based on extracted features.

#### Usage
To use `matching`, navigate to the `bin/` directory after building the project and execute the command in the following format:
`./matching <method> <path/target_image_name> <Top N>`


- `<method>`: The feature comparison method to be used for matching.
- `<path/target_image_name>`: The path to the target image file that will be compared against the dataset.
- `<Top N>`: The number of top matching results to retrieve, default is `3`.

#### Methods
Specify one of the following methods for the `<method>` parameter to determine how the matching will be performed:

- `b`: Baseline method for straightforward matching.
- `h2`: RG 2D Histogram method for matching based on 2D histograms in the RG color space.
- `h3`: RGB 3D Histogram method for matching using 3D histograms in the RGB color space.
- `m`: Multi-histogram method that utilizes multiple histograms for a comprehensive matching process.
- `tc`: Texture and Color method that considers both texture and color features for matching.
- `glcm`: GLCM (Gray Level Co-occurrence Matrix) filter for texture-based matching.
- `l`: Laws' filter for matching based on Laws' texture energy measures.
- `gabor`: Gabor filter for matching using Gabor filter responses.
- `custom`: A custom method that integrates Gabor features, edge histograms, Laws' texture features, and color texture features for the matching process.

#### Example
To match a target image named `example.jpg` using the RGB 3D Histogram method and retrieve the top 5 matching results, you would run:

`./matching h3 path_of_directory_of_images/example.jpg 5`


This command compares the target image against the dataset using the specified method and outputs the top 5 most similar images.

### Using `dnn_embedding`

The `dnn_embedding` executable is a sophisticated component of Project2_YZ, designed to utilize deep neural network (DNN) embeddings for identifying the top N matching images in a dataset. This tool leverages a pre-computed CSV file containing feature vectors extracted via a DNN (e.g., ResNet18) to perform similarity comparisons.

`dnn_embedding` compares a target image against a dataset of images based on their DNN embeddings. It calculates the cosine similarity between the feature vector of the target image and those of the dataset images, ranking them to find the most similar images.

#### Usage
To use `dnn_embedding`, ensure you are in the `bin/` directory after building the project, then execute the command as follows:

`./dnn_embedding <target_image_name> <Top N>`

- `<target_image_name>`: The name of the target image file you wish to compare against the dataset.
- `<Top N>`: The number of top matching results you wish to retrieve, default is `3`.

#### Prerequisites
A CSV file containing the DNN embeddings of the images in your dataset. The path to this file is typically hardcoded in the source code (e.g., `/Users/jeff/Desktop/Project2_YZ/olympus/ResNet18_olym.csv`). Ensure this file is correctly located and accessible.

### Example
To find the top 3 images most similar to `example.jpg` based on DNN embeddings, run:
`./bin/dnn_embedding example.jpg 3`


This command processes the target image by comparing its feature vector, as found in the specified CSV file, against those of the dataset images. It then outputs the top 3 images with the highest similarity scores.



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Utilization of Travel Days
This project requires three days for completion. The submission deadline is set for February 10th, 11:59:59 PM, while my submission time will be prior to February 13th, 11:59:59 PM.


