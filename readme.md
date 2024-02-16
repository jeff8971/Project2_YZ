# Project2_YZ: CS5330 Project 2 - Content-based Image Retrieval

[Project Repository](https://github.com/jeff8971/Project2_YZ)

## Overview
Project2_YZ is developed as part of the CS5330 Computer Vision course, focusing on Content-based Image Retrieval (CBIR). The project aims to implement a system capable of retrieving images from a database by similarity to a query image. This involves feature extraction, feature matching, utilizing deep neural network embeddings, and face detecting function to facilitate the retrieval process.

## System Environment
- **IDE**: Visual Studio Code or any preferred C++ IDE
- **Compiler**: C++ Compiler supporting C++11 standard
- **System Configuration**: Compatible with various operating systems, including macOS, Linux, and Windows.
- **Dependencies**: Requires OpenCV for image processing and feature extraction functionalities.

## Project Structure
- `src/`: Source files implementing the core functionality of the project.
  - `extractFeature2csv.cpp`: Extracts features from images and saves them in CSV format in `./bin`.
  - `matchings.cpp`: Implements the feature matching logic.
  - `csv_util.cpp`: Utilities for handling CSV files.
  - `csv2matching.cpp`: Converts CSV data to matching pairs.
  - `dnn_embedding.cpp`: Utilizes deep neural network embeddings for image retrieval.
  - `faceDetect.cpp`: Project 1 code, face detect functions.
  - `faceDetecting.cpp`: Main entry for face detect for all the images.
- `include/`: Contains header files for the project.
- `bin/`: Face detecting feature `.xml`, and executable files generated after building the project are stored here.
- `CMakeLists.txt`: Configuration file for building the project using CMake.
- `data`: Contain results screenshots.
- `olympus`: Directory of images, with sub-directories dedicated to various analytical methods and an image database for comprehensive processing and analysis.

## Features
- **Feature Extraction and Saving**: Extract features from images and save them in a structured CSV format for further processing.
- **Feature Matching**: Match features between different images to find similarities.
- **Deep Neural Network Embeddings**: Utilize advanced DNN embeddings to improve the accuracy and efficiency of image retrieval.
- **Face Detecting**: Utilize face box function to get all the images with face features.
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
- `./faceDetecting`: To display the results of all images identified as containing facial features.


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
- `gabor`: Extracting features using Gabor filters method.
- `glcm`: Extracting features using Gray-Level Co-occurrence Matrix method (GLCM).
- `custom_s`/`custom_m`/`custom_l`: custom methods that emphasizes the weighting of different parts of an image to enhance the detection of small/medium/large objects within it.
- `face`: extract face features from the directory of the images.

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
- `l`: Laws' filter for matching based on Laws' texture energy measures.
- `glcm`: GLCM (Gray Level Co-occurrence Matrix) method for texture-based matching.
- `gabor`: Gabor filter for matching using Gabor filter responses.
- `custom_s`/`custom_m`/`custom_l`: Custom methods that emphasizes the weighting of different parts of an image to enhance the matching the small/medium/large objects within it.

#### Example
To match a target image named `example.jpg` using the RGB 3D Histogram method and retrieve the top 5 matching results, you would run:

`./matching h3 path_of_directory_of_images/example.jpg 5`


### Using `dnn_embedding`

The `dnn_embedding` executable is an advanced feature of Project2_YZ, crafted to harness deep neural network (DNN) embeddings to ascertain the top N matching images within a dataset. This utility capitalizes on a pre-processed CSV file that houses feature vectors derived from a DNN model (for instance, ResNet18), facilitating similarity assessments. It executes this by computing the cosine similarity between the feature vector of a target image and the feature vectors from the dataset, subsequently ranking these images to identify those with the highest similarity.

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

### Using `faceDetecting`

The `faceDetecting` component of Project2_YZ is designed to identify images within a dataset that contain face features. This functionality leverages a CSV file generated by `extractFeature2csv.cpp`, which should contain feature vectors indicative of face presence in images.

#### Overview
`faceDetecting` reads through the specified CSV file, analyzing the feature data associated with each image. It identifies images that have been marked with face features, indicating the presence of faces within those images.

#### Usage
To use `faceDetecting`, compile the project including the `faceDetecting.cpp` source file, and execute the resulting binary. The tool expects the path to a CSV file as an input, which contains the image features extracted by `extractFeature2csv.cpp`.

#### Prerequisites
A CSV file containing image features, specifically indicating the presence of face features. This file is typically generated by the `extractFeature2csv.cpp` tool as part of the project's feature extraction process.

#### Example Command
Assuming the CSV file is named `image_features_face.csv` and located in the `bin/` directory, the command to run `faceDetecting` might look like this:
`./bin/faceDetecting`


This command will process the `image_features_face.csv` file, outputting the names of images that contain face features.

### Note
- Ensure that the path to the CSV file within `faceDetecting.cpp` is correctly set to match the location of your `image_features_face.csv` file. The default path is set as `/Users/jeff/Desktop/Project2_YZ/bin/image_features_face.csv`, which may need to be adjusted according to your project setup.
- The effectiveness of face detection depends on the accuracy of the feature extraction process. Ensure that the `extractFeature2csv.cpp` tool is correctly implemented and configured to identify face features within images.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Utilization of Travel Days
This project requires three days for completion. The submission deadline is set for February 10th, 11:59:59 PM, while my submission time will be prior to February 13th, 11:59:59 PM.


