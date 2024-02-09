/**
 * @file csv2matching.cpp
 * @author Yuan Zhao (zhao.yuan2@northeatern.edu)
 * @brief use the target image, and csv file's features data to find the Top N matching images
 * @version 0.1
 * @date 2024-02-03
*/


#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "matchings.h"
#include "csv_util.h"


// Menu for the user
void Menu(){
    printf("Usage: ./matching <method> <target_image_name> <Top N>\n");
    printf("method:\n");
    printf("  b: use the Baseline method to extract the feature\n");
    printf("  h2: use the RG 2D Histogram method to extract the feature\n");
    printf("  h3: use the RGB 3D Histogram method to extract the feature\n");
    printf("  m: use the Multi-histogram method to extract the feature\n");
    printf("  t: use the Texture method to extract the feature\n");
    printf("  c: use the Color method to extract the feature\n");
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        Menu();
        return EXIT_FAILURE;
    }

    std::string method = argv[1];
    // Check if the method is valid
    if (method != "b" && method != "h2" && method != "h3" && method != "m" && method != "t" && method != "c") {
        std::cerr << "Error: invalid method" << std::endl;
        Menu();
        return EXIT_FAILURE;
    }

    // Set the target image path from the command line, 2nd argument
    std::string target_image_path = argv[2];

    // Set the value for N
    int N = 3;  // Default value for N
    if (argc == 4) {
        N = std::stoi(argv[3]);
    }
    if (N < 1) {
        std::cerr << "Error: invalid N" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "N is set to " << N << std::endl;

    // Construct the full name of the method
    std::string methodFullname;
    if (method == "b") {
        methodFullname = "baseline";
    } else if (method == "h2") {
        methodFullname = "2D_histogram";
    } else if (method == "h3") {
        methodFullname = "3D_histogram";
    } else if (method == "m") {
        methodFullname = "multi_histogram";
    } else if (method == "t") {
        methodFullname = "texture";
    } else if (method == "c") {
        methodFullname = "color";
    } else {
        std::cerr << "Error: invalid method" << std::endl;
        return EXIT_FAILURE;
    }
    // Print the method
    std::cout << "Method is set to " << methodFullname << std::endl;

    // Construct the CSV file path based on the method
    std::string csvFile = "/Users/jeff/Desktop/Project2_YZ/bin/image_features_" + methodFullname + ".csv";

    // Read the target image and extract its feature vector
    cv::Mat target_image = cv::imread(target_image_path, cv::IMREAD_COLOR);
    // if target image is not exist
    if (target_image.empty()) {
        std::cerr << "Could not read the target image: " << target_image_path << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<float> target_features;
    if (method == "b"){
        target_features = extract7x7FeatureVector(target_image);
    } else if (method == "h2"){
        target_features = calculateRG_2DChromaHistogram(target_image, BINS_2D);
    } else if (method == "h3"){
        target_features = calculateRGB_3DChromaHistogram(target_image, BINS_3D);
    }
    /*
    else if (method == "m"){
        std::vector<float> target_features = calculateMultiRGChromaHistogram(target_image, 8);
    } else if (method == "t"){
        std::vector<float> target_features = calculateTextureFeatureVector(target_image);
    } else if (method == "c"){
        std::vector<float> target_features = calculateColorFeatureVector(target_image);
    }*/ else {
        std::cerr << "Error: invalid method" << std::endl;
        return EXIT_FAILURE;
    }

    // Use the existing read_image_data_csv function to read the CSV file
    std::vector<char*> filenames;
    std::vector<std::vector<float>> data;
    if (read_image_data_csv(const_cast<char*>(csvFile.c_str()), filenames, data, 0) != 0) {
        std::cerr << "Failed to read image data from CSV" << std::endl;
        return EXIT_FAILURE;
    }

    // Compute distances between target image and each image in the CSV
    std::vector<std::pair<float, std::string>> distances;
    if (method == "b"){
        for (size_t i = 0; i < data.size(); i++) {
            float distance = computeSSD(target_features, data[i]);
            distances.emplace_back(distance, std::string(filenames[i]));
        }
    } else if (method == "h2"){
        for (size_t i = 0; i < data.size(); i++) {
            float distance = computeHistogramIntersection(target_features, data[i]);
            distances.emplace_back(distance, std::string(filenames[i]));
        }
    } else if (method == "h3"){
        for (size_t i = 0; i < data.size(); i++) {
            float distance = computeHistogramIntersection(target_features, data[i]);
            distances.emplace_back(distance, std::string(filenames[i]));
        }
    }

    // Clean up dynamically allocated filenames
    for (char* fname : filenames) {
        delete[] fname;
    }

    if (method == "h2" || method == "h3") {
        // Sort in descending order for histogram intersection
        std::sort(distances.begin(), distances.end(), [](const std::pair<float, std::string>& a, const std::pair<float, std::string>& b) {
            return a.first > b.first; // For higher intersection values
        });
    } else {
        // Sort in ascending order for SSD
        std::sort(distances.begin(), distances.end());
    }

    std::cout << "Top " << N << " Matches: " << std::endl;
    // Start loop from 1 to skip the target image, assuming it's the first match
    int matchesToShow = N + 1; // Increase by one to account for skipping the target image
    for (int i = 1; i < matchesToShow && i < distances.size(); i++) {
        std::cout << distances[i].second << " with distance: " << distances[i].first << std::endl;
    }

    return 0;
}
