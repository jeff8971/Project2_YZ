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
    printf("  h: use the Histogram method to extract the feature\n");
    printf("  m: use the Multi-histogram method to extract the feature\n");
    printf("  t: use the Texture method to extract the feature\n");
    printf("  c: use the Color method to extract the feature\n");
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <method> <target_image_path> <Top N>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string method = argv[1];
    std::string target_image_path = argv[2];

    int N = 3;  // Default value for N
    if (argc == 4) {
        N = std::stoi(argv[3]);
    }
    if (N < 1) {
        std::cerr << "Error: invalid N" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "N is set to " << N << std::endl;

    // Check if the method is valid
    if (method != "b" && method != "h" && method != "m" && method != "t" && method != "c") {
        std::cerr << "Error: invalid method" << std::endl;
        Menu();
        return EXIT_FAILURE;
    }

    // Construct the full name of the method
    std::string methodFullname;
    if (method == "b") {
        methodFullname = "baseline";
    } else if (method == "h") {
        methodFullname = "histogram";
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

    std::cout << "Method is set to " << methodFullname << std::endl;


    // Construct the CSV file path based on the method
    std::string csvFile = "/Users/jeff/Desktop/Project2_YZ/bin/image_features_" + methodFullname + ".csv";

    // Read the target image and extract its feature vector
    cv::Mat target_image = cv::imread(target_image_path, cv::IMREAD_COLOR);
    if (target_image.empty()) {
        std::cerr << "Could not read the target image: " << target_image_path << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<float> target_features = extract7x7FeatureVector(target_image);

    // Use the existing read_image_data_csv function to read the CSV file
    std::vector<char*> filenames;
    std::vector<std::vector<float>> data;
    if (read_image_data_csv(const_cast<char*>(csvFile.c_str()), filenames, data, 0) != 0) {
        std::cerr << "Failed to read image data from CSV" << std::endl;
        return EXIT_FAILURE;
    }

    // Compute distances between target image and each image in the CSV
    std::vector<std::pair<float, std::string>> distances;
    for (size_t i = 0; i < data.size(); ++i) {
        float distance = computeSSD(target_features, data[i]);
        distances.emplace_back(distance, std::string(filenames[i]));
    }

    // Clean up dynamically allocated filenames
    for (char* fname : filenames) {
        delete[] fname;
    }

    // Sort by distance and output the top N matches
    std::sort(distances.begin(), distances.end());
    std::cout << "Top " << N << " Matches (first is target image itself, 0 to confirm):" << std::endl;
    for (int i = 0; i < (N + 1) && i < distances.size(); ++i) {
        std::cout << distances[i].second << " with distance: " << distances[i].first << std::endl;
    }

    return 0;
}
