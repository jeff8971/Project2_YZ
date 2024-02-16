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


// matchingMenu for the user
void matchingMenu(){
    printf("Usage: ./matching <method> <path/target_image_name> <Top N>\n");
    printf("method:\n");
    printf("  b: use the Baseline method to matching\n");
    printf("  h2: use the RG 2D Histogram method to matching\n");
    printf("  h3: use the RGB 3D Histogram method to matching\n");
    printf("  m: use the Multi-histogram method to matching\n");
    printf("  tc: use the Texture and Color method to matching\n");
    printf("  glcm: use the GLCM filter to matching\n");
    printf("  l: use the Laws' filter to matching\n");
    printf("  gabor: use the Gabor filter to matching\n");
    printf("  custom_s: use the custom_s method to matching the small object\n");
    printf("  custom_m: use the custom_m method to matching the medium object\n");
    printf("  custom_l: use the custom_l method to matching the large object\n");
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        matchingMenu();
        return EXIT_FAILURE;
    }

    std::string method = argv[1];
    // Check if the method is valid
    if (method != "b" 
    && method != "h2" 
    && method != "h3" 
    && method != "m" 
    && method != "tc" 
    && method != "glcm"
    && method != "l"
    && method != "gabor"
    && method != "custom_s"
    && method != "custom_m"
    && method != "custom_l"
    && method != "face") {
        std::cerr << "Error: invalid method" << std::endl;
        matchingMenu();
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
    std::string embeddingCvsFile;
    if (method == "b") {
        methodFullname = "baseline";
    } else if (method == "h2") {
        methodFullname = "2D_histogram";
    } else if (method == "h3") {
        methodFullname = "3D_histogram";
    } else if (method == "m") {
        methodFullname = "multi_histogram";
    } else if (method == "tc") {
        methodFullname = "texturecolor";
    } else if (method == "glcm") {
        methodFullname = "glcm";
    } else if (method == "l"){
        methodFullname = "laws";
    } else if (method == "gabor"){
        methodFullname = "gabor";
    } else if (method == "custom_s") {
        methodFullname = "custom_s";
    } else if (method == "custom_m") {
        methodFullname = "custom_m";
    } else if (method == "custom_l") {
        methodFullname = "custom_l";
    } else {
        std::cerr << "Error: invalid method" << std::endl;
        return EXIT_FAILURE;
    }
    // Print the method
    std::cout << "Method is set to " << methodFullname << std::endl;

    // Construct the CSV file path based on the method
    std::string csvFile = "/Users/jeff/Desktop/Project2_YZ/bin/image_features_" + methodFullname + ".csv";
    std::cout << "CSV file is set to " << csvFile << std::endl;


    // Read the target image and extract its feature vector
    cv::Mat target_image = cv::imread(target_image_path, cv::IMREAD_COLOR);
    // if target image is not exist
    if (target_image.empty()) {
        std::cerr << "Could not read the target image: " << target_image_path << std::endl;
        return EXIT_FAILURE;
    }

    // Extract the target features, csv files and compare them
    std::vector<float> target_features;
    std::vector<char*> filenames;
    std::vector<std::vector<float>> data;

    // Use the existing read_image_data_csv function to read the CSV file
    if (read_image_data_csv(const_cast<char*>(csvFile.c_str()), filenames, data, false) != 0) {
        std::cerr << "Failed to read image data from CSV" << std::endl;
        return EXIT_FAILURE;
    }

    // Extract the target features based on the method
    if (method == "b"){
        target_features = extract7x7FeatureVector(target_image);
    } else if (method == "h2"){
        target_features = calculateRG_2DChromaHistogram(target_image, BINS_2D);
    } else if (method == "h3"){
        target_features = calculateRGB_3DChromaHistogram(target_image, BINS_3D);
    } else if (method == "m") {
        target_features = calculateMultiPartRGBHistogram(target_image, BINS_3D); // Multi-histogram method
    } else if (method == "tc"){
        target_features = calculateColorTextureFeatureVector(target_image, COLOR_BINS, TEXTURE_BINS);
    } else if (method == "glcm"){
        target_features = calculateGLCMFeatures(target_image, GLCM_DISTANCE, GLCM_ANGLE, GLCM_LEVELS);
    } else if (method == "l"){
        target_features = calculateLawsTextureFeatures(target_image);
    } else if (method == "gabor"){
        target_features = computeGaborFeatures(target_image);
    } else if (method == "custom_s") {
        target_features = calculateCustomFeature(target_image, BINS_3D, WEIGHT_CONFIG_S);
    } else if (method == "custom_m") {
        target_features = calculateCustomFeature(target_image, BINS_3D, WEIGHT_CONFIG_M);
    } else if (method == "custom_l") {
        target_features = calculateCustomFeature(target_image, BINS_3D, WEIGHT_CONFIG_L);
    } else {
        std::cerr << "Error: invalid method" << std::endl;
        return EXIT_FAILURE;
    }

    // Compute similarities between target image and each image in the CSV
    std::vector<std::pair<float, std::string>> similarities;
    if (method == "b"){
        for (size_t i = 0; i < data.size(); i++) {
            float distance = computeSSD(target_features, data[i]);
            similarities.emplace_back(distance, std::string(filenames[i]));
        }
    } else if (method == "h2"){
        for (size_t i = 0; i < data.size(); i++) {
            float distance = computeHistogramIntersection(target_features, data[i]);
            similarities.emplace_back(distance, std::string(filenames[i]));
        }
    } else if (method == "h3"){
        for (size_t i = 0; i < data.size(); i++) {
            float distance = computeHistogramIntersection(target_features, data[i]);
            similarities.emplace_back(distance, std::string(filenames[i]));
        }
    } else if (method == "m"){
        // Compute similarities using combined histogram intersection
        for (size_t i = 0; i < data.size(); i++) {
            float distance = combinedHistogramIntersection(target_features, data[i], SPLIT_POINT);
            // Store the inverted distance for consistency with other methods
            similarities.emplace_back(distance, std::string(filenames[i]));
        }
    } else if (method == "tc"){
        for (size_t i = 0; i < data.size(); i++) {
            float distance = combinedHistogramIntersection(target_features, data[i], SPLIT_POINT);
            similarities.emplace_back(distance, std::string(filenames[i]));
        }
    } else if (method == "glcm"){
        for (size_t i = 0; i < data.size(); i++) {
            float distance = computeSSD(target_features, data[i]);
            similarities.emplace_back(distance, std::string(filenames[i]));
        }
    } else if (method == "l"){
        for (size_t i = 0; i < data.size(); i++) {
            float distance = computeSSD(target_features, data[i]);
            similarities.emplace_back(distance, std::string(filenames[i]));
        }
    } else if (method == "gabor"){
        for (size_t i = 0; i < data.size(); i++) {
            float distance = computeSSD(target_features, data[i]);
            similarities.emplace_back(distance, std::string(filenames[i]));
        }
    } else if (method == "custom_s") {
        for (size_t i = 0; i < data.size(); i++) {
            float distance = computeHistogramIntersection(target_features, data[i]);
            similarities.emplace_back(distance, std::string(filenames[i]));
        }
    } else if (method == "custom_m") {
        for (size_t i = 0; i < data.size(); i++) {
            float distance = computeHistogramIntersection(target_features, data[i]);
            similarities.emplace_back(distance, std::string(filenames[i]));
        }
    } else if (method == "custom_l") {
        for (size_t i = 0; i < data.size(); i++) {
            float distance = computeHistogramIntersection(target_features, data[i]);
            similarities.emplace_back(distance, std::string(filenames[i]));
        }
    } else if (method == "face"){
        printf("face detect done.\n");
    } else {
        std::cerr << "Error: invalid method" << std::endl;
        return EXIT_FAILURE;
    }

    // Clean up dynamically allocated filenames
    for (char* fname : filenames) {
        delete[] fname;
    }
    // by using histogram intersection, the higher the value, the more similar the images are
    if (method == "h2" || method == "h3" || method == "m" || method == "tc" || method == "custom_s" || method == "custom_m" || method == "custom_l") {
        // Sort in descending order for histogram intersection
        std::sort(similarities.begin(), similarities.end(), [](const std::pair<float, std::string>& a, const std::pair<float, std::string>& b) {
            return a.first > b.first; // For higher intersection values
        });
    } else {
        // Sort in ascending order for SSD, the lower, the more similar
        std::sort(similarities.begin(), similarities.end());
    }

    std::cout << "Top " << N << " Matches: " << std::endl;
    // Start loop from 1 to skip the target image, assuming it's the first match
    int matchesToShow = N + 1; // Increase by one to account for skipping the target image
    for (int i = 1; i < matchesToShow && i < similarities.size(); i++) {
        std::cout << similarities[i].second << " with similarity: " << similarities[i].first << std::endl;
    }

    return 0;
}
