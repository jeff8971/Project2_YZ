/**
 * @file dnn_embedding.cpp
 * @author Yuan Zhao (zhao.yuan2@northeatern.edu)
 * @brief DNN-embedding results in csv file to find the Top N matching images
 * @version 0.1
 * @date 2024-02-03
*/


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "csv_util.h" 


// Function to calculate the cosine similarity between two vectors.
float calculateCosineSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float dotProduct = 0.0, normVec1 = 0.0, normVec2 = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        dotProduct += vec1[i] * vec2[i];
        normVec1 += vec1[i] * vec1[i];
        normVec2 += vec2[i] * vec2[i];
    }
    return dotProduct / (std::sqrt(normVec1) * std::sqrt(normVec2));
}

// Main entry
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << "<target_image_name> <Top N>" << std::endl;
        return -1;
    }

    // Path to the CSV file containing the feature vectors.
    std::string csvFilePath = "/Users/jeff/Desktop/Project2_YZ/olympus/ResNet18_olym.csv";

    // Name of the target image.
    std::string targetImageName = argv[1];
    int N = 3;
    if (argc == 3) {
        N = std::stoi(argv[2]);
    }

    // Load the CSV file into memory.
    std::vector<char*> filenames;
    std::vector<std::vector<float>> data;
    if (read_image_data_csv(const_cast<char*>(csvFilePath.c_str()), filenames, data, false) != 0) {
        std::cerr << "Error reading CSV file" << std::endl;
        return -1;
    }

    // Find the feature vector for the target image.
    std::vector<float> targetFeatureVector;
    bool found = false;
    for (size_t i = 0; i < filenames.size(); ++i) {
        if (std::string(filenames[i]) == targetImageName) {
            targetFeatureVector = data[i];
            found = true;
            break;
        }
    }

    if (!found) {
        std::cerr << "Target image not found in CSV" << std::endl;
        return -1;
    }

    // Calculate similarity and store results.
    std::vector<std::pair<float, std::string>> similarityScores;
    for (size_t i = 0; i < data.size(); ++i) {
        float similarity = calculateCosineSimilarity(targetFeatureVector, data[i]);
        similarityScores.emplace_back(similarity, std::string(filenames[i]));
    }

    // Sort based on similarity (higher first).
    std::sort(similarityScores.begin(), similarityScores.end(), [](const std::pair<float, std::string>& a, const std::pair<float, std::string>& b) {
        return a.first > b.first;
    });

    // Print top N similar images.
    std::cout << "Top " << N << " similar images:" << std::endl;
    for (int i = 1; i < (N + 1) && i < similarityScores.size(); i++) {
        std::cout << i << ": " << similarityScores[i].second << " (Similarity: " << similarityScores[i].first << ")" << std::endl;
    }

    // Cleanup.
    for (auto& fname : filenames) {
        delete[] fname;
    }

    return 0;
}
