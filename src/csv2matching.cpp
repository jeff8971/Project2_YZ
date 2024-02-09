/**
 * @file csv2matching.cpp
 * @author Yuan Zhao (zhao.yuan2@northeatern.edu)
 * @brief use the csv file's features data to find the matching images
 * @version 0.1
 * @date 2024-02-03
*/


#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "matchings.h"


void Menu(){
    printf("Usage: ./extractFeature <method> <target_image_name> <directory_of_images> <Top N>\n");
    printf("method:\n");
    printf("  b: use the Baseline method to extract the feature\n");
    printf("  h: use the Histogram method to extract the feature\n");
    printf("  m: use the Multi-histogram method to extract the feature\n");
    printf("  t: use the Texture method to extract the feature\n");
    printf("  c: use the Color method to extract the feature\n");
}

int main(int argc, char* argv[]){
    // Check the number of arguments
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <method> <target_image_name> <directory_of_images> <Top N>" << std::endl;
        Menu();
        return EXIT_FAILURE;
    }
    // method is the first argument
    std::string method = std::string(argv[1]);
    if (method != "b" && method != "h" && method != "m" && method != "t" && method != "c") {
        std::cerr << "Error: invalid method" << std::endl;
        Menu();
        return EXIT_FAILURE;
    }

    // Set the directory path from the command line, 2nd and 3rd argument
    std::string target_image_name = std::string(argv[2]);
    std::string image_directory_path = std::string(argv[3]);

    // Set the number of top N images to output
    int N = 4;
    if (argc == 5) {
        N = std::stoi(argv[4]);
    }
    std::cout << "N is set to " << N << std::endl;
    // Check if N is valid
    if (N < 1) {
        std::cerr << "Error: invalid N" << std::endl;
        return EXIT_FAILURE;
    }


    // Set the csv file name
    std::string csvFile;
    if (method == "b") {
        csvFile = "/bin/image_features_baseline.csv";
    } else if (method == "h") {
        csvFile = "/bin/image_features_histogram.csv";
    } else if (method == "m") {
        csvFile = "/bin/image_features_multi_histogram.csv";
    } else if (method == "t") {
        csvFile = "/bin/image_features_texture.csv";
    } else if (method == "c") {
        csvFile = "/bin/image_features_color.csv";
    } else {
        std::cerr << "Error: invalid method" << std::endl;
        return EXIT_FAILURE;
    }


    // read the target image
    std::string target_image_directory_path = image_directory_path + "/" + target_image_name;




    // Delete the existing file 
    std::remove(csvFile.c_str());

    // Read the target image
    cv::Mat target_image = cv::imread(target_image_directory_path, cv::IMREAD_COLOR);
    if (target_image.empty()) {
        std::cerr << "Error: cannot read image " << target_image_directory_path << std::endl;
        return EXIT_FAILURE;
    }

    // Extract the feature vector from the target image
    std::vector<float> target_feature = extract7x7FeatureVector(target_image);
    
    // Read the images from the directory
    std::vector<std::pair<std::string, float>> image_distance_pairs;
    DIR *dir;
    struct dirent* ent;

    if ((dir = opendir(image_directory_path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string file_name = ent->d_name;
            // Skip current directory and parent directory entries
            if (file_name == "." || file_name == "..") continue;

            std::string full_file_path = image_directory_path + "/" + file_name;
            std::string extension = file_name.substr(file_name.find_last_of(".") + 1);

            // Skip files that are not images
            if (extension != "jpg" && extension != "jpeg" && extension != "png" && extension != "tif") {
                std::cerr << "Skipping non-image file: " << full_file_path << std::endl;
                continue;
            }

            cv::Mat img = cv::imread(full_file_path, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Could not read the image: " << full_file_path << std::endl;
                continue;
            }

            std::vector<float> feature = extract7x7FeatureVector(img);
            float distance = computeSSD(target_feature, feature);
            image_distance_pairs.push_back(std::make_pair(full_file_path, distance));
        }
        closedir(dir);
    } else {
        std::cerr << "Error: cannot open directory " << image_directory_path << std::endl;
        return EXIT_FAILURE;
    }

    // Sort the image-distance pairs by distance
    std::sort (image_distance_pairs.begin(), image_distance_pairs.end(), 
        [](const std::pair<std::string, float> &a, const std::pair<std::string, float> &b) -> bool {
            return a.second < b.second;
        }
    );

    // Output the top N images and their distances
    for (int i = 0; i < (N + 1); i++) {
        std::cout << "Image: " << image_distance_pairs[i].first << " Distance: " << image_distance_pairs[i].second << std::endl;
    }

    return 0;

}