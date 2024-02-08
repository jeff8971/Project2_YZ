/**
 * @file extractImgFeature2csv.cpp
 * @author Yuan Zhao (zhao.yuan2@northeatern.edu)
 * @brief main entry for the project
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

int main(int argc, char* argv[]){
    // N default is 3
    int N = 3;
    // Check the number of arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <target_image_name> <directory_of_images> <N>" << std::endl;
        return EXIT_FAILURE;
    } else if (argc >= 4) {
        // if N is supplied
        N = std::atoi(argv[3]);
    }
    std::cout << "N is set to " << N << std::endl;

    const std::string PROJECT_DIRECTORY = "/Users/jeff/Desktop/Project2_YZ";
    // const std::string IMAGE_DIRECTORY_NAME = "/Users/jeff/Desktop/Project2_YZ/olympus";
    const std::string IMAGE_CSV_FILE = "/bin/image_features.csv";

    // Set the directory path from the command line
    std::string target_image_name = std::string(argv[1]);
    std::string image_directory_path = std::string(argv[2]);
    std::string target_image_directory_path = image_directory_path + "/" + target_image_name;


    std::string filename = IMAGE_CSV_FILE;

    // Delete the existing file 
    std::remove(filename.c_str());

    // Read the target image
    cv::Mat target_image = cv::imread(target_image_directory_path, cv::IMREAD_COLOR);
    if (target_image.empty()) {
        std::cerr << "Error: cannot read image " << target_image_directory_path << std::endl;
        return EXIT_FAILURE;
    }

    // Extract the feature vector from the target image
    std::vector<float> target_feature = extract7x7FeatureVector(target_image);
    

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


    std::sort (image_distance_pairs.begin(), image_distance_pairs.end(), 
        [](const std::pair<std::string, float> &a, const std::pair<std::string, float> &b) -> bool {
            return a.second < b.second;
        }
    );

    for (int i = 0; i < N; i++) {
        std::cout << "Image: " << image_distance_pairs[i].first << " Distance: " << image_distance_pairs[i].second << std::endl;
    }


    return 0;
  

}