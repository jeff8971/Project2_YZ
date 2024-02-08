/**
 * @file imgFeature2csv.cpp
 * @author Yuan Zhao (zhao.yuan2@northeatern.edu)
 * @brief various matching methods for the project
 * @version 0.1
 * @date 2024-02-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "csv_util.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <dirent.h> // For directory operations in POSIX
#include <cstring>
#include <iostream>


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return EXIT_FAILURE;
    }

    // Set the directory path from the first command-line argument
    std::string directory_path = argv[1];
    std::string filename = "img_database.csv"; // Assuming the CSV is to be saved in the current directory

    // Delete the existing file to avoid appending to old data
    std::remove(filename.c_str());

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directory_path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string file_name = ent->d_name;
            // Skip current directory and parent directory entries
            if (file_name == "." || file_name == "..") continue;

            std::string full_file_path = directory_path + "/" + file_name;
            std::string extension = file_name.substr(file_name.find_last_of(".") + 1);

            // Skip files that are not images
            if (extension != "jpg" && extension != "jpeg" && extension != "png") {
                std::cerr << "Skipping non-image file: " << full_file_path << std::endl;
                continue;
            }

            cv::Mat img = cv::imread(full_file_path, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Could not read the image: " << full_file_path << std::endl;
                continue;
            }

            int center_x = img.cols / 2;
            int center_y = img.rows / 2;
            cv::Rect roi(center_x - 4, center_y - 4, 9, 9);
            cv::Mat feature_mat = img(roi).clone();

            std::vector<float> features;
            for (int i = 0; i < feature_mat.rows; i++) {
                for (int j = 0; j < feature_mat.cols; j++) {
                    cv::Vec3b pixel = feature_mat.at<cv::Vec3b>(i, j);
                    for (int k = 0; k < 3; k++) {
                        features.push_back(static_cast<float>(pixel[k]));
                    }
                }
            }

            // Write the extracted features and image path to the CSV file
            append_image_data_csv(const_cast<char*>(filename.c_str()), const_cast<char*>(full_file_path.c_str()), features);
            features.clear();
        }
        closedir(dir);
    } else {
        std::cerr << "Cannot open directory: " << directory_path << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
