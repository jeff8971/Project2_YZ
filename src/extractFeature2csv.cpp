/**
 * @file extractFeature2csv.cpp
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
#include "csv_util.h"


// Menu for the user
void extractMenu(){
    printf("Usage: ./extractFeature <method> <directory_of_images>\n");
    printf("method:\n");
    printf("  b: use the Baseline method to extract the feature\n");
    printf("  h2: use the RG 2D Histogram method to extract the feature\n");
    printf("  h3: use the RGB 3D Histogram method to extract the feature\n");
    printf("  m: use the Multi-histogram method to extract the feature\n");
    printf("  tc: use the Texture and Color method to extract the feature\n");
    printf("  glcm: use the GLCM filter to extract the feature\n");
    printf("  l: use the Laws' Histogram to extract the feature\n");
    printf("  gabor: use the Gabor Histogram to extract the feature\n");
    printf("  custom: use the custom method to extract the feature\n");
}


int main(int argc, char* argv[]){
    // Check the number of arguments
    if (argc < 3) {
        extractMenu();
        return EXIT_FAILURE;
    }
    // method is the first argument
    std::string method = argv[1];
    if (method != "b" 
    && method != "h2" 
    && method!= "h3" 
    && method != "m" 
    && method != "tc" 
    && method != "glcm"
    && method != "l"
    && method != "gabor"
    && method != "custom") {
        std::cerr << "Error: invalid method" << std::endl;
        extractMenu();
        return EXIT_FAILURE;
    }

    // Set the directory path from the command line, 2nd argument
    std::string directory_of_images = argv[2];

    // Set the csv file name
    std::string csvFile = "image_features_";
    if (method == "b") {
        csvFile += "baseline.csv";
    } else if (method == "h2") {
        csvFile += "2D_histogram.csv";
    } else if (method == "h3") {
        csvFile += "3D_histogram.csv";
    } else if (method == "m") {
        csvFile += "multi_histogram.csv";
    } else if (method == "tc") {
        csvFile += "texturecolor.csv";
    } else if (method == "glcm") {
        csvFile += "glcm.csv";
    } else if (method == "l") {
        csvFile += "laws.csv";
    } else if (method == "gabor"){
        csvFile += "gabor.csv";
    } else if (method == "custom") {
        csvFile += "custom.csv";
    } else {
        std::cerr << "Error: invalid method" << std::endl;
        return EXIT_FAILURE;
    }

    // delete the existing csv file
    if (std::remove(csvFile.c_str()) == 0) {
        std::cout << "Existing CSV file deleted successfully." << std::endl;
    } else {
        // if the file does not exist
        std::cout << "No existing CSV file to delete or deletion failed." << std::endl;
    }
    
    // Read the images from the directory
    std::vector<std::pair<std::string, float>> image_distance_pairs;
    DIR *dir;
    struct dirent* ent;

    if ((dir = opendir(directory_of_images.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string file_name = ent->d_name;
            // Skip current directory and parent directory entries
            if (file_name == "." || file_name == "..") continue;

            std::string full_file_path = directory_of_images + "/" + file_name;
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

            // Extract the feature vector from the image from specified method
            if (method == "b") {
                // extract the feature vector from the all the images
                std::vector<float> feature = extract7x7FeatureVector(img);
                int error = append_image_data_csv(const_cast<char*>(csvFile.c_str()), const_cast<char*>(file_name.c_str()), feature, false);
                if (error) {
                    std::cerr << "Error: cannot append to the csv file" << std::endl;
                    return EXIT_FAILURE;
                }
            } else if (method == "h2") {
                // Extract RG Chroma Histogram features
                std::vector<float> feature = calculateRG_2DChromaHistogram(img, BINS_2D); // 16 bins per channel as an example
                // Write the features to the CSV file
                int error = append_image_data_csv(const_cast<char*>(csvFile.c_str()), const_cast<char*>(file_name.c_str()), feature, false);
                if (error) {
                    std::cerr << "Error: cannot append to the csv file" << std::endl;
                    return EXIT_FAILURE;
                }
            } else if (method == "h3") {
                // Extract RGB 3D Color Histogram features
                std::vector<float> feature = calculateRGB_3DChromaHistogram(img, BINS_3D); // 8 bins per channel as an example
                // Write the features to the CSV file
                int error = append_image_data_csv(const_cast<char*>(csvFile.c_str()), const_cast<char*>(file_name.c_str()), feature, false);
                if (error) {
                    std::cerr << "Error: cannot append to the csv file" << std::endl;
                    return EXIT_FAILURE;
                }
            } else if (method == "m") {
                // Extract Multi-RG Chroma Histogram features
                std::vector<float> feature = calculateMultiPartRGBHistogram(img, BINS_3D); // 8 bins per channel as an example
                // Write the features to the CSV file
                int error = append_image_data_csv(const_cast<char*>(csvFile.c_str()), const_cast<char*>(file_name.c_str()), feature, false);
                if (error) {
                    std::cerr << "Error: cannot append to the csv file" << std::endl;
                    return EXIT_FAILURE;
                }
            } else if (method == "tc") {
                // Extract Texture features
                std::vector<float> feature = calculateColorTextureFeatureVector(img, COLOR_BINS, TEXTURE_BINS);
                // Write the features to the CSV file
                int error = append_image_data_csv(const_cast<char*>(csvFile.c_str()), const_cast<char*>(file_name.c_str()), feature, false);
                if (error) {
                    std::cerr << "Error: cannot append to the csv file" << std::endl;
                    return EXIT_FAILURE;
                }
            } else if (method == "glcm") {
                // Extract Color features
                std::vector<float> feature = calculateGLCMFeatures(img, GLCM_DISTANCE, GLCM_ANGLE, GLCM_LEVELS);
                // Write the features to the CSV file
                int error = append_image_data_csv(const_cast<char*>(csvFile.c_str()), const_cast<char*>(file_name.c_str()), feature, false);
                if (error) {
                    std::cerr << "Error: cannot append to the csv file" << std::endl;
                    return EXIT_FAILURE;
                }
            } else if (method == "l") {
                // Extract Color features
                std::vector<float> feature = calculateLawsTextureFeatures(img);
                // Write the features to the CSV file
                int error = append_image_data_csv(const_cast<char*>(csvFile.c_str()), const_cast<char*>(file_name.c_str()), feature, false);
                if (error) {
                    std::cerr << "Error: cannot append to the csv file" << std::endl;
                    return EXIT_FAILURE;
                }
            } else if (method == "gabor") {
                std::vector<float> feature = computeGaborFeatures(img);
                // Write the features to the CSV file
                int error = append_image_data_csv(const_cast<char*>(csvFile.c_str()), const_cast<char*>(file_name.c_str()), feature, false);
                if (error) {
                    std::cerr << "Error: cannot append to the csv file" << std::endl;
                    return EXIT_FAILURE;
                }
            } else if (method == "custom") {
                std::vector<float> feature = calculateCustomFeature(img);
                // Write the features to the CSV file
                int error = append_image_data_csv(const_cast<char*>(csvFile.c_str()), const_cast<char*>(file_name.c_str()), feature, false);
                if (error) {
                    std::cerr << "Error: cannot append to the csv file" << std::endl;
                    return EXIT_FAILURE;
                }
            } else {
                std::cerr << "Error: invalid method" << std::endl;
                return EXIT_FAILURE;
            } 
        }
        closedir(dir);
    } else {
        std::cerr << "Error: cannot open directory " << directory_of_images << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Feature extraction is written to " << csvFile << std::endl;
    return 0;

}