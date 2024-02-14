/**
 * @file faceDetecting.cpp
 * @author Yuan Zhao (zhao.yuan2@northeatern.edu)
 * @brief use the csv file by created by extractFeature2csv.cpp
 *        to find all the images with face features
 * @version 0.1
 * @date 2024-02-03
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>



// Function to read a CSV file and print filenames and their data
void printImagesWithData(const std::string& csvFilePath) {
    std::ifstream file(csvFilePath);
    std::string line;

    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << csvFilePath << std::endl;
        return;
    }

    while (getline(file, line)) {
        std::istringstream ss(line);
        std::string filename;
        std::string data;
        bool hasData = false;

        // Extract the filename
        if (!getline(ss, filename, ',')) {
            continue; // Skip if the line is empty
        }

        // Check if there is additional data
        while (getline(ss, data, ',')) {
            if (!data.empty()) {
                hasData = true;
                break;
            }
        }

        // Print the filename if it has associated data
        if (hasData) {
            std::cout << filename << " has face feature" << std::endl;
        }
    }

    file.close();
}

int main() {
    const char* csvFilePath = "/Users/jeff/Desktop/Project2_YZ/bin/image_features_face.csv"; // Update this with the actual path to your CSV file
    printImagesWithData(csvFilePath);
    return 0;
}