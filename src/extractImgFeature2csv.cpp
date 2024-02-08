/**
 * @file extractImgFeature2csv.cpp
 * @author Yuan Zhao (zhao.yuan2@northeatern.edu)
 * @brief main entry for the project
 * @version 0.1
 * @date 2024-02-03
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "matchings.h"
#include <vector>

#define IMAGE_DIRECTORY "/Users/jeff/Desktop/Project2_YZ/olympus"
#define IMAGE_CSV_FILE "/Users/jeff/Desktop/Project2_YZ/bin/image_data.csv"


int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return EXIT_FAILURE;
    }

    // Set the directory path from the command line
    std::string directory_path = argv[1];
    std::string filename = IMAGE_CSV_FILE;

    // Delete the existing file 
    std::remove(filename.c_str());

    

    
  

}