/**
 * @file matchings.cpp
 * @author Yuan Zhao (zhao.yuan2@northeatern.edu)
 * @brief various matching methods for the project
 * @version 0.1
 * @date 2024-02-03
*/


#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include "matchings.h"
#include "csv_util.h"


// Extract 7x7 feature vector from the center of the image, make it into a 1D vector
std::vector<float> extract7x7FeatureVector(const cv::Mat &image) {
  // if image is empty, throw runtime error
  if (image.empty()) {
    throw std::runtime_error("Image is empty");
  }

  // if image is smaller thann 7x7, throw runtime error
  if (image.rows < 7 || image.cols < 7) {
    throw std::runtime_error("Image is too small");
  }

  // Find the center of the image
  int centerX = image.cols / 2;
  int centerY = image.rows / 2;

  // calculate the start and end point for 7x7 square
  int startX = centerX - 3;
  int endX = centerX + 4;

  int startY = centerY - 3;
  int endY = centerY + 4;

  // Extract the 7x7 region
  cv::Mat region = image(cv::Range(startY, endY), cv::Range(startX, endX));
  
  // Convert the 7x7 region to a 1D vector
    std::vector<float> featureVector;
    for (int i = 0; i < region.rows; ++i) {
        for (int j = 0; j < region.cols; ++j) {
            for (int c = 0; c < region.channels(); ++c) {
                // Add the pixel value to the feature vector
                featureVector.push_back(static_cast<float>(region.at<cv::Vec3b>(i, j)[c]));
            }
        }
    }

    return featureVector;
}


// Compute the distance metric (sum of squared differences) between two feature vectors
float computeSSD(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    // Check if vectors are of the same size
    if (vec1.size() != vec2.size()) {
        throw std::runtime_error("Feature vectors must be of the same size");
    }
    // ssd = sum of squared differences
    float ssd = 0.0;

    // Compute the sum of squared differences
    for (size_t i = 0; i < vec1.size(); ++i) {
        float diff = vec1[i] - vec2[i];
        ssd += diff * diff;
    }

    return ssd;
}

