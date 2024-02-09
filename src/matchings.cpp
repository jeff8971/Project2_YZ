/**
 * @file matchings.cpp
 * @author Yuan Zhao (zhao.yuan2@northeatern.edu)
 * @brief various matching methods for the project
 * @version 0.1
 * @date 2024-02-03
*/


#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "matchings.h"
#include "csv_util.h"

// Task 1: baseline matching
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

// Task 2: 2D & 3D histogram matching
// Extract the (RG) 2D histogram feature vector from an image
std::vector<float> calculateRG_2DChromaHistogram(const cv::Mat& image, int binsPerChannel) {
    std::vector<float> featureVector(binsPerChannel * binsPerChannel, 0.0f);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            int sum = pixel[0] + pixel[1] + pixel[2];

            // Skip this pixel if the sum is 0 to avoid division by zero
            if (sum == 0) continue;

            float r = pixel[2] / static_cast<float>(sum);
            float g = pixel[1] / static_cast<float>(sum);

            int binR = std::min(static_cast<int>(r * binsPerChannel), binsPerChannel - 1);
            int binG = std::min(static_cast<int>(g * binsPerChannel), binsPerChannel - 1);

            featureVector[binR * binsPerChannel + binG] += 1;
        }
    }

    // Normalize the histogram
    float total = std::accumulate(featureVector.begin(), featureVector.end(), 0.0f);
    for (auto& val : featureVector) {
        val /= total;
    }

    return featureVector;
}

// Extract the RGB 3D histogram feature vector from an image
std::vector<float> calculateRGB_3DChromaHistogram(const cv::Mat& image, int binsPerChannel) {
    int bins3D = binsPerChannel * binsPerChannel * binsPerChannel;
    std::vector<float> featureVector(bins3D, 0.0f);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);

            // Calculate the bin index for each color channel
            int binR = std::min(static_cast<int>(pixel[2] * binsPerChannel / 256.0), binsPerChannel - 1);
            int binG = std::min(static_cast<int>(pixel[1] * binsPerChannel / 256.0), binsPerChannel - 1);
            int binB = std::min(static_cast<int>(pixel[0] * binsPerChannel / 256.0), binsPerChannel - 1);

            // Increment the appropriate bin
            featureVector[binR * binsPerChannel * binsPerChannel + binG * binsPerChannel + binB] += 1;
        }
    }

    // Normalize the histogram so that the sum of bin values equals 1
    float total = std::accumulate(featureVector.begin(), featureVector.end(), 0.0f);
    for (auto& val : featureVector) {
        val /= total;
    }

    return featureVector;
}

// Function to compute the histogram intersection distance between two vectors
float computeHistogramIntersection(const std::vector<float>& vec1, const std::vector<float>& vec2){
    // Check if vectors are of the same size
    if (vec1.size() != vec2.size()) {
        throw std::runtime_error("Feature vectors must be of the same size");
    }

    // Compute the histogram intersection distance
    float intersection = 0.0;
    for (size_t i = 0; i < vec1.size(); i++) {
        intersection += std::min(vec1[i], vec2[i]);
    }
    return intersection;
}