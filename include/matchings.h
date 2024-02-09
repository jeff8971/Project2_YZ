#ifndef MATCHINGS_H
#define MATCHINGS_H


#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#define BINS_2D 16
#define BINS_3D 8


// Task 1: baseline matching
// Function to extract 7x7 feature vector from an image
std::vector<float> extract7x7FeatureVector(const cv::Mat &image);
// Function to compute the sum of squared differences between two vectors
float computeSSD(const std::vector<float>& vec1, const std::vector<float>& vec2);

// Task 2: 2D & 3D histogram matching
// Function to extract the 2D histogram feature vector from an image
std::vector<float> calculateRG_2DChromaHistogram(const cv::Mat& image, int binsPerChannel);
// Function to extract the 3D histogram feature vector from an image
std::vector<float> calculateRGB_3DChromaHistogram(const cv::Mat& image, int binsPerChannel);
// Function to compute the histogram intersection distance between two vectors
float computeHistogramIntersection(const std::vector<float>& vec1, const std::vector<float>& vec2);


#endif