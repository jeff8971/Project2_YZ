#ifndef MATCHINGS_H
#define MATCHINGS_H


#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


// Task 1: baseline matching
// Function to extract 7x7 feature vector from an image
std::vector<float> extract7x7FeatureVector(const cv::Mat &image);
// Function to compute the sum of squared differences between two vectors
float computeSSD(const std::vector<float>& vec1, const std::vector<float>& vec2);



#endif