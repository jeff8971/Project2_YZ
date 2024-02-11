#ifndef MATCHINGS_H
#define MATCHINGS_H


#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#define BINS_2D 16
#define BINS_3D 8
#define COLOR_BINS 8
#define TEXTURE_BINS 8

#define SPLIT_POINT (BINS_3D * BINS_3D * BINS_3D)


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

// Task 3: Multi-histogram matching
// Extract the multi-channel histogram feature vector from an image
// Divided the image into 2 parts, top and bottom
std::vector<float> calculateMultiPartRGBHistogram(const cv::Mat& image, int binsPerChannel);
// Function to compute the histogram intersection distance between two vectors
float combinedHistogramIntersection(const std::vector<float>& vec1, const std::vector<float>& vec2, size_t splitPoint);

// Task 4: Texture and Color matching

// SobelX and SobelY filter from Project 1
// Sobel_X 3 x 3 function
int sobelX3x3(const cv::Mat &src, cv::Mat &dst );
// Sobel_Y 3 x 3 function
int sobelY3x3(const cv::Mat &src, cv::Mat &dst );
// generates a gradient magnitude image from the X and Y Sobel images
int magnitude(const cv::Mat &sx, const cv::Mat &sy, cv::Mat &dst);
// Extract the Texture Histogram from Sobel Magnitude Image
std::vector<float> calculateTextureHistogram(const cv::Mat& magnitudeImage, int bins);
// Combine the color and texture histograms into a single feature vector, giving equal weight to both
std::vector<float> calculateCombinedFeatureVector(const cv::Mat& image, int colorBinsPerChannel, int textureBins);
// Function to compute the histogram intersection distance
// A simple approach: Euclidean distance separately for the color and texture parts
// of the combined vector and then average these distances:
float calculateColorTextureIntersection(const std::vector<float>& vec1, const std::vector<float>& vec2, size_t splitPoint);



#endif