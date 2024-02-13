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

// Task 3: Multi-histogram matching
// Extract the multi-channel histogram feature vector from an image
// Divided the image into 2 parts, top and bottom
std::vector<float> calculateMultiPartRGBHistogram(const cv::Mat& image, int binsPerChannel) {
    // Divide the image into top and bottom halves
    cv::Rect topHalf(0, 0, image.cols, image.rows / 2);
    cv::Rect bottomHalf(0, image.rows / 2, image.cols, image.rows / 2);

    cv::Mat topPart = image(topHalf);
    cv::Mat bottomPart = image(bottomHalf);

    // Calculate histograms for each part
    std::vector<float> topFeatureVector = calculateRGB_3DChromaHistogram(topPart, binsPerChannel);
    std::vector<float> bottomFeatureVector = calculateRGB_3DChromaHistogram(bottomPart, binsPerChannel);

    // Combine the two histograms into a single feature vector
    topFeatureVector.insert(topFeatureVector.end(), bottomFeatureVector.begin(), bottomFeatureVector.end());

    return topFeatureVector;
}

// Function to compute the histogram intersection distance between two vectors
float combinedHistogramIntersection(const std::vector<float>& vec1, const std::vector<float>& vec2, size_t splitPoint) {
    // Assumes vec1 and vec2 are combined histograms from two parts
    // splitPoint is the index where the second histogram starts in the combined vector

    // validate the vec1 and vec2 size
    if (vec1.size() != vec2.size()) {
        throw std::runtime_error("Feature vectors must be of the same size");
    }
    // validate the split point
    if (splitPoint >= vec1.size() || splitPoint == 0) {
        throw std::runtime_error("Split point must be within the range");
    }

    // Calculate intersection for each part
    float intersection1 = computeHistogramIntersection(std::vector<float>(vec1.begin(), vec1.begin() + splitPoint), 
                                                         std::vector<float>(vec2.begin(), vec2.begin() + splitPoint));
    float intersection2 = computeHistogramIntersection(std::vector<float>(vec1.begin() + splitPoint, vec1.end()), 
                                                            std::vector<float>(vec2.begin() + splitPoint, vec2.end()));

    // Combine the intersections (example: simple average)
    return (intersection1 + intersection2) / 2.0f;
}


// Task 4: Texture and Color matching
// SobelX and SobelY filter from Project 1
// Sobel_X 3 x 3 function
int sobelX3x3(const cv::Mat &src, cv::Mat &dst ){
    if (src.empty()) {
        return -1;
    }

    dst.create(src.size(), CV_16SC3);

    // Horizontal kernel [-1, 0, 1]
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            cv::Vec3s sum(0, 0, 0);
            for (int dx = -1; dx <= 1; dx++) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(y, x + dx);
                for (int c = 0; c < 3; c++) {
                    sum[c] += pixel[c] * dx;
                }
            }
            dst.at<cv::Vec3s>(y, x) = sum;
        }
    }
    return 0;
}

// Sobel_Y 3 x 3 function
int sobelY3x3(const cv::Mat &src, cv::Mat &dst ){
    if (src.empty()) {
        return -1;
    }

    dst.create(src.size(), CV_16SC3);

    // Vertical kernel [-1, 0, 1] transposed
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            cv::Vec3s sum(0, 0, 0);
            for (int dy = -1; dy <= 1; dy++) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(y + dy, x);
                for (int c = 0; c < 3; c++) {
                    sum[c] += pixel[c] * dy;
                }
            }
            dst.at<cv::Vec3s>(y, x) = sum;
        }
    }
    return 0;
}

// generates a gradient magnitude image from the X and Y Sobel images
int magnitude(const cv::Mat &sx, const cv::Mat &sy, cv::Mat &dst) {
    if (sx.empty() || sy.empty() || sx.size() != sy.size() || sx.type() != sy.type()) {
        return -1;
    }

    dst.create(sx.size(), CV_8UC3);

    for (int y = 0; y < sx.rows; y++) {
        for (int x = 0; x < sx.cols; x++) {
            cv::Vec3b magnitudeColor;
            for (int c = 0; c < 3; c++) {
                float gradX = sx.at<cv::Vec3s>(y, x)[c];
                float gradY = sy.at<cv::Vec3s>(y, x)[c];
                magnitudeColor[c] = cv::saturate_cast<uchar>(std::sqrt(gradX * gradX + gradY * gradY));
            }
            dst.at<cv::Vec3b>(y, x) = magnitudeColor;
        }
    }
    return 0;
}


// Extract the Texture Histogram from Sobel Magnitude Image
std::vector<float> calculateTextureHistogram(const cv::Mat& magnitudeImage, int bins) {
    cv::Mat hist;
    int histSize[] = {bins};
    float range[] = {0, 256};
    const float* ranges[] = {range};
    cv::calcHist(&magnitudeImage, 1, nullptr, cv::Mat(), hist, 1, histSize, ranges, true, false);
    cv::normalize(hist, hist, 1, 0, cv::NORM_L1); // Normalize to make the sum of bins equal to 1

    return std::vector<float>(hist.begin<float>(), hist.end<float>());
}


// Combine the color and texture histograms into a single feature vector, giving equal weight to both
std::vector<float> calculateColorTextureFeatureVector(const cv::Mat& image, int colorBinsPerChannel, int textureBins) {
    // Calculate color histogram
    std::vector<float> colorHist = calculateRGB_3DChromaHistogram(image, colorBinsPerChannel);
    
    // Convert the Sobel magnitude image to grayscale if it's not already
    cv::Mat sobelX, sobelY, grayImage, magnitudeImage;
    if (magnitudeImage.channels() > 1) {
        cv::cvtColor(magnitudeImage, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = magnitudeImage; // Use as-is if already single-channel
    }

    // Calculate Sobel magnitude image
    sobelX3x3(image, sobelX); // Assume these functions handle multi-channel images correctly
    sobelY3x3(image, sobelY);
    magnitude(sobelX, sobelY, magnitudeImage); // Results in a multi-channel magnitude image

    // Calculate texture histogram from the grayscale magnitude image
    std::vector<float> textureHist = calculateTextureHistogram(grayImage, textureBins);

    // Combine histograms
    std::vector<float> colorTextureFeatureVector = colorHist;
    colorTextureFeatureVector.insert(colorTextureFeatureVector.end(), textureHist.begin(), textureHist.end());

    return colorTextureFeatureVector;
}

// Task 5: Deep Network Embeddings
// Function to calculate the cosine similarity between two vectors.
float calculateCosineSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float dotProduct = 0.0, normVec1 = 0.0, normVec2 = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        dotProduct += vec1[i] * vec2[i];
        normVec1 += vec1[i] * vec1[i];
        normVec2 += vec2[i] * vec2[i];
    }
    return dotProduct / (std::sqrt(normVec1) * std::sqrt(normVec2));
}


// Task 7: Custom Design
// Calculate the custom feature vector from an image
// Use a combination of Gabor features, edge histogram, Laws' texture features, and color texture features
std::vector<float> calculateCustomFeature(const cv::Mat& img) {
    // Convert the image to grayscale for certain feature calculations
    cv::Mat grayImg;
    if (img.channels() > 1) {
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
    } else {
        grayImg = img.clone();
    }

    // Calculate features using the specified methods
    std::vector<float> gaborFeatures = computeGaborFeatures(grayImg);
    std::vector<float> lawsFeatures = calculateLawsTextureFeatures(grayImg);
    std::vector<float> centralPatchFeatures = extract7x7FeatureVector(img);
    std::vector<float> glcmFeatures = calculateGLCMFeatures(grayImg, GLCM_DISTANCE, GLCM_ANGLE, GLCM_LEVELS); 

    // Combine all features into a single vector
    std::vector<float> combinedFeatures;
    combinedFeatures.insert(combinedFeatures.end(), gaborFeatures.begin(), gaborFeatures.end());
    combinedFeatures.insert(combinedFeatures.end(), lawsFeatures.begin(), lawsFeatures.end());
    combinedFeatures.insert(combinedFeatures.end(), centralPatchFeatures.begin(), centralPatchFeatures.end());
    combinedFeatures.insert(combinedFeatures.end(), glcmFeatures.begin(), glcmFeatures.end());

    return combinedFeatures;
}


/************************************************************************************************
 EXTENSIONS
************************************************************************************************/
// Extension: GLCM texture features
std::vector<float> calculateGLCMFeatures(const cv::Mat& src, int distance, int angle, int levels) {
    cv::Mat gray;
    if (src.channels() > 1) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // Downscale the image to reduce the number of gray levels for simplification
    gray.convertTo(gray, CV_8U, levels / 255.0);

    cv::Mat glcm(levels, levels, CV_32F, cv::Scalar(0));
    int dx = 0;
    int dy = 0;

    // Define dx and dy based on angle
    if (angle == 0) { dx = distance; dy = 0; } // Horizontal
    else if (angle == 45) { dx = distance; dy = -distance; } // Diagonal 45 degree
    else if (angle == 90) { dx = 0; dy = -distance; } // Vertical
    else if (angle == 135) { dx = -distance; dy = -distance; } // Diagonal 135 degree

    // Fill the GLCM
    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            int pixelValue = gray.at<uchar>(y, x);
            int neighborX = x + dx;
            int neighborY = y + dy;

            if (neighborX >= 0 && neighborX < gray.cols && neighborY >= 0 && neighborY < gray.rows) {
                int neighborValue = gray.at<uchar>(neighborY, neighborX);
                glcm.at<float>(pixelValue, neighborValue) += 1.0;
            }
        }
    }

    // Normalize the GLCM
    cv::normalize(glcm, glcm, 1.0, 0.0, cv::NORM_L1);

    // Extract features
    float entropy = 0.0, contrast = 0.0, energy = 0.0, homogeneity = 0.0, maxProbability = 0.0;
    for (int i = 0; i < levels; ++i) {
        for (int j = 0; j < levels; ++j) {
            float value = glcm.at<float>(i, j);
            if (value > 0) {
                entropy -= value * log2(value);
            }
            contrast += value * std::pow(i - j, 2);
            energy += value * value;
            homogeneity += value / (1 + std::abs(i - j));
            maxProbability = std::max(maxProbability, value);
        }
    }

    std::vector<float> features = {energy, entropy, contrast, homogeneity, maxProbability};
    return features;
}


// EXTENSION: Laws' Histogram method
// Generate Laws' Histogram from two vectors
cv::Mat generateLawsFilter(const std::vector<int>& v1, const std::vector<int>& v2) {
    cv::Mat filter(v1.size(), v2.size(), CV_32F);
    for (size_t i = 0; i < v1.size(); ++i) {
        for (size_t j = 0; j < v2.size(); ++j) {
            filter.at<float>(i, j) = v1[i] * v2[j];
        }
    }
    return filter;
}

// Apply Laws' filter and compute texture energy
cv::Mat applyLawsFilter(const cv::Mat& src, const cv::Mat& filter) {
    cv::Mat filtered, energyMap;
    cv::filter2D(src, filtered, CV_32F, filter);
    cv::pow(filtered, 2, energyMap); // Square to get energy
    return energyMap;
}

// Calculate texture energy feature vector for an image using Laws' filters
std::vector<float> calculateLawsTextureFeatures(const cv::Mat& src) {
    // Defining Laws' vectors
    const std::vector<int> L5 = {1, 4, 6, 4, 1};  // Level
    const std::vector<int> E5 = {-1, -2, 0, 2, 1}; // Edge
    const std::vector<int> S5 = {-1, 0, 2, 0, -1}; // Spot
    const std::vector<int> W5 = {-1, 2, 0, -2, 1}; // Wave
    const std::vector<int> R5 = {1, -4, 6, -4, 1}; // Ripple
    
    const std::vector<std::vector<int>> vectors = {L5, E5, S5, W5, R5};

    cv::Mat gray;
    // Convert to grayscale if the source image is not already grayscale
    if (src.channels() > 1) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src;
    }

    std::vector<float> features;
    for (size_t i = 0; i < vectors.size(); ++i) {
        for (size_t j = 0; j < vectors.size(); ++j) {
            cv::Mat filter = generateLawsFilter(vectors[i], vectors[j]);
            cv::Mat energyMap = applyLawsFilter(gray, filter);
            float energy = cv::sum(energyMap)[0];
            features.push_back(energy);
        }
    }
    return features;
}

// EXTENSION: Gabor filter method
std::vector<float> computeGaborFeatures(const cv::Mat& img) {
    // Convert to grayscale if the image is not already
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }

    std::vector<float> features;
    int kernel_size = 31;
    double sigma = 2.5;
    double gamma = 0.5;
    double psi = CV_PI * 0.5; // Convert degrees to radians if needed
    int num_thetas = 4; // Number of orientations
    std::vector<double> lambdas = {10.0, 20.0, 30.0}; // Example wavelengths (λ) for multi-scale analysis

    for (double lambda : lambdas) {
        for (int i = 0; i < num_thetas; ++i) {
            double theta = i * CV_PI / num_thetas; // Vary orientation
            cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta, lambda, gamma, psi, CV_32F);
            cv::Mat dest;
            cv::filter2D(gray, dest, CV_32F, kernel);

            // Compute simple statistical features from the filter response
            cv::Scalar mean, stddev;
            cv::meanStdDev(dest, mean, stddev);
            features.push_back(static_cast<float>(mean[0]));
            features.push_back(static_cast<float>(stddev[0]));
        }
    }

    return features;
}


