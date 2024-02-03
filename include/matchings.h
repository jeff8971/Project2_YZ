#ifndef MATCHINGS_H
#define MATCHINGS_H


#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#define BASELINE_TARGET_IMAGE "/Users/jeff/Desktop/Project2_YZ/olympus/pic.1016.jpg"

std::vector<float> extract7x7FeatureVector(const cv::Mat &image);



#endif