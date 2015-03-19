#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

#define NUM_CELLS 81

class Extractor
{
private:
	cv::Mat result_image;
	int low_threshold;
	int high_threshold;
	int kernel_size;
	cv::vector<cv::Vec2f> lines;
	float theta_vertical;
	float theta_horizontal;
	float theta_error;
	cv::vector<cv::vector<cv::Point>> contours;
    cv::vector<cv::Vec4i> hierarchy;
	int max_indexes[NUM_CELLS];
	int max_areas[NUM_CELLS];
	float min_scale;
	float max_scale;
protected:
public:
	Extractor();
	~Extractor() { };
	std::vector<cv::Mat> extract(cv::Mat image);
	void filterLines(cv::vector<cv::Vec2f> & unfiltered);
	void filterContours(cv::vector<cv::vector<cv::Point>> & unfiltered);
	void drawLines(cv::vector<cv::Vec2f> & lines, cv::Mat & dst) const;
	void drawContours(cv::vector<cv::vector<cv::Point>> & contours, cv::Mat & dst) const;
};