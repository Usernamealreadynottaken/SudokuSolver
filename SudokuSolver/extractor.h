#pragma once

#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class Extractor
{
private:
	cv::Mat detected_edges;
	int low_threshold;
	int high_threshold;
	int kernel_size;
protected:
public:
	Extractor();
	~Extractor() { };
	void extract(cv::Mat image);
};