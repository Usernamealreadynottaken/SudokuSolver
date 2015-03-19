#include "extractor.h"

Extractor::Extractor() : low_threshold(10), high_threshold(50), kernel_size(3) { }

void Extractor::extract(cv::Mat image)
{
	// double blur to remove noise
	blur( image, detected_edges, cv::Size(3,3) );
	blur( detected_edges, detected_edges, cv::Size(3,3) );
	// canny to detect edges
	cv::Canny(detected_edges, detected_edges, low_threshold, high_threshold, kernel_size);

	cv::Mat dst;
	dst = cv::Scalar::all(0);
	image.copyTo( dst, detected_edges);

	cv::namedWindow( "extracted", cv::WINDOW_AUTOSIZE );
    cv::imshow( "extracted", dst );
}