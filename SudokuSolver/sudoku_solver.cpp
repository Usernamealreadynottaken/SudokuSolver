#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "extractor.h"

static const std::string image_window_name = "Original";
static const std::string assets_dir = "assets/";
static const std::string puzzles_dir = "puzzles/";
static const std::string digits_dir = "digits/";
static const std::string image_name = "image1.jpg";

int main( int argc, char** argv )
{
	Extractor extractor;
    cv::Mat image;
	std::vector<cv::Mat> digits;
	image = cv::imread(assets_dir + puzzles_dir + image_name, cv::IMREAD_GRAYSCALE);

	cv::namedWindow( image_window_name, cv::WINDOW_AUTOSIZE );
    cv::imshow( image_window_name, image );

	digits = extractor.extract(image);
	// save digits, optional
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	for (size_t i = 0; i < digits.size(); ++i) {
		cv::imwrite(assets_dir + digits_dir + image_name.substr(0, image_name.size() - 4) + "-" + std::to_string(i) + ".png",
			digits[i], compression_params);
	}

    cv::waitKey(0);
    return 0;
}