#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "extractor.h";

static const std::string image_window_name = "Original";
static const std::string assets_dir = "assets/";
static const std::string image_name = "sudoku.jpg";

int main( int argc, char** argv )
{
	Extractor extractor;
    cv::Mat image;
	image = cv::imread(assets_dir + image_name, cv::IMREAD_GRAYSCALE);

	cv::namedWindow( image_window_name, cv::WINDOW_AUTOSIZE );
    cv::imshow( image_window_name, image );

	extractor.extract(image);

    cv::waitKey(0);
    return 0;
}