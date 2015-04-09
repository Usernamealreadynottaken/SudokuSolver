#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "globals.h"
#include "extractor.h"

static const std::string image_name = "simple1.jpg";

int main( int argc, char** argv )
{
	int sudoku[NUM_CELLS];
	
	Extractor extractor;
    cv::Mat image;
	std::vector<cv::Mat> digits;

	image = cv::imread(assets_dir + puzzles_dir + image_name, cv::IMREAD_GRAYSCALE);
	
	cv::namedWindow( image_window_name, cv::WINDOW_AUTOSIZE );
    cv::imshow( image_window_name, image );

	digits = extractor.extract(image, sudoku);

    cv::waitKey(0);
    return 0;
}