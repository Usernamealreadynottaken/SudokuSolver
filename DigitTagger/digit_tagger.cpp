#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <filesystem>
#include <iostream>
#include <fstream>

#include "SudokuSolver/globals.h"

int main( int argc, char** argv )
{
    cv::Mat image;
	int digit_value = 0;

	cv::namedWindow( image_window_name, cv::WINDOW_AUTOSIZE );
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	std::tr2::sys::directory_iterator start(assets_dir + digits_dir), end;
	for ( ; start != end; ++start) {
		std::cout << start->path().string() << ' ';
		image = cv::imread(assets_dir + digits_dir + start->path().string(), cv::IMREAD_GRAYSCALE);
		cv::imshow( image_window_name, image );

		digit_value = cv::waitKey(0);
		if (digit_value == 27) {
			std::cout << "\nbreak!";
			break;
		}
		while (digit_value < 48 || digit_value > 57) {
			digit_value = cv::waitKey(0);
			if (digit_value == 27) {
				std::cout << "break!";
				break;
			}
		}
		digit_value -= 48;
		if (digit_value > 0) {
			cv::imwrite(assets_dir + inputs_dir + std::to_string(digit_value) + start->path().string().substr(0, start->path().string().size() - 4) + ".png",
				image, compression_params);
			std::cout << digit_value << '\n';
		} else {
			std::cout << "not a number\n";
		}
	}

	cv::waitKey(0);
    return 0;
}