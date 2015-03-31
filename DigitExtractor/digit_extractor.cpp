#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <filesystem>

#include "SudokuSolver\globals.h"
#include "SudokuSolver\extractor.h"

int main( int argc, char** argv )
{
	int sudoku[NUM_CELLS];
	
	Extractor extractor;
    cv::Mat image;
	std::vector<cv::Mat> digits;

	cv::namedWindow( image_window_name, cv::WINDOW_AUTOSIZE );
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	std::tr2::sys::directory_iterator start(assets_dir + training_dir), end;
	for ( ; start != end; ++start) {
		std::cout << assets_dir + training_dir + start->path().string() << '\n';
		image = cv::imread(assets_dir + training_dir + start->path().string(), cv::IMREAD_GRAYSCALE);
		cv::imshow( image_window_name, image );
		digits = extractor.extract(image, sudoku);

		for (size_t i = 0; i < digits.size(); ++i) {
			cv::imwrite(assets_dir + digits_dir + start->path().string().substr(0, start->path().string().size() - 4) + "-" + std::to_string(i) + ".png",
				digits[i], compression_params);
		}
	}

    cv::waitKey(0);
    return 0;
}