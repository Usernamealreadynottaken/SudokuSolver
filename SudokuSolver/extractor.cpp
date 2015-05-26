#include "extractor.h"

Extractor::Extractor() : low_threshold(10), high_threshold(50), kernel_size(3),
	theta_vertical(0.0f), theta_horizontal(1.5708f), theta_error(0.001f),
	min_scale(0.15f), max_scale(0.7f)
{ }

std::vector<cv::Mat> Extractor::extract(cv::Mat image, int (&sudoku)[NUM_CELLS])
{
	lines.clear();
	contours.clear();
	hierarchy.clear();

	std::vector<cv::Mat> digits;

	cv::vector<cv::Vec2f> unfiltered_lines;
	cv::vector<cv::vector<cv::Point>> unfiltered_contours;
	
	// thicken lines
	cv::dilate(image, result_image, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
	// double blur to remove noise
	blur( result_image, result_image, cv::Size(3,3) );
	//blur( result_image, result_image, cv::Size(3,3) );
	// Canny to detect edges
	cv::Canny(result_image, result_image, low_threshold, high_threshold, kernel_size);

	// Hough transform to detect lines
	//cv::HoughLines(result_image, unfiltered_lines, 1, CV_PI / 180, 100, 0, 0);
	//filterLines(unfiltered_lines);

	cv::GaussianBlur(result_image, result_image, cv::Size(3, 3), 3);
	
	// find digit contours
	cv::findContours( result_image, unfiltered_contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	filterContours(unfiltered_contours, sudoku);

	// show result_image
	/*cv::Mat dst;
	dst = cv::Scalar::all(0);
	image.copyTo( dst, result_image);
	cv::namedWindow( "result", cv::WINDOW_AUTOSIZE );
    cv::imshow( "result", dst );*/

	// draw intermediate steps
	//drawLines(lines, dst);
	drawContours(contours, image);

	// get all digits
	// skip 0(grid)
	cv::Size size(WIDTH, HEIGHT);
	cv::Mat src;
	for (size_t i = 1; i < contours.size(); ++i) {
		src = cv::Mat(image, cv::boundingRect(contours[i]));
		cv::resize(src, src, size);
		digits.push_back(src);
	}

	return digits;
}

void Extractor::filterLines(cv::vector<cv::Vec2f> & unfiltered)
{
	for (size_t i = 0; i < unfiltered.size(); ++i) {
		if ((unfiltered[i][1] > theta_vertical - theta_error && 
			unfiltered[i][1] < theta_vertical + theta_error) ||
			(unfiltered[i][1] > theta_horizontal - theta_error && 
			unfiltered[i][1] < theta_horizontal + theta_error)) {

			lines.push_back(unfiltered[i]);
		}
	}
}

void Extractor::filterContours(cv::vector<cv::vector<cv::Point>> & unfiltered, int (&sudoku)[NUM_CELLS])
{
	if (!unfiltered.size()) {
		return;
	}
	// get biggest contour
	// assume its sudoku grid
	int max_index = 0;
	int max_area = cv::boundingRect(unfiltered[0]).area();
	for (size_t i = 1; i < unfiltered.size(); ++i) {
		if (cv::boundingRect(unfiltered[i]).area() > max_area) {
			max_index = i;
			max_area = cv::boundingRect(unfiltered[i]).area();
		}
	}
	contours.push_back(unfiltered[max_index]);

	cv::Rect grid = cv::boundingRect(unfiltered[max_index]);
	int width = grid.width / 9;
	int height = grid.height / 9;

	for (int i = 0; i < NUM_CELLS; ++i) {
		max_indexes[i] = -1;
		max_areas[i] = -1;
	}

	cv::Rect cell;
	int index;
	int index_x;
	int index_y;
	for (size_t i = 0; i < unfiltered.size(); ++i) {
		cell = cv::boundingRect(unfiltered[i]);
		// x part of index
		index_x = (cell.x - grid.x) / width;
		// y part of index
		index_y = (cell.y - grid.y) / height;
		
		if (cell.x > grid.x && cell.x < grid.x + grid.width &&
			cell.y > grid.y && cell.y < grid.y + grid.height && 
			cell.x + cell.width < grid.x + (index_x+1) * width &&
			cell.y + cell.height < grid.y + (index_y+1) * height) {
				
				index = index_x + 9 * index_y;
				if (cell.area() > max_areas[index] && index < 81 && index >= 0
					&& cell.width < width * max_scale && cell.height < height * max_scale
					&& cell.width > width * min_scale  && cell.height > height * min_scale) {

					max_indexes[index] = i;
					max_areas[index] = cell.area();
				}
		}
	}

	for (size_t i = 0; i < NUM_CELLS; ++i) {
		if (max_indexes[i] > -1) {
			contours.push_back(unfiltered[max_indexes[i]]);
			// digit
			sudoku[i] = -1;
		} else {
			// no digit
			sudoku[i] = 0;
		}
	}
}

void Extractor::drawLines(cv::vector<cv::Vec2f> & lines, cv::Mat & dst) const
{
	float rho;
	float theta;
	cv::Point pt1, pt2;
	double a;
	double b;
	double x0;
	double y0;
	for( size_t i = 0; i < lines.size(); i++ )
	{
		rho = lines[i][0];
		theta = lines[i][1];
		a = cos(theta);
		b = sin(theta);
		x0 = a*rho;
		y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		line( dst, pt1, pt2, cv::Scalar(255, 255, 255), 3, CV_AA);
	}
	cv::namedWindow( "lines", cv::WINDOW_AUTOSIZE );
    cv::imshow( "lines", dst );
}

void Extractor::drawContours(cv::vector<cv::vector<cv::Point>> & contours, cv::Mat & dst) const
{
	for (size_t i = 0; i < contours.size(); ++i) {
		rectangle(dst, cv::boundingRect(contours[i]), cv::Scalar(0, 255, 0), 1, CV_AA);
	}

	cv::namedWindow( "digits", cv::WINDOW_AUTOSIZE );
    cv::imshow( "digits", dst );
}