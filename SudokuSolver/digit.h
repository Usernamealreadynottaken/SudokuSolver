#include <opencv2/core/core.hpp>

class Digit
{
private:
	int value;
	cv::Mat image;
protected:
public:
	Digit() { };
	Digit(int v, cv::Mat & i) : value(v), image(i) { }
	~Digit() { }
	void setValue(int v) { value = v; }
	int getValue() { return value; }
	void setImage(cv::Mat i) { image = i; }
	cv::Mat getImage() { return image; }
};