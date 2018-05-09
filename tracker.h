#pragma once
#include <opencv2/opencv.hpp>  
#include <string>  

class XBTracker
{
public:
	XBTracker() {}
	virtual  ~XBTracker() { }

	virtual void init(const cv::Rect &roi, cv::Mat image) = 0;
	virtual cv::Rect  update(cv::Mat image) = 0;


protected:
	cv::Rect_<float> _roi;
};

