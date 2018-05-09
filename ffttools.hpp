#pragma once  

#include <opencv2/opencv.hpp>  

#ifndef _OPENCV_FFTTOOLS_HPP_  
#define _OPENCV_FFTTOOLS_HPP_  
#endif  

//NOTE: FFTW support is still shaky, disabled for now.  
/*#ifdef USE_FFTW
#include <fftw3.h>
#endif*/

namespace FFTTools
{
	// Previous declarations, to avoid warnings  
	cv::Mat fftd(cv::Mat img, bool backwards = false);
	cv::Mat real(cv::Mat img);
	cv::Mat imag(cv::Mat img);
	cv::Mat magnitude(cv::Mat img);
	cv::Mat complexMultiplication(cv::Mat a, cv::Mat b);
	cv::Mat complexDivision(cv::Mat a, cv::Mat b);
	void rearrange(cv::Mat &img);
	void normalizedLogTransform(cv::Mat &img);


	// ��dfft  
	cv::Mat fftd(cv::Mat img, bool backwards)
	{
		if (img.channels() == 1)
		{
			cv::Mat planes[] = { cv::Mat_<float>(img), cv::Mat_<float>::zeros(img.size()) };
			//cv::Mat planes[] = {cv::Mat_<double> (img), cv::Mat_<double>::zeros(img.size())};  
			cv::merge(planes, 2, img);
		}
		cv::dft(img, img, backwards ? (cv::DFT_INVERSE | cv::DFT_SCALE) : 0);

		return img;
	}

	// ͼ��ʵ��  
	cv::Mat real(cv::Mat img)
	{
		std::vector<cv::Mat> planes;
		cv::split(img, planes);
		return planes[0];
	}

	// ͼ���鲿  
	cv::Mat imag(cv::Mat img)
	{
		std::vector<cv::Mat> planes;
		cv::split(img, planes);
		return planes[1];
	}

	// ������ģ  
	cv::Mat magnitude(cv::Mat img)
	{
		cv::Mat res;
		std::vector<cv::Mat> planes;
		cv::split(img, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I)) ʵ���鲿�ֿ�  
		if (planes.size() == 1) res = cv::abs(img);
		else if (planes.size() == 2) cv::magnitude(planes[0], planes[1], res); // planes[0] = magnitude ������ģ��sqrt(x(I)^2+y(I)^2)��  
		else assert(0);
		return res;
	}

	// �����˷�  
	cv::Mat complexMultiplication(cv::Mat a, cv::Mat b)
	{
		std::vector<cv::Mat> pa;
		std::vector<cv::Mat> pb;
		cv::split(a, pa);       //ͨ�����  
		cv::split(b, pb);       //ͨ�����  

		std::vector<cv::Mat> pres;
		pres.push_back(pa[0].mul(pb[0]) - pa[1].mul(pb[1]));
		pres.push_back(pa[0].mul(pb[1]) + pa[1].mul(pb[0]));

		cv::Mat res;
		cv::merge(pres, res);       //�ϲ�ͨ��  

		return res;
	}

	// ��������  
	cv::Mat complexDivision(cv::Mat a, cv::Mat b)
	{
		std::vector<cv::Mat> pa;
		std::vector<cv::Mat> pb;
		cv::split(a, pa);       // ͨ�����  
		cv::split(b, pb);       // ͨ�����  

		cv::Mat divisor = 1. / (pb[0].mul(pb[0]) + pb[1].mul(pb[1]));       // ʵ���鲿�������  

		std::vector<cv::Mat> pres;

		pres.push_back((pa[0].mul(pb[0]) + pa[1].mul(pb[1])).mul(divisor));
		pres.push_back((pa[1].mul(pb[0]) + pa[0].mul(pb[1])).mul(divisor));

		cv::Mat res;
		cv::merge(pres, res);       // �ϲ�ͨ��  
		return res;
	}

	// �������  
	// 0 1 ���-> 3 2  
	// 2 3       1 0  
	void rearrange(cv::Mat &img)
	{
		// img = img(cv::Rect(0, 0, img.cols & -2, img.rows & -2));  
		int cx = img.cols / 2;
		int cy = img.rows / 2;

		cv::Mat q0(img, cv::Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant  
		cv::Mat q1(img, cv::Rect(cx, 0, cx, cy)); // Top-Right  
		cv::Mat q2(img, cv::Rect(0, cy, cx, cy)); // Bottom-Left  
		cv::Mat q3(img, cv::Rect(cx, cy, cx, cy)); // Bottom-Right  

		cv::Mat tmp; // swap quadrants (Top-Left with Bottom-Right)  
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);
		q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)  
		q2.copyTo(q1);
		tmp.copyTo(q2);
	}
	/*
	template < typename type>
	cv::Mat fouriertransFull(const cv::Mat & in)
	{
	return fftd(in);

	cv::Mat planes[] = {cv::Mat_<type > (in), cv::Mat_<type>::zeros(in.size())};
	cv::Mat t;
	assert(planes[0].depth() == planes[1].depth());
	assert(planes[0].size == planes[1].size);
	cv::merge(planes, 2, t);
	cv::dft(t, t);

	//cv::normalize(a, a, 0, 1, CV_MINMAX);
	//cv::normalize(t, t, 0, 1, CV_MINMAX);

	// cv::imshow("a",real(a));
	//  cv::imshow("b",real(t));
	// cv::waitKey(0);

	return t;
	}*/

	// �������任��ʼ��  
	void normalizedLogTransform(cv::Mat &img)
	{
		img = cv::abs(img);
		img += cv::Scalar::all(1);
		cv::log(img, img);
		// cv::normalize(img, img, 0, 1, CV_MINMAX);  
	}

}
