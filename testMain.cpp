#include "opencv2/video/tracking.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "mix.h"
#include <numeric>
using namespace cv;
using namespace std;

int main()
{
	VideoCapture cap; //定义一个摄像头捕捉的类对象   labOriginal
	//cap.open(0);
	cap.open("D:\\graduate\\video\\allMy\\good-ok-wwz.mp4");
	//cap.open("D:\\graduate\\video\\allMy\\labOriginal.mp4");
	//cap.open("D:\\graduate\\video\\dateset\\5\\scale_luo (1).mp4");
	if (!cap.isOpened())
	{
		cout << "***Could not initialize capturing...***\n";
		return -1;
	}

	//namedWindow("scaled", WINDOW_NORMAL);
	Rect2d x;
	vector<Rect> hogx;
	Mat frame,img;
	mix mixTracker;

	int64 t1, t2, tick, tick_counter = 0;
	char fpsStr[10];
	double fps;
	vector<double> fpss;
	int frameIdx = 1;

	for (;;) {
		cap >> frame;
		frameIdx++;
		if (frame.empty()) {
			double mean1 = std::accumulate(std::begin(fpss) + 1, std::begin(fpss) + 900, 0.0) / 900;
			double mean2 = std::accumulate(std::begin(fpss) + 901, std::begin(fpss) + 1800, 0.0) / 900;
			double mean3 = std::accumulate(std::begin(fpss) + 1801, std::begin(fpss) + 2700, 0.0) / 900;
			double mean4 = std::accumulate(std::begin(fpss) + 2701, std::begin(fpss) + 3600, 0.0) / 900;

			cout << "mean1:" << mean1 << endl;
			cout << "mean2:" << mean2 << endl;
			cout << "mean3:" << mean3 << endl;
			cout << "mean4:" << mean4 << endl; 
			break; 
		}

		t1 = cv::getTickCount();
		//int getit = mixTracker.go1(frame, img, x, hogx);
		int getit = mixTracker.go2(frame, img, x, hogx);

		t2 = cv::getTickCount();
		tick = t2 - t1;
		tick_counter += tick;

		fps = getTickFrequency() / double(tick);
		fpss.push_back(fps);

		putText(img, "mix", Point(x.tl().x, x.tl().y), FONT_HERSHEY_PLAIN, 1.0, Scalar( 0, 0, 255),2);
		putText(img, to_string(frameIdx), Point(10, 10), 1, 1.0, Scalar(0, 0, 255));

		sprintf_s(fpsStr, "%.2f", fps);
		std::string fpsString("FPS:");
		fpsString += fpsStr;
		putText(img, fpsString, Point(0,20), FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 0, 0),2);

		if (getit){
			rectangle(img, x, Scalar(0, 0, 255), 2);
		}
		else {
			putText(img, "lost, please stand in center", Point(10, 10), 1, 1.0, Scalar(0, 0, 255), 2);
			for (int i = 0; i < hogx.size(); ++i)
				rectangle(img, hogx[i], Scalar(0, 255, 0), 1);
		}
		imshow("tracking", img);
		imwrite("D:\\graduate\\video\\dateset\\10\\result\\MIX\\wwz\\" + to_string(frameIdx) + ".jpg", img);
		waitKey(1);
	}
	getchar();
	return 0;
}