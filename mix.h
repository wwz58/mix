#pragma once
#include <opencv2\core\core.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\tracking\tracking.hpp>
#include <vector>
#include "kcftracker.hpp"
using namespace std;
using namespace cv;
class mix {
public:
	enum MODE {
		KCF = 1,
		CSRT = 2,
		MOSSE = 3,
		KCF_M = 4
	};
	struct Params {
		//TrackerKCF::Params kcf_params;
		//TrackerCSRT::Params csrt_params;

		/*struct detector_params {
			Size _winSize;
			Size _blockSize;
			Size _blockStride;
			Size _cellSize;
			int _nbins;
			int _derivAperture;
			double _winSigma;
			int _histogramNormType;
			double _L2HysThreshold;
			bool _gammaCorrection;
			int _nlevels;
			bool _signedGradient;
		};*/
		bool use_hog;
		bool use_resize;
		int m_stdHeight;
		MODE trackerName;

		float offset;
		float alpha;
		float beta;

		bool m_ShowNewImg;

		float thresh;
		Params();
		Params(
		bool _use_hog, 
		bool _use_resize,
		int _m_stdHeight,

		MODE _trackerName = KCF,

		float _offset=0.2,
		float _alpha=0.2,
		float _beta=0.2,

		bool _m_ShowNewImg=1,

		float _thresh=0.17
		);

	};

	mix();
	mix(bool _use_hog,
		bool _use_resize,
		int _m_stdHeight,

		MODE _trackerName = KCF_M,

		float _offset = 0.2,
		float _alpha = 0.2,
		float _beta = 0.2,

		bool _m_ShowNewImg = 1,

		float _thresh = 0.17
	);
	int go1(Mat & frame, Mat & img, Rect2d & x, vector <Rect> & hogx);
	int go2(Mat & frame, Mat & img, Rect2d & x, vector <Rect> & hogx);

protected:
	Params params;
	//Ptr<TrackerKCF> tracker;
	Ptr<Tracker> tracker;
	KCFTracker tracker_m;
	HOGDescriptor hogDetector;
	CascadeClassifier cascadeDetector;
	Rect2d box;
	bool getit; 
	bool hoggetit;
	int gowhere;//2,detected,but not go; 1, detected, go; 0, not detected; -1, go back.

	int countk;
	int first;
	int w_init;
	//Rect pre_rect;
	Rect roi;
	MatND std_Hhist;

	void detect(Mat& img, vector<Rect>& found_filtered,float offset);
	//void setShowNewImg(bool);
	//void change(Rect & b, Rect & x, float scale);
	int calcRectsDist(Rect&r1, Rect&r2);
	int  selectCenterRect(vector<Rect>& rects,Mat& img);
	int  selectNearRect(vector<Rect>& rects);

	int  selectSimilarColorRect(Mat& image, vector<Rect>& rects, MatND &std_hist, int n = 1);
	void getHist(Mat & image, Rect &b, MatND & Hhist);

};
