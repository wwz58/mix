#include "mix.h"
#define DEBUG
/*mix::Params::Params()
{
	Params(false, false, 0, KCF_M);
}*/
mix::Params::Params() {
	use_hog = 1;
	use_resize = 1,
	m_stdHeight = 200;
	trackerName = KCF_M;
	offset = 0.2;
	alpha = 0.2;
	beta = 0.2;
	m_ShowNewImg = 1;
	thresh = 0.17;
}

mix::Params::Params(bool _use_hog, bool _use_resize, int _m_stdHeight, MODE _trackerName, float _offset, float _alpha, float _beta, bool _m_ShowNewImg, float _thresh)
:trackerName (_trackerName),
use_hog ( _use_hog),
offset (_offset),
alpha ( _alpha),
beta ( _beta),

use_resize ( _use_resize),
m_stdHeight ( _m_stdHeight),
m_ShowNewImg (_m_ShowNewImg),

thresh (_thresh)
{}
	

//params使用默认构造函-->tracker::paramsuse default constructor-->tracker使用默认构造函 detector直接使用默认构造函64*128,或在mix构造函后使用下面的构造函
mix::mix()
{
	//构造detector 什么都不写直接使用默认构造函64*128,或在mix构造函后使用下面的构造函
	//detector(cv::Size(48,96),cv::Size(16,16),cv::Size(8,8),cv::Size(8,8),9,1,-1,cv::HOGDescriptor::L2Hys,0.2,true,cv::HOGDescriptor::DEFAULT_NLEVELS);
	if (params.use_hog)
		hogDetector.setSVMDetector(hogDetector.getDefaultPeopleDetector());//得到检测器
	else
		cascadeDetector.load("D:\\opencv341\\build_no_contrib\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml");
	//cascadeDetector.load("D:\\graduate\\proj\\tracking\\conf\\haarcascade_mcs_upperbody.xml");
	switch (params.trackerName) {
	case KCF:
		tracker = TrackerKCF::create();
		break;
	case MOSSE:
		tracker = TrackerMOSSE::create();
		break;
	case CSRT:
		tracker = TrackerCSRT::create();
		break;
	case KCF_M:
		tracker_m = KCFTracker(true, false, true, true);
		break;
	}

	first = 1;
	countk = 0;
	getit = false;
	gowhere = 0;
}

mix::mix(bool _use_hog, bool _use_resize, int _m_stdHeight, MODE _trackerName, float _offset, float _alpha, float _beta, bool _m_ShowNewImg, float _thresh)
:params(_use_hog,
	_use_resize,
	_m_stdHeight,

	 _trackerName = KCF,

	_offset = 0.2,
	_alpha = 0.2,
	_beta = 0.2,

	_m_ShowNewImg = 1,

	_thresh = 0.17
)
{}

int mix::go1(Mat & frame, Mat & img, Rect2d & x, vector<Rect>& hogx)
{
	if (params.use_resize) {
		int m_stdWidth = 1.0 *params.m_stdHeight / frame.rows*frame.cols;
		resize(frame, img, Size(m_stdWidth, params.m_stdHeight), 0.0, 0.0, INTER_CUBIC);
	}
	else
		frame.copyTo(img);

	vector<Rect> found_filtered;
	detect(img, found_filtered, params.offset);

	if (!found_filtered.empty()) {
		getit = true;
		hogx = found_filtered;

		if (found_filtered.size() == 1) {
			x = found_filtered[0];
			box = Rect2d(found_filtered[0]);
			if (params.trackerName == KCF_M)
				tracker_m.init(box, img);
			else
				tracker->init(img, box);
			if (first) {
				first = 0;
				//w_init = box.width;
				getHist(img, Rect(box), std_Hhist);
			}
		}

		else {
			if (first) {
				first = 0;
				int idx = selectCenterRect(found_filtered, img);
				x = found_filtered[idx];
				box = Rect2d(found_filtered[idx]);
				if (params.trackerName == KCF_M)
					tracker_m.init(box, img);
				else
					tracker->init(img, box);

				//w_init = box.width;
				getHist(img, Rect(box), std_Hhist);
			}
			else {
				//int idx = selectSimilarColorRect(img, found_filtered, std_Hhist);
				int idx = selectNearRect(found_filtered);

				x = found_filtered[idx];
				box = Rect2d(found_filtered[idx]);
				if (params.trackerName == KCF_M)
					tracker_m.init(box, img);
				else
					tracker->init(img, box);
			}
		}
	}
	else {
		//hoggetit = false;
		if (!getit) {
			first = 1;
			getit = false;
		}
		else {

			if (params.trackerName == KCF_M) {
				Rect roi = box;
				float peak = tracker_m.update(img, roi);
				box = Rect2d(roi);
				getit = (peak > params.thresh);
				x = box;
			}
			else {
				getit = tracker->update(img, box);
				x = box;
			}
		}
	}
	return getit;
}

int mix::go2(Mat & frame, Mat & img, Rect2d & x, vector<Rect>& hogx)
{
	if (params.use_resize) {
		int m_stdWidth = 1.0 *params.m_stdHeight / frame.rows*frame.cols;
		resize(frame, img, Size(m_stdWidth, params.m_stdHeight), 0.0, 0.0, INTER_CUBIC);
	}
	else
		frame.copyTo(img);

	if (!getit) {
		vector<Rect> found_filtered;
		detect(img, found_filtered, params.offset);
		if (!found_filtered.empty()) {
			getit = 1;
			hogx = found_filtered;
			if (found_filtered.size() == 1) {
				box = found_filtered[0];
				x = box;

				if (params.trackerName == KCF_M)
					tracker_m.init(box, img);
				else
					tracker->init(img, box);
			}
			else {
				int idx = selectCenterRect(found_filtered, img);
				box = found_filtered[0];
				x = box;
#ifdef DEBUG
				cout << box.x << "\n" << box.y << "\n" << box.width << "\n" << box.height << "\n";
#endif
				if (params.trackerName == KCF_M)
					tracker_m.init(box, img);
				else
					tracker->init(img, box);
#ifdef DEBUG
				cout << "init success\n";
#endif
			}
		}
		if (first) {
			first = 0;
			w_init = box.area();
		}
	}
	else {
		
		if (params.trackerName == KCF_M) {
			Rect roi = box;
			float peak = tracker_m.update(img, roi);
#ifdef DEBUG
			cout << "update success\n" << roi.x << "\n" << roi.y << "\n" << roi.width << "\n" << roi.height << "\n";
#endif
			if ((roi.area() > w_init * 1.3)|| (roi.area() < w_init / 1.5))
				getit = 0;
			else {
				box = Rect2d(roi);
				getit = (peak > params.thresh);
				x = box;
			}
		}
		else {
			getit = tracker->update(img, box);
			x = box;
		}
	}
	return getit;
}


void mix::detect(Mat & img, vector<Rect>& found_filtered, float offset)
{
	Rect r(offset*img.cols, 0, img.cols*(1 - 2 * offset), img.rows);
	Mat clip(img, r);
	vector<Rect> found;

	if (params.use_hog)
		hogDetector.detectMultiScale(clip, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
	else
		cascadeDetector.detectMultiScale(img, found, 1.1, 3, 0, Size(10, 10));

	for (vector<Rect>::iterator iter = found.begin(); iter != found.end(); ++iter) {
		if (iter->width < 3 || iter->height < 3)
			found.erase(iter);
	}

	int i, j;
	for (i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		for (j = 0; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}
	//printf("tdetection time = %gms, found_filtered size:%d\n", t*1000./cv::getTickFrequency(),found_filtered.size());

	if (params.use_hog) {
		//slightly shrink the HOG rectangles
		for (i = 0; i < found_filtered.size(); i++)
		{
			Rect &r = found_filtered[i];
			r.x += cvRound(r.width*params.alpha) + params.offset*img.cols;
			r.width = cvRound(r.width*(1 - 2 * params.alpha));
			r.y += cvRound(r.height*params.beta);
			r.height = cvRound(r.height*(1 - 2 * params.beta));
			//rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 5);
		}
	}
}

int mix::calcRectsDist(Rect & r1, Rect & r2)
{
	Point c1 = (r1.tl() + r1.br()) / 2;
	Point c2 = (r2.tl() + r2.br()) / 2;
	Point diff = c1 - c2;
	int dis = diff.x*diff.x + diff.y*diff.y;
	return dis;
}

int mix::selectCenterRect(vector<Rect>& rects, Mat& img)
{
	Rect imgRect(0, 0, img.cols, img.rows);
	int minDist = calcRectsDist(imgRect, rects[0]);
	int idx = 0;
	for (int i = 1; i < rects.size(); i++) {
		int dist = calcRectsDist(imgRect, rects[i]);
		if (minDist > dist) {
			minDist = dist;
			idx = i;
		}
	}
	return idx;
}

int mix::selectNearRect(vector<Rect>& rects)
{
	int minDist = calcRectsDist(Rect(box), rects[0]);
	int idx = 0;
	for (int i = 1; i < rects.size(); i++) {
		int dist = calcRectsDist(Rect(box), rects[i]);
		if (minDist > dist) {
			minDist = dist;
			idx = i;
		}
	}
	return idx;
}

int mix::selectSimilarColorRect(Mat & image, vector<Rect>& rects, MatND & std_hist, int n)
{
	//int n=1;
	//直方图hist1和hist2的比较
	//n=1用的是Correlation，c1越大表示相似度越高
	//n=2用的是Chi-square，c1越小表示相似度越高
	//n=3用的是Intersection，c1越大表示相似度越高
	//n=4用的是Bhattacharyya距离，c1越小表示相似度越高
	//cout << hist1.type()<<":"<<hist2.type()<<" ";
	//cout << hist1.depth()<<":"<<hist2.depth()<<" ";
	MatND hist;
	int idx = 0;
	getHist(image, rects[0], hist);
	double minDist = compareHist(std_Hhist, hist, n);
	for (int i = 1; i < rects.size(); i++) {
		getHist(image, rects[0], hist);
		double dist = compareHist(std_Hhist, hist, n);
		if (dist < minDist) {
			minDist = dist;
			idx = i;
		}
	}
	return idx;
}

void mix::getHist(Mat & image, Rect & b, MatND & Hhist)
{
	Mat img(image, b);
#ifdef DEBUG
	cout << b.x << b.y << b.area() << endl;
#endif // DEBUG
	Mat hsv, hue;
	cvtColor(img, hsv, CV_BGR2HSV);
	hue.create(hsv.size(), hsv.depth());
	int ch[] = { 0, 0 };
	mixChannels(&hsv, 1, &hue, 1, ch, 1);

	int hsize = 16;
	float hranges[] = { 0,180 };
	const float* phranges = hranges;
	calcHist(&hue, 1, 0, Mat(), Hhist, 1, &hsize, &phranges);
	normalize(Hhist, Hhist, 0, 255, CV_MINMAX);
#ifdef DEBUG_
	{
		//show hist img
		Mat histimg = Mat::zeros(200, 320, CV_8UC3);
		int binW = histimg.cols / hsize;
		Mat buf(1, hsize, CV_8UC3);
		for (int i = 0; i < hsize; i++)
			buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180. / hsize), 255, 255);
		cvtColor(buf, buf, CV_HSV2BGR);

		for (int i = 0; i < hsize; i++)
		{
			int val = saturate_cast<int>(Hhist.at<float>(i)*histimg.rows / MaxHeight);
			rectangle(histimg, Point(i*binW, histimg.rows),
				Point((i + 1)*binW, histimg.rows - val),
				Scalar(buf.at<Vec3b>(i)), -1, 8);
		}
		imshow("hist", histimg);
	}
#endif
}
