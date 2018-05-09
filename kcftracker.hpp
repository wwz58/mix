#pragma once  

#include "tracker.h"  

#ifndef _OPENCV_KCFTRACKER_HPP_  
#define _OPENCV_KCFTRACKER_HPP_  
#endif  

class KCFTracker : public XBTracker
{
public:
	// Constructor  
	// ����KCF����������  
	KCFTracker(bool hog = true, bool fixed_window = true, bool multiscale = true, bool lab = true);

										// Initialize tracker  
										// ��ʼ���������� roi ��Ŀ���ʼ������ã� image �ǽ�����ٵĵ�һ֡ͼ��  
	virtual void init(const cv::Rect &roi, cv::Mat image);

	// Update position based on the new frame  
	// ʹ����һ֡����ͼ�� image ����һ֡ͼ��  
	virtual cv::Rect update(cv::Mat image);
	float update(cv::Mat image,cv::Rect & newroi);

	float interp_factor;        // linear interpolation factor for adaptation  
								// ����Ӧ�����Բ�ֵ���ӣ�����Ϊhog��lab��ѡ����仯  
	float sigma;                // gaussian kernel bandwidth  
								// ��˹����˴�������Ϊhog��lab��ѡ����仯  
	float lambda;               // regularization  
								// ���򻯣�0.0001  
	int cell_size;              // HOG cell size  
								// HOGԪ������ߴ磬4  
	int cell_sizeQ;             // cell size^2, to avoid repeated operations  
								// Ԫ��������������Ŀ��16��Ϊ�˼���ʡ��  
	float padding;              // extra area surrounding the target  
								// Ŀ����չ����������2.5  
	float output_sigma_factor;  // bandwidth of gaussian target  
								// ��˹Ŀ��Ĵ�����ͬhog��lab�᲻ͬ  
	int template_size;          // template size  
								// ģ���С���ڼ���_tmpl_szʱ��  
								// �ϴ��ɱ���һ��96������С�߳���������С  
	float scale_step;           // scale step for multi-scale estimation  
								// ��߶ȹ��Ƶ�ʱ��ĳ߶Ȳ���  
	float scale_weight;         // to downweight detection scores of other scales for added stability  
								// Ϊ�����������߶ȼ��ʱ���ȶ��ԣ����������ֵ��һ��˥����Ϊԭ����0.95��  

protected:
	// Detect object in the current frame.  
	// ��⵱ǰ֡��Ŀ��  
	//z��ǰһ���ѵ��/��һ֡�ĳ�ʼ������� x�ǵ�ǰ֡��ǰ�߶��µ������� peak_value�Ǽ������ֵ  
	cv::Point2f detect(cv::Mat z, cv::Mat x, float &peak_value);

	// train tracker with a single image  
	// ʹ�õ�ǰͼ��ļ��������ѵ��  
	// x�ǵ�ǰ֡��ǰ�߶��µ������� train_interp_factor��interp_factor  
	void train(cv::Mat x, float train_interp_factor);

	// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y,  
	// which must both be MxN. They must also be periodic (ie., pre-processed with a cosine window).  
	// ʹ�ô���SIGMA�����˹���������������ͼ��X��Y֮������λ��  
	// ���붼��MxN��С�����߱��붼�����ڵģ�����ͨ��һ��cos���ڽ���Ԥ����  
	cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2);

	// Create Gaussian Peak. Function called only in the first frame.  
	// ������˹�庯��������ֻ�ڵ�һ֡��ʱ��ִ��  
	cv::Mat createGaussianPeak(int sizey, int sizex);

	// Obtain sub-window from image, with replication-padding and extract features  
	// ��ͼ��õ��Ӵ��ڣ�ͨ����ֵ��䲢�������  
	cv::Mat getFeatures(const cv::Mat & image, bool inithann, float scale_adjust = 1.0f);

	// Initialize Hanning window. Function called only in the first frame.  
	// ��ʼ��hanning���ڡ�����ֻ�ڵ�һ֡��ִ�С�  
	void createHanningMats();

	// Calculate sub-pixel peak for one dimension  
	// ����һά�����ط�ֵ  
	float subPixelPeak(float left, float center, float right);

	cv::Mat _alphaf;            // ��ʼ��/ѵ�����alphaf�����ڼ�ⲿ���н���ļ���  
	cv::Mat _prob;              // ��ʼ�����prob�����ٸ��ģ�����ѵ��  
	cv::Mat _tmpl;              // ��ʼ��/ѵ���Ľ��������detect��z  
	cv::Mat _num;               // ����ע�͵���  
	cv::Mat _den;               // ����ע�͵���  
	cv::Mat _labCentroids;      // lab��������  

private:
	int size_patch[3];          // hog������sizeY��sizeX��numFeatures  
	cv::Mat hann;               // createHanningMats()�ļ�����  
	cv::Size _tmpl_sz;          // hogԪ����Ӧ�������С  
	float _scale;               // ������_tmpl_sz��ĳ߶ȴ�С  
	int _gaussian_size;         // δ���ã�����  
	bool _hogfeatures;          // hog��־λ  
	bool _labfeatures;          // lab��־λ  
};
