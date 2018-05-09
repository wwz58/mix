#ifndef _KCFTRACKER_HEADERS  
#include "kcftracker.hpp"  
#include "ffttools.hpp"  
#include "recttools.hpp"  
#include "fhog.hpp"  
#include "labdata.hpp"  
#endif  

// Constructor  
// ��ʼ��KCF�����  
KCFTracker::KCFTracker(bool hog,               
	bool fixed_window,       
	bool multiscale,          
	bool lab)
{

	// Parameters equal in all cases  
	lambda = 0;
	padding = 2;
	//output_sigma_factor = 0.1;  
	output_sigma_factor = 0.01;


	if (hog) {    // HOG  
				  // VOT  
		interp_factor = 0.0125;
		sigma = 0.6;
		// TPAMI  
		//interp_factor = 0.02;  
		//sigma = 0.5;   
		cell_size = 4;
		_hogfeatures = true;

		if (lab) {
			interp_factor = 0.005;
			sigma = 0.4;
			//output_sigma_factor = 0.025;  
			output_sigma_factor = 0.1;

			_labfeatures = true;
			_labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &data);
			cell_sizeQ = cell_size*cell_size;
		}
		else {
			_labfeatures = false;
		}
	}
	else {   // RAW  
		interp_factor = 0.075;
		sigma = 0.2;
		cell_size = 1;
		_hogfeatures = false;

		if (lab) {
			printf("Lab features are only used with HOG features.\n");
			_labfeatures = false;
		}
	}


	if (multiscale) { // multiscale  
		template_size = 96;
		//template_size = 100;  
		scale_step = 1.20;//1.05;  
		scale_weight = 0.95;
		if (!fixed_window) {
			//printf("Multiscale does not support non-fixed window.\n");  
			fixed_window = true;
		}
	}
	else if (fixed_window) {  // fit correction without multiscale  
		template_size = 96;
		//template_size = 100;  
		scale_step = 1;
	}
	else {
		template_size = 1;
		scale_step = 1;
	}
}

// Initialize tracker   
// ʹ�õ�һ֡�����ĸ��ٿ򣬳�ʼ��KCF������  
void KCFTracker::init(const cv::Rect &roi, cv::Mat image)
{
	_roi = roi;
	assert(roi.width >= 0 && roi.height >= 0);
	_tmpl = getFeatures(image, 1);                                                                              // ��ȡ��������train����ÿ֡�޸�  
	_prob = createGaussianPeak(size_patch[0], size_patch[1]);                           // ������޸��ˣ�ֻ��ʼ��һ��  
	_alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));    // ��ȡ��������train����ÿ֡�޸�  
																			//_num = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));  
																			//_den = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));  
	train(_tmpl, 1.0); // train with initial frame  
}

// Update position based on the new frame  
// ���ڵ�ǰ֡����Ŀ��λ��  
cv::Rect  KCFTracker::update(cv::Mat image)
{
	// �����߽�  
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

	// ���ٿ�����  
	float cx = _roi.x + _roi.width / 2.0f;
	float cy = _roi.y + _roi.height / 2.0f;

	
	cv::Point2f res;
	if (scale_step > 1.0) {
		float search_size[5] = { 1.0f,0.95f, 0.9f, 1.05f, 1.1f };
		//float search_size[7] = { 1.0f,0.985f, 0.99f, 0.995f, 1.005f, 1.01f, 1.015f };
		std::vector <float> peak_value(5, 0.0f);
		std::vector<cv::Point2f>ress;

		for (int i = 0; i <= 4; ++i) {
			ress.push_back(detect(_tmpl, getFeatures(image, 0, search_size[i]), peak_value[i]));
		}
		std::vector<float>::iterator biggest = std::max_element(peak_value.begin(), peak_value.end());
		int idx = std::distance(peak_value.begin(), biggest);
		res = ress[idx];
		float best_scale_step = search_size[idx];
		_scale *= best_scale_step;
		_roi.width *= best_scale_step;
		_roi.height *= best_scale_step;
	}
	
	/*
	// �߶Ȳ���ʱ����ֵ���  
	float peak_value;
	cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value);
	
	// �Դ�߶Ⱥ���С�߶Ƚ��м��  
	if (scale_step != 1) { 
		// Test at a smaller _scale  
		// ʹ��һ��С��ĳ߶Ȳ���  
		float new_peak_value;
		cv::Point2f new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value);

		// �����滹��ͬ�߶ȴ����Ϊ��Ŀ��  
		if (scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			_scale /= scale_step;
			_roi.width /= scale_step;
			_roi.height /= scale_step;
		}

		// Test at a bigger _scale  
		new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value);

		if (scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			_scale *= scale_step;
			_roi.width *= scale_step;
			_roi.height *= scale_step;
		}
	}
	//cout << peak_value << endl;*/
	// Adjust by cell size and _scale  
	// ��Ϊ���ص�ֻ���������꣬ʹ�ó߶Ⱥ������������Ŀ���  
	_roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale);
	_roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale);

	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

	assert(_roi.width >= 0 && _roi.height >= 0);

	// ʹ�õ�ǰ�ļ�����ѵ����������  
	cv::Mat x = getFeatures(image, 0);
	train(x, interp_factor);

	return _roi;        //���ؼ���
	//return peak_value;
}

float KCFTracker::update(cv::Mat image,cv::Rect & newroi)
{
	// �����߽�  
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

	// ���ٿ�����  
	float cx = _roi.x + _roi.width / 2.0f;
	float cy = _roi.y + _roi.height / 2.0f;

	float best_peak;
	cv::Point2f res;
	if (scale_step > 1.0) {
		float search_size[5] = { 1.0f,0.95f, 0.9f, 1.05f, 1.1f };
		//float search_size[7] = { 1.0f,0.985f, 0.99f, 0.995f, 1.005f, 1.01f, 1.015f };
		std::vector <float> peak_value(5, 0.0f);
		std::vector<cv::Point2f>ress;

		for (int i = 0; i <= 4; ++i) {
			ress.push_back(detect(_tmpl, getFeatures(image, 0, search_size[i]), peak_value[i]));
		}
		std::vector<float>::iterator biggest = std::max_element(peak_value.begin(), peak_value.end());
		int idx = std::distance(peak_value.begin(), biggest);
		res = ress[idx];
		float best_scale_step = search_size[idx];
		best_peak = peak_value[idx];

		_scale *= best_scale_step;
		_roi.width *= best_scale_step;
		_roi.height *= best_scale_step;
	}
	else {
		cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), best_peak);
	}
	/*
	// �߶Ȳ���ʱ����ֵ���  
	float peak_value;
	cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value);
	
	// �Դ�߶Ⱥ���С�߶Ƚ��м��  
	if (scale_step != 1) { 
		// Test at a smaller _scale  
		// ʹ��һ��С��ĳ߶Ȳ���  
		float new_peak_value;
		cv::Point2f new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value);

		// �����滹��ͬ�߶ȴ����Ϊ��Ŀ��  
		if (scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			_scale /= scale_step;
			_roi.width /= scale_step;
			_roi.height /= scale_step;
		}

		// Test at a bigger _scale  
		new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value);

		if (scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			_scale *= scale_step;
			_roi.width *= scale_step;
			_roi.height *= scale_step;
		}
	}
	*/
	//cout << peak_value << endl;
	// Adjust by cell size and _scale  
	// ��Ϊ���ص�ֻ���������꣬ʹ�ó߶Ⱥ������������Ŀ���  
	_roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale);
	_roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale);

	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

	assert(_roi.width >= 0 && _roi.height >= 0);

	// ʹ�õ�ǰ�ļ�����ѵ����������  
	cv::Mat x = getFeatures(image, 0);
	train(x, interp_factor);

	newroi = _roi;        //���ؼ���
	return best_peak;
}


// Detect object in the current frame.  
// zΪǰһ֡����  
// xΪ��ǰ֡ͼ��  
// peak_valueΪ����ķ�ֵ  
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value)
{
	using namespace FFTTools;

	// ���任�õ�������res  
	cv::Mat k = gaussianCorrelation(x, z);
	cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));

	//minMaxLoc only accepts doubles for the peak, and integer points for the coordinates  
	// ʹ��opencv��minMaxLoc����λ��ֵ����λ��  
	cv::Point2i pi;
	double pv;
	cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
	peak_value = (float)pv;

	//subpixel peak estimation, coordinates will be non-integer  
	// �����ط�ֵ��⣬�����Ƿ����ε�  
	cv::Point2f p((float)pi.x, (float)pi.y);

	if (pi.x > 0 && pi.x < res.cols - 1) {
		p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
	}

	if (pi.y > 0 && pi.y < res.rows - 1) {
		p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
	}

	p.x -= (res.cols) / 2;
	p.y -= (res.rows) / 2;

	return p;
}

// train tracker with a single image  
// ʹ��ͼ�����ѵ�����õ���ǰ֡��_tmpl��_alphaf  
void KCFTracker::train(cv::Mat x, float train_interp_factor)
{
	using namespace FFTTools;

	cv::Mat k = gaussianCorrelation(x, x);
	cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));

	_tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor)* x;
	_alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor)* alphaf;


	/*cv::Mat kf = fftd(gaussianCorrelation(x, x));
	cv::Mat num = complexMultiplication(kf, _prob);
	cv::Mat den = complexMultiplication(kf, kf + lambda);

	_tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
	_num = (1 - train_interp_factor) * _num + (train_interp_factor) * num;
	_den = (1 - train_interp_factor) * _den + (train_interp_factor) * den;

	_alphaf = complexDivision(_num, _den);*/

}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y,  
// which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).  
// ʹ�ô���SIGMA�����˹���������������ͼ��X��Y֮������λ��  
// ���붼��MxN��С�����߱��붼�����ڵģ�����ͨ��һ��cos���ڽ���Ԥ����  
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
	using namespace FFTTools;
	cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
	// HOG features  
	if (_hogfeatures) {
		cv::Mat caux;
		cv::Mat x1aux;
		cv::Mat x2aux;
		for (int i = 0; i < size_patch[2]; i++) {
			x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug  
			x1aux = x1aux.reshape(1, size_patch[0]);
			x2aux = x2.row(i).reshape(1, size_patch[0]);
			cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
			caux = fftd(caux, true);
			rearrange(caux);
			caux.convertTo(caux, CV_32F);
			c = c + real(caux);
		}
	}
	// Gray features  
	else {
		cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
		c = fftd(c, true);
		rearrange(c);
		c = real(c);
	}
	cv::Mat d;
	cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) / (size_patch[0] * size_patch[1] * size_patch[2]), 0, d);

	cv::Mat k;
	cv::exp((-d / (sigma * sigma)), k);
	return k;
}

// Create Gaussian Peak. Function called only in the first frame.  
// ������˹�庯��������ֻ�ڵ�һ֡��ʱ��ִ��  
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
	cv::Mat_<float> res(sizey, sizex);

	int syh = (sizey) / 2;
	int sxh = (sizex) / 2;

	float output_sigma = std::sqrt((float)sizex * sizey) / padding * output_sigma_factor;
	float mult = -0.5 / (output_sigma * output_sigma);

	for (int i = 0; i < sizey; i++)
		for (int j = 0; j < sizex; j++)
		{
			int ih = i - syh;
			int jh = j - sxh;
			res(i, j) = std::exp(mult * (float)(ih * ih + jh * jh));
		}
	return FFTTools::fftd(res);
}

// Obtain sub-window from image, with replication-padding and extract features  
// ��ͼ��õ��Ӵ��ڣ�ͨ����ֵ��䲢�������  
cv::Mat KCFTracker::getFeatures(const cv::Mat & image, bool inithann, float scale_adjust)
{
	cv::Rect extracted_roi;

	float cx = _roi.x + _roi.width / 2;
	float cy = _roi.y + _roi.height / 2;

	// ��ʼ��hanning���� ��ʵִֻ��һ�Σ�ֻ�ڵ�һ֡��ʱ��inithann=1  
	if (inithann) {
		int padded_w = _roi.width * padding;
		int padded_h = _roi.height * padding;


		// ���ճ�������޸ĳ����С����֤�Ƚϴ�ı�Ϊtemplate_size��С  
		if (template_size > 1) {  // Fit largest dimension to the given template size  
			if (padded_w >= padded_h)  //fit to width  
				_scale = padded_w / (float)template_size;
			else
				_scale = padded_h / (float)template_size;

			_tmpl_sz.width = padded_w / _scale;
			_tmpl_sz.height = padded_h / _scale;
		}
		else {  //No template size given, use ROI size  
			_tmpl_sz.width = padded_w;
			_tmpl_sz.height = padded_h;
			_scale = 1;
			// original code from paper:  
			/*if (sqrt(padded_w * padded_h) >= 100) {   //Normal size
			_tmpl_sz.width = padded_w;
			_tmpl_sz.height = padded_h;
			_scale = 1;
			}
			else {   //ROI is too big, track at half size
			_tmpl_sz.width = padded_w / 2;
			_tmpl_sz.height = padded_h / 2;
			_scale = 2;
			}*/
		}

		// ����_tmpl_sz�ĳ�������ȡԭ���������С2*cell_size��  
		// ���У��ϴ�߳�Ϊ104  
		if (_hogfeatures) {
			// Round to cell size and also make it even  
			_tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
			_tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
		}
		else {  //Make number of pixels even (helps with some logic involving half-dimensions)  
			_tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
			_tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
		}
	}

	// ��������С  
	extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
	extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;

	// center roi with new size  
	// ����������Ͻ�����  
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;

	// ��ȡĿ���������أ����߽��������  
	cv::Mat FeaturesMap;
	cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);

	// ���ձ�����С�߽��С  
	if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
		cv::resize(z, z, _tmpl_sz);
	}

	// HOG features  
	// ��ȡHOG������  
	if (_hogfeatures) {
		IplImage z_ipl = z;
		CvLSVMFeatureMapCaskade *map;                                   // ����ָ��  
		getFeatureMaps(&z_ipl, cell_size, &map);            // ��map���и�ֵ  
		normalizeAndTruncate(map, 0.2f);                             // ��һ��  
		PCAFeatureMaps(map);                                                    // ��HOG������ΪPCA-HOG  
		size_patch[0] = map->sizeY;
		size_patch[1] = map->sizeX;
		size_patch[2] = map->numFeatures;

		FeaturesMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug  
		FeaturesMap = FeaturesMap.t();
		freeFeatureMapObject(&map);

		// Lab features  
		 
		if (_labfeatures) {
			cv::Mat imgLab;
			cvtColor(z, imgLab, CV_BGR2Lab);
			unsigned char *input = (unsigned char*)(imgLab.data);

			// Sparse output vector  
			cv::Mat outputLab = cv::Mat(_labCentroids.rows, size_patch[0] * size_patch[1], CV_32F, float(0));

			int cntCell = 0;
			// Iterate through each cell  
			for (int cY = cell_size; cY < z.rows - cell_size; cY += cell_size) {
				for (int cX = cell_size; cX < z.cols - cell_size; cX += cell_size) {
					// Iterate through each pixel of cell (cX,cY)  
					for (int y = cY; y < cY + cell_size; ++y) {
						for (int x = cX; x < cX + cell_size; ++x) {
							// Lab components for each pixel  
							float l = (float)input[(z.cols * y + x) * 3];
							float a = (float)input[(z.cols * y + x) * 3 + 1];
							float b = (float)input[(z.cols * y + x) * 3 + 2];

							// Iterate trough each centroid  
							float minDist = FLT_MAX;
							int minIdx = 0;
							float *inputCentroid = (float*)(_labCentroids.data);
							for (int k = 0; k < _labCentroids.rows; ++k) {
								float dist = ((l - inputCentroid[3 * k]) * (l - inputCentroid[3 * k]))
									+ ((a - inputCentroid[3 * k + 1]) * (a - inputCentroid[3 * k + 1]))
									+ ((b - inputCentroid[3 * k + 2]) * (b - inputCentroid[3 * k + 2]));
								if (dist < minDist) {
									minDist = dist;
									minIdx = k;
								}
							}
							// Store result at output  
							outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ;
							//((float*) outputLab.data)[minIdx * (size_patch[0]*size_patch[1]) + cntCell] += 1.0 / cell_sizeQ;   
						}
					}
					cntCell++;
				}
			}
			// Update size_patch[2] and add features to FeaturesMap  
			size_patch[2] += _labCentroids.rows;
			FeaturesMap.push_back(outputLab);
		}
	}
	else {
		FeaturesMap = RectTools::getGrayImage(z);
		FeaturesMap -= (float) 0.5; // In Paper;  
		size_patch[0] = z.rows;
		size_patch[1] = z.cols;
		size_patch[2] = 1;
	}

	if (inithann) {
		createHanningMats();
	}
	FeaturesMap = hann.mul(FeaturesMap);
	return FeaturesMap;
}

// Initialize Hanning window. Function called only in the first frame.  
// ��ʼ��hanning����ִֻ��һ�Σ�ʹ��opencv��������  
void KCFTracker::createHanningMats()
{
	cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1], 1), CV_32F, cv::Scalar(0));
	cv::Mat hann2t = cv::Mat(cv::Size(1, size_patch[0]), CV_32F, cv::Scalar(0));

	for (int i = 0; i < hann1t.cols; i++)
		hann1t.at<float >(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
	for (int i = 0; i < hann2t.rows; i++)
		hann2t.at<float >(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

	cv::Mat hann2d = hann2t * hann1t;
	// HOG features  
	if (_hogfeatures) {
		cv::Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with cv::Mat multichannel bug  

		hann = cv::Mat(cv::Size(size_patch[0] * size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
		for (int i = 0; i < size_patch[2]; i++) {
			for (int j = 0; j<size_patch[0] * size_patch[1]; j++) {
				hann.at<float>(i, j) = hann1d.at<float>(0, j);
			}
		}
	}
	// Gray features  
	else {
		hann = hann2d;
	}
}

// Calculate sub-pixel peak for one dimension  
// ʹ�÷�ֵ��������λ��ֵ��λ�ã����ص�����Ҫ�ı��ƫ������С  
float KCFTracker::subPixelPeak(float left, float center, float right)
{
	float divisor = 2 * center - right - left;

	if (divisor == 0)
		return 0;

	return 0.5 * (right - left) / divisor;
}