#include <opencv2\opencv.hpp>
#include <iostream>
#include<opencv2/core.hpp>
using namespace cv;
using namespace std;

typedef struct winInfo_ {
	double blurVal = 0.0;
	cv::Rect winRect;
}winInfo;


static bool intensityJudgment(const cv::Mat &image) {

	cv::Mat histogram = cv::Mat::zeros(Size(16, 1), CV_32SC1);
	int rows = image.rows;
	int cols = image.cols;

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			int index = int(image.at<uchar>(r, c));
			index /= 16;
			histogram.at<int>(0, index) += 1;
		}
	}
	double minVal, maxVal;
	int    minIdx[2] = {}, maxIdx[2] = {};
	cv::minMaxIdx(histogram, &minVal, &maxVal, minIdx, maxIdx);
	histogram.release();

	if (0 < maxIdx[1] and maxIdx[1] < 15) {
		// cout << "need enhance" << endl;
		return 1;
	}
	else {
		return 0;
	}
}


static int grayHist(const cv::Mat& image, cv::Mat histogram) {

	int rows = image.rows;
	int cols = image.cols;
	int white = 0;
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			int index = int(image.at<uchar>(r, c));
			histogram.at<int>(0, index) += 1;
			if (index == 255) {
				white++;
			}
		}
	}
	return white;
}


static void matConditionsAssignment(cv::Mat &dstImg, const cv::Mat &mask, 
	const int &cdVal, const int &newVal, const int &flag) {

	for (size_t i = 0; i < dstImg.rows; i++) {
		for (size_t j = 0; j < dstImg.cols; j++) {
			int maskVal = int(mask.at<uchar>(i, j));
			if (flag == 0) {
				if (maskVal == cdVal) {
					dstImg.at<uchar>(i, j) = newVal;
				}
			}else if(flag == 1) {
				if (maskVal > cdVal) {
					dstImg.at<uchar>(i, j) = newVal;
				}
			}
			else {
				if (maskVal < cdVal) {
					dstImg.at<uchar>(i, j) = newVal;
				}
			}	
		}
	}
}


static double blurDegreeCal(cv::Mat image) {

	cv::Mat lap_image, lap_meanV, lap_varV;
	cv::Laplacian(image, lap_image, CV_32FC1);
	cv::meanStdDev(lap_image, lap_meanV, lap_varV);

	
	cv::Mat meanV, varV;
	cv::meanStdDev(image, meanV, varV);
	double score = lap_varV.at<double>(0, 0) * varV.at<double>(0, 0);

	return score;
}


static bool cmp(const winInfo& a, const winInfo& b) {
	return a.blurVal < b.blurVal;
}


static void GammaCorrection(const cv::Mat &img, cv::Mat &dst, double gammaFactor)
{
	Mat lookUpTable(1.0, 256.0, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gammaFactor) * 255.0);
	// Mat res = img.clone();
	LUT(img, lookUpTable, dst);
}


static void localBalance(cv::Mat &subImg) {

	cv::Mat subHist = cv::Mat::zeros(Size(256, 1), CV_32SC1);
	int white = grayHist(subImg, subHist);
	
	int valid_pixels_shape = subImg.rows * subImg.cols - white;
	if (valid_pixels_shape > 100) {
		float ratio = 0.5;
		int validIndex = int((1 - ratio) * valid_pixels_shape);

		int pixelIndex = 0;
		int valid_limit_pixel = 0;
		for (size_t b = 0; b < 255; b++) {
			int magiude = subHist.at<int>(0, b);
			pixelIndex += magiude;
			if (pixelIndex >= validIndex) {
				valid_limit_pixel = b;
				break;
			}
		}
		matConditionsAssignment(subImg, subImg, valid_limit_pixel, valid_limit_pixel, 1);
		cv::normalize(subImg, subImg, 0, 255, cv::NORM_MINMAX);
	}
}


static cv::Mat slideProcess(cv::Mat image) {

	cv::Size slideStep(6,8);
	cv::Size window(0, 0);
	window.width = image.cols / slideStep.width;
	window.height = image.rows / slideStep.height;
	int winW = 0;
	int winH = 0;
	vector<winInfo> allWins;
	winInfo temp;
	cv::Rect rec;
	Mat subImg;
	double blurVal = 0.0;
	for (size_t y = 0; y < image.rows; ) {
		if ((image.rows - y - window.height) < window.height)
		{
			winH = image.rows - y;
		}
		else {
			winH = window.height;
		}
		for (size_t x = 0; x < image.cols; ) {
			if ((image.cols - x - window.width) < window.width)
			{
				winW = image.cols - x;
			}
			else {
				winW = window.width;
			}
			rec = cv::Rect(x, y, winW, winH);
			subImg = image(rec);
			blurVal = blurDegreeCal(subImg);
			
			temp.blurVal = blurVal;
			temp.winRect = rec;
			allWins.push_back(temp);
			x += winW;
		}
		y += winH;
	}


	sort(allWins.begin(), allWins.end(), cmp);

	double gap = double(2.0 - 0.5) / double(slideStep.width * slideStep.height);
	cv::Rect subRect;
	cv::Mat swImg;
	cv::Mat blankImg = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	for (size_t i = 0; i < allWins.size(); i++) {
		subRect = allWins[i].winRect;
		subImg = image(subRect);
		double gammacoef = round((2 - gap * i) * 10) / 10;
		//std::cout << "gammacoef: " << gammacoef << endl;
		GammaCorrection(subImg, swImg, gammacoef);

		localBalance(swImg);

		Mat aimImg = blankImg(Range(subRect.y, subRect.y + subRect.height), Range(subRect.x, subRect.x + subRect.width));
		swImg.copyTo(aimImg);

	}

	return blankImg;
}


static cv::Mat enhanceImg(cv::Mat &oriImg) {

	cv::Mat gaussImg, cannyImg;
	cv::GaussianBlur(oriImg.clone(), gaussImg, cv::Size(3, 3), 0);
	cv::Canny(gaussImg, cannyImg, 50, 150);
	gaussImg.release();
	//Rect ccomp;
	//floodFill(cannyImg, Point(0, 0), Scalar(255, 255, 255), &ccomp, Scalar(20, 20, 20), Scalar(80, 80, 80));
	cv::Mat mm = ~cannyImg;
	cv::Mat kernelsize0 = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::erode(mm, mm, kernelsize0,Point(-1,-1),4);
	//cv::erode(mm, mm, kernelsize0);

	cv::Mat mm1;
	mm = ~mm;
	mm1 = mm.clone();
	mm.release();
	matConditionsAssignment(cannyImg, mm1, 0, 0, 0);

	cv::Mat kernelsize1 = getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	cv::morphologyEx(mm1, mm1, cv::MORPH_CLOSE, kernelsize1);
	cv:dilate(mm1, mm1, kernelsize0);
	cv::Mat textureImg;
	oriImg.copyTo(textureImg);
	matConditionsAssignment(textureImg, mm1, 0, 255, 0);
	mm1.release();

	cv::Mat textureImg1 = slideProcess(textureImg);
	textureImg.release();

	cv::Mat noshadow = textureImg1.clone();
	textureImg1.release();
	GammaCorrection(noshadow, noshadow, 2);

	cannyImg = ~cannyImg;
	cv::Mat highEdge = cannyImg.clone();
	cannyImg.release();
	cv::resize(highEdge, highEdge, cv::Size(0, 0), 0.5, 0.5, cv::INTER_AREA);
	cv::resize(highEdge, highEdge, noshadow.size(), 0, 0, cv::INTER_CUBIC);
	matConditionsAssignment(noshadow, highEdge, 0, 0, 0);

	cv::GaussianBlur(noshadow, noshadow, cv::Size(3, 3), 0);
	cv::Mat kernel0 = (Mat_<char>(3, 3) << 0, -1, 0,
										  -1, 5, -1,
										   0, -1, 0);
	Mat dstImage;
	cv::filter2D(noshadow, dstImage, -1, kernel0);

	noshadow.release();

	return dstImage;
}


cv::Mat denoseImg2_0(cv::Mat &rawImg) {
	if (rawImg.channels() > 1) {
		cv::cvtColor(rawImg, rawImg, cv::COLOR_BGR2GRAY);
	}

	cv::Mat rawImgs = rawImg.clone();
	cv::Mat display;
	if (intensityJudgment(rawImg)) {
		display = enhanceImg(rawImgs);
	}
	else {
		cv::medianBlur(rawImgs, rawImgs, 3);
		cv::bilateralFilter(rawImgs,rawImgs,7,40,40);
		cv::normalize(rawImgs, display, 0, 255, cv::NORM_MINMAX);
	}
	//threshold(display,display,150,255,THRESH_OTSU);
	return display;
}
void main4()
{
	const char* path = "D:\\11\\aa\\1_1.jpg";
	cv::Mat raw_image = imread(path, 1);
	Mat display = denoseImg2_0(raw_image);
	imwrite("D:/11/aa/000.jpg", display);
 	cout << "ok" << endl;
}