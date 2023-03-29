#include <opencv2\opencv.hpp>
#include <iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
using namespace cv;
using namespace std;


static int intensityJudgment(const cv::Mat& image) {

	cv::Mat histogram = cv::Mat::zeros(Size(200, 1), CV_32SC1);
	int rows = image.rows;
	int cols = image.cols;

	int blurAmplitude = 0;
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			int index = int(image.at<uchar>(r, c));
			index /= 200;
			histogram.at<int>(0, index) += 1;
			if (index >= 10 && index <= 190) {
				blurAmplitude += 1;
			}
		}
	}

	float blurRation = float(blurAmplitude) / float(rows * cols);

	if (blurRation > 0.2) {
		cout << "need enhance" << endl;
		return 1;
	}
	else {
		cout << "need not enhance" << endl;
		return 0;
	}
}


static cv::Mat _slideProcess(cv::Mat image) {

	cv::Mat kernelsize2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat blackHatImg;
	cv::morphologyEx(image, blackHatImg, cv::MORPH_BLACKHAT, kernelsize2, cv::Point(-1, -1), 20);
	blackHatImg = ~blackHatImg;

	int winW = 40;
	int winH = 40;
	cv::Size window(40, 40);
	cv::Mat subImg;
	cv::Rect rec;
	cv::Mat blankImg = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	for (size_t y = 0; y < blackHatImg.rows; ) {
		if ((blackHatImg.rows - y - window.height) < window.height)
		{
			winH = blackHatImg.rows - y;
		}
		else {
			winH = window.height;
		}
		for (size_t x = 0; x < blackHatImg.cols; ) {
			if ((blackHatImg.cols - x - window.width) < window.width)
			{
				winW = blackHatImg.cols - x;
			}
			else {
				winW = window.width;
			}
			//std::cout << "x: " << x << " y: " << y << " winW: " << winW << " winH: " << winH << endl;
			rec = cv::Rect(x, y, winW, winH);
			subImg = blackHatImg(rec);
			cv::normalize(subImg, subImg, 0, 255, cv::NORM_MINMAX);

			Mat aimImg = blankImg(Range(y, y + winH), Range(x, x + winW));
			subImg.copyTo(aimImg);

			x += winW;
		}
		y += winH;
	}
	return blankImg;
}


static void matConditionsAssignment(cv::Mat& dstImg, const cv::Mat& mask,
	const int& cdVal, const int& newVal, const int& flag) {

	for (size_t i = 0; i < dstImg.rows; i++) {
		for (size_t j = 0; j < dstImg.cols; j++) {
			int maskVal = int(mask.at<uchar>(i, j));
			if (flag == 0) {
				if (maskVal == cdVal) {
					dstImg.at<uchar>(i, j) = newVal;
				}
			}
			else if (flag == 1) {
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


static void GammaCorrection(const cv::Mat& img, cv::Mat& dst, double gammaFactor)
{
	Mat lookUpTable(1.0, 256.0, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gammaFactor) * 255.0);
	//Mat res = img.clone();
	LUT(img, lookUpTable, dst);
}


static cv::Mat enhanceImg(cv::Mat& oriImg) {

	// step0
	cv::Mat gaussImg, cannyImg;
	cv::GaussianBlur(oriImg.clone(), gaussImg, cv::Size(3, 3), 0);
	cv::Canny(gaussImg, cannyImg, 50, 150);
	gaussImg.release();

	// step1
	cv::Mat edgeM = ~cannyImg;
	cv::Mat kernelsize0 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::erode(edgeM, edgeM, kernelsize0, cv::Point(-1, -1), 3);

	cv::Mat edgeM1;
	edgeM1 = ~edgeM;
	edgeM.release();
	Mat edgeM2 = edgeM1.clone();
	Mat canny1 = cannyImg.clone();
	matConditionsAssignment(cannyImg, edgeM1, 0, 0, 0);

	// step2
	cv::Mat kernelsize1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	cv::morphologyEx(edgeM1, edgeM1, cv::MORPH_CLOSE, kernelsize1);
	cv::erode(edgeM1, edgeM1, kernelsize0, cv::Point(-1, -1), 2);

	cv::Mat noshadow = _slideProcess(oriImg.clone());
	matConditionsAssignment(noshadow, edgeM1, 0, 255, 0);

	GammaCorrection(noshadow, noshadow, 2);

	// step3
	cannyImg = ~cannyImg;
	cv::Mat highEdge = cannyImg.clone();
	cannyImg.release();
	cv::resize(highEdge, highEdge, cv::Size(0, 0), 0.5, 0.5, cv::INTER_AREA);
	cv::resize(highEdge, highEdge, noshadow.size(), 0, 0, cv::INTER_CUBIC);
	matConditionsAssignment(noshadow, highEdge, 0, 0, 0);

	// step4
	Mat noshadow1;
	cv::GaussianBlur(noshadow, noshadow1, cv::Size(3, 3), 0);
	cv::Mat kernel0 = (Mat_<char>(3, 3) << 0, -1, 0,
		-1, 5, -1,
		0, -1, 0);
	Mat sharpenImg;
	cv::filter2D(noshadow1, sharpenImg, -1, kernel0);
	Mat div;
	cv::divide(noshadow, noshadow1, div, 50, -1);
	noshadow.release();

	int valid_max_pixel = 100;
	int valid_max_pixel_255 = 255;

	matConditionsAssignment(sharpenImg, sharpenImg, valid_max_pixel, valid_max_pixel_255, 1);
	cv::normalize(sharpenImg, sharpenImg, 0, 255, cv::NORM_MINMAX);

	return sharpenImg;
}


cv::Mat denoseImg2_1(cv::Mat& rawImg) {

	if (rawImg.channels() > 1) {
		cv::cvtColor(rawImg, rawImg, cv::COLOR_BGR2GRAY);
	}

	cv::Mat display = enhanceImg(rawImg);

	return display;
}
void main11()
{	
	vector<String> filename;
	string a = "D:/11/aa";
	glob(a, filename);
	const char* path = "D:/11/aa/piano/1640591382738.jpg";
	cv::Mat raw_image = imread(path, 1);
	//Mat raw_image1;
	//cv::GaussianBlur(raw_image, raw_image1, cv::Size(105, 105), 0);
	//
	//Mat div,diff;
	//cv::divide(raw_image, raw_image1, div, 250, -1);

	////cv::absdiff(raw_image,raw_image1,diff);
	//Mat can;
	//cv::Canny(div, can, 50, 150, 3);
	//cv::cvtColor(raw_image,raw_image, cv::COLOR_RGB2GRAY);
	//Mat dst =;
	Mat display = denoseImg2_1(raw_image);
	imwrite("D:/11/aa/000.jpg", display);
	cout << "ok" << endl;
}