#include<opencv2/opencv.hpp>
#include<iostream>
#include <vector>
#include <algorithm>
using namespace cv;
using namespace std;
double getDistance(Point2f point1, Point2f point2)
{
	double distance = sqrtf(powf((point1.x - point2.x), 2) + powf((point1.y - point2.y), 2));

	return distance;
}

Mat correct(Mat &raw_img,vector<Point2f> &corners)
{
	Mat gray;
	if (raw_img.channels() == 3)
	{

		cvtColor(raw_img, gray, COLOR_BGR2GRAY);
	}
	else
	{
		gray = raw_img;
	}
	float h_rows = raw_img.rows;
	float w_cols = raw_img.cols;
	float min_x = w_cols, min_y = h_rows, max_x = 0, max_y = 0;


	for (int i = 0; i < corners.size(); ++i)
	{
		min_x = (corners[i].x > min_x) ? min_x : corners[i].x;
		min_y = (corners[i].y > min_y) ? min_y : corners[i].y;
		max_x = (corners[i].x < max_x) ? max_x : corners[i].x;
		max_y = (corners[i].y < max_y) ? max_y : corners[i].y;
	}
	Point2d lt(0, 0);
	int temp_x = int(max_x - min_x);
	int temp_y = int(max_y - min_y);
	Point2d rt(temp_x, 0);
	Point2d rd(temp_x, temp_y);
	Point2d ld(0, temp_y);
	vector<Point2d> aimp;

	aimp.push_back(lt);
	aimp.push_back(rt);
	aimp.push_back(rd);
	aimp.push_back(ld);

	vector<Point2f> cc(4);
	cc.swap(corners);

	vector<Point2f> pts1;
	vector<Point2f> pts2;
	vector<Point2f> new_cornes;
	for (auto p : aimp)
	{
		pts2.push_back(p);
		vector<float> td;
		for (auto c : cc)
		{
			Point2d dd(p.x + min_x, p.y + min_y);
			float temp = getDistance(dd, c);
			td.push_back(temp);
		}
		auto cid = min_element(td.begin(), td.end());
		int index = distance(begin(td), cid);
		pts1.push_back((cc[index]));
	}
	

	cv::Mat result_images(pts2[2].y, pts2[2].x, CV_32FC3, cv::Scalar(0, 0,0));  //创建一副图像
	
	Mat warpmatrix = getPerspectiveTransform(pts1, pts2); 
	warpPerspective(gray, result_images, warpmatrix,result_images.size());
		
	return result_images;
}

Mat morphology() 
{
	Mat src = imread("D:/11/aa/1.jpg");
	imshow("原图", src);
	//1.将图像转为灰度图
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	//定义腐蚀和膨胀的结构化元素和迭代次数
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	int iteration = 7;
	//2.将灰度图进行闭运算操作
	Mat closeMat;
	morphologyEx(gray, closeMat, MORPH_CLOSE, element, Point(-1, -1), iteration);
	//imshow("闭运算", closeMat);
	//4.闭运算后的图减去原灰度图再进行取反操作
	Mat calcMat = ~(closeMat - gray);
	//imshow("calc", calcMat);
	//5.使用规一化将原来背景白色的改了和原来灰度图差不多的灰色
	Mat removeShadowMat;
	normalize(calcMat, removeShadowMat, 0, 200, NORM_MINMAX);
	cout<<"ok"<<endl;
	return removeShadowMat;
	//imshow("dst", removeShadowMat);
}

void main2() 
{
	const char* img_path = "D:\\11\\aa\\0.jpg";
	cv::Mat oriImg = imread(img_path, 1);
	Mat seg = imread("C:/Users/14302/Desktop/3.jpeg", 0);
	Mat mask = imread("C:/Users/14302/Desktop/1.jpeg", 0);
	
	Mat aas = morphology();
	vector<Point2f> corners;
	Point2f a(378.141449, 236.479706), b(2957.973633, 160.266327), c(233.114258 ,3991.955322), d(3432.748047, 3745.715332);

	corners.push_back(a);
	corners.push_back(b);
	corners.push_back(c);
	corners.push_back(d);
	//Mat dst = correct(oriImg,corners);
	//imwrite("D:/11/aa/00.jpg",dst);
}
//#include <opencv2\opencv.hpp>
//#include <iostream>
//using namespace cv;
//using namespace std;
//static void matConditionsAssignment(cv::Mat& dstImg, const cv::Mat& mask,
//	const int& cdVal, const int& newVal, const int& flag) {
//
//	for (size_t i = 0; i < dstImg.rows; i++) {
//		for (size_t j = 0; j < dstImg.cols; j++) {
//			int maskVal = int(mask.at<uchar>(i, j));
//			if (flag == 0) {
//				if (maskVal == cdVal) {
//					dstImg.at<int>(i, j) = newVal;
//				}
//			}
//			else if (flag == 1) {
//				if (maskVal > cdVal) {
//					dstImg.at<int>(i, j) = newVal;
//				}
//			}
//			else {
//				if (maskVal < cdVal) {
//					dstImg.at<int>(i, j) = newVal;
//				}
//			}
//
//		}
//	}
//}
//int main1()
//{
//	system("color F0");  //��DOS������ɰ׵׺���
//
//	const char* img_path = "D:\\11\\aa\\0004.png";
//	cv::Mat img = imread(img_path, 1);
//	if (!(img.data))
//	{
//		cout << "��ȡͼ�������ȷ��ͼ���ļ��Ƿ���ȷ" << endl;
//		return -1;
//	}
//
//	RNG rng(10086);//�����
//
//	//���ò�����־flags
//	int connectivity = 4;  //��ͨ����ʽ
//	int maskVal = 255;  //����ͼ�����ֵ
//	int flags = connectivity | (maskVal << 8) | FLOODFILL_FIXED_RANGE;  //��ˮ��������ʽ��־ 
//
//		//������ѡ�����ص�Ĳ�ֵ
//	Scalar loDiff = Scalar(20, 20, 20);
//	Scalar upDiff = Scalar(20, 20, 20);
//
//	//������ģ�������
//
//
//	Mat mask = Mat::zeros(img.rows + 2, img.cols + 2, CV_8UC1);
//
//	while (true)
//	{
//		//�������ͼ����ĳһ���ص�
//		int py = rng.uniform(0, img.rows - 1);
//		int px = rng.uniform(0, img.cols - 1);
//		Point point = Point(px, py);
//
//		//��ɫͼ������������ֵ
//		Scalar newVal = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
//			rng.uniform(0, 255));
//
//		//��ˮ��亯��
//		int area = floodFill(img, mask, point, newVal, &Rect(), loDiff, upDiff, flags);
//
//		//������ص������������Ŀ
//		//cout << "���ص�x��" << point.x << "  y:" << point.y
//		//	<< "     ���������Ŀ��" << area << endl;
//
//		////�������ͼ����
//		//imshow("���Ĳ�ɫͼ��", img);
//		//imshow("��ģͼ��", mask);
//
//		//�ж��Ƿ��������
//		//int c = waitKey(0);
//		/*if ((c & 255) == 27)
//		{
//			break;
//		}*/
// 	}
//	return 0;
//}