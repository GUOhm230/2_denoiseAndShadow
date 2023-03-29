////#include <opencv2\opencv.hpp>
////#include <iostream>
////#include<opencv2/core.hpp>
////#include<opencv2/imgproc.hpp>
////#include<string>
////using namespace cv;
////using namespace std;
////
////
////
////void main()
////{
////	char a[5] = "Type";
////
////	//char a[123] = { 'Z', 's', 'p', 'l', 'j', 'r', 'q', 'v', 'n', 'm', 'C', 'F', 'D', 'B', 'A', '2', '0', 'Z', 'a' };
////	for (int i = 0; i <strlen(a); i++) {//字母排序
////		for (int j = i + 1; j < strlen(a); j++) {
////			if (tolower(a[j]) < tolower(a[i])) {
////				char pTem = a[j];
////				a[j] = a[i];
////				a[i] = pTem; 
////			}
////		}
////	}
////	
////	cout<<a<<endl;
////
////}
//#include<algorithm>
//#include<math.h>
//#include<iostream>
//#include <string.h>
//#include <stdio.h>
//#include<vector>
//using namespace std;
//void char_split() 
//{
//
//	char string[] = "A string\tof ,,tokens\nand some  more tokens";
//	char seps[] = " ,\t\n";
//	char* token = NULL;
//	printf("Tokens:\n");
//	char* ptr = NULL;
//	token = strtok_s(string, seps, &ptr);//相较于strtok()函数，strtok_s函数需要用户传入一个指针，用于函数内部判断从哪里开始处理字符串
//	while (token != NULL) {
//		printf("%s\n", token);
//		token = strtok_s(NULL, seps, &ptr);//其他的使用与strtok()函数相同
//	}
//
//}
//
//typedef struct Bbox {
//    int x;
//    int y;
//    int w;
//    int h;
//    float score;
//}Bbox;
//
//class Solution {
//public:
//
//    static bool sort_score(Bbox box1, Bbox box2) {
//        return box1.score > box2.score ? true : false;
//    }
//	static bool sort_score(Bbox box1, Bbox box2) {
//	
//		return box1.score > box2.score ? true : false;
//	}
//
//    float iou(Bbox box1, Bbox box2) {
//        int x1 = max(box1.x, box2.x);
//        int y1 = max(box1.y, box2.y);
//        int x2 = min(box1.x + box1.w, box2.x + box2.w);
//        int y2 = min(box1.y + box1.h, box2.y + box2.h);
//        int w = max(0, x2 - x1 + 1);
//        int h = max(0, y2 - y1 + 1);
//        float over_area = w * h;
//        return over_area / (box1.w * box1.h + box2.w * box2.h - over_area);
//    }
//
//    vector<Bbox> nms(std::vector<Bbox>& vec_boxs, float threshold) 
//	{
//        vector<Bbox>results;
//        std::sort(vec_boxs.begin(), vec_boxs.end(), sort_score);
//        while (vec_boxs.size() > 0)
//        {
//            results.push_back(vec_boxs[0]);
//            int index = 1;
//            while (index < vec_boxs.size()) {
//                float iou_value = iou(vec_boxs[0], vec_boxs[index]);
//                cout << "iou:" << iou_value << endl;
//                if (iou_value > threshold)
//                    vec_boxs.erase(vec_boxs.begin() + index);
//                else
//                    index++;
//            }
//            vec_boxs.erase(vec_boxs.begin());
//        }
//        return results;
//    }
//
//	vector<Bbox> nms(std::vector<Bbox>& vec_boxs, float threshold) 
//	{
//		vector<Bbox> results;
//		std::sort(vec_boxs.begin(), vec_boxs.end(), sort_score);
//		while (vec_boxs.size()>0)
//		{
//			results.push_back(vec_boxs[0]);
//			int index = 1;
//			while (index < vec_boxs.size())
//			{
//				float iou_value = iou(vec_boxs[0],vec_boxs[index]);
//				if (iou_value > threshold)
//					vec_boxs.erase(vec_boxs.begin() + index);
//				else
//				{
//					index++;
//				}
//			}
//			vec_boxs.erase(vec_boxs.begin());
//		}
//		return results;
//	}
//
//
//};
//typedef struct {
//	Rect box;
//	float confidence;
//	int index;
//}BBOX;
//
//static float get_iou_value(Rect rect1, Rect rect2)
//{
//	int xx1, yy1, xx2, yy2;
//
//	xx1 = max(rect1.x, rect2.x);
//	yy1 = max(rect1.y, rect2.y);
//	xx2 = min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
//	yy2 = min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);
//
//	int insection_width, insection_height;
//	insection_width = max(0, xx2 - xx1 + 1);
//	insection_height = max(0, yy2 - yy1 + 1);
//
//	float insection_area, union_area, iou;
//	insection_area = float(insection_width) * insection_height;
//	union_area = float(rect1.width * rect1.height + rect2.width * rect2.height - insection_area);
//	iou = insection_area / union_area;
//	return iou;
//}
//
////input:  boxes: 原始检测框集合;
////input:  confidences：原始检测框对应的置信度值集合
////input:  confThreshold 和 nmsThreshold 分别是 检测框置信度阈值以及做nms时的阈值
////output:  indices  经过上面两个阈值过滤后剩下的检测框的index
//void nms_boxes(vector<Rect>& boxes, vector<float>& confidences, float confThreshold, float nmsThreshold, vector<int>& indices)
//{
//	BBOX bbox;
//	vector<BBOX> bboxes;
//	int i, j;
//	for (i = 0; i < boxes.size(); i++)
//	{
//		bbox.box = boxes[i];
//		bbox.confidence = confidences[i];
//		bbox.index = i;
//		bboxes.push_back(bbox);
//	}
//	sort(bboxes.begin(), bboxes.end(), comp);
//
//	int updated_size = bboxes.size();
//	for (i = 0; i < updated_size; i++)
//	{
//		if (bboxes[i].confidence < confThreshold)
//			continue;
//		indices.push_back(bboxes[i].index);
//		for (j = i + 1; j < updated_size; j++)
//		{
//			float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
//			if (iou > nmsThreshold)
//			{
//				bboxes.erase(bboxes.begin() + j);
//				updated_size = bboxes.size();
//			}
//		}
//	}
//}
//
//
//int main() 
//{
//	char str[5] = "BabA";
//
//	
//	for (int i = 0; i < strlen(str); i++)
//	{
//		for (int j = i + 1; j < strlen(str); j++)
//		{
//			if (tolower(str[j]) <= tolower(str[i]))
//			{
//				char temp = str[j];
//				str[j] = str[i];
//				str[i] = temp;
//
//			}
//			
//		}
//	}
//	cout<<str<<endl;
//
//	return 0;
//}
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include<cmath>
#include <ctype.h>
using namespace  std;
using namespace  cv;

Mat Kernel_test_3_3 = (
	Mat_<double>(3, 3) <<
	0, -1, 0,
	-1, 5, -1,
	0, -1, 0);
void Convlution(Mat  InputImage, Mat  OutputImage, Mat kernel)
{
	
	//计算卷积核的半径
	int sub_x = kernel.cols / 2;
	int sub_y = kernel.rows / 2;
	//遍历图片  
	for (int image_y = 0; image_y < InputImage.rows - 2 * sub_y; image_y++)
	{
		for (int image_x = 0; image_x < InputImage.cols - 2 * sub_x; image_x++)
		{
			int pix_value = 0;
			for (int kernel_y = 0; kernel_y < kernel.rows; kernel_y++)
			{
				for (int kernel_x = 0; kernel_x < kernel.cols; kernel_x++)
				{
					double  weihgt = kernel.at<double>(kernel_y, kernel_x);
					int value = (int)InputImage.at<uchar>(image_y + kernel_y, image_x + kernel_x);
					pix_value += weihgt * value;
				}
			}
			OutputImage.at<uchar>(image_y + sub_y, image_x + sub_x) = (uchar)pix_value;
		}
	}
}

vector<int> quickSort(int left, int right, vector<int> & arr)
{
	if (left >= right)
		return arr;
	int i, j, base, temp;
	i = left, j = right;
	base = arr[left];  //取最左边的数为基准数
	while (i < j)
	{
		while (arr[j] >= base && i < j)
			j--;
		while (arr[i] <= base && i < j)
			i++;
		if (i < j)
		{
			temp = arr[i];
			arr[i] = arr[j];
			arr[j] = temp;
		}
	}
	//基准数归位
	arr[left] = arr[i];
	arr[i] = base;
	quickSort(left, i - 1, arr);//递归左边
	quickSort(i + 1, right, arr);//递归右边
	return arr;
}

int main()
{
	/*vector<int> arr = { 6,1,2,7,5,9,3,4,5,10,8 };

	vector<int> arr1 =  quickSort(0, 10, arr);*/
	//fun2();
	string a = "ty?a";
	string b=a;
	
	vector<int> index;
	//str.erase(remove(str.begin(), str.end(), '?'), str.end());
	for (int i = 0; i < a.size(); i++)
	{
		if (!isalpha(a[i]))
		{
			index.push_back(i);
			a.erase(i,1);
		}
		
	}
	for (int i = 0; i < a.size(); i++)
	{
		for (int j = i + 1; j < a.size(); j++)
		{
			if (tolower(a[j]) <= tolower(a[i]))
			{
				char temp = a[j];
				a[j] = a[i];
				a[i] = temp;
			}

		}
	}
	for(auto lin : index)
	{
		char aa = b[lin];
		string aaa(1, aa);
		a.insert(lin,aaa);
	}
	waitKey(0);
	return 0;
	//Mat srcImage = imread("C:\\Users\\14302\\Desktop\\new_img10.png", 0);
	//namedWindow("srcImage", WINDOW_AUTOSIZE);
	////imshow("原图", srcImage);

	////filter2D卷积
	//Mat dstImage_oprncv(srcImage.rows, srcImage.cols, CV_8UC1, Scalar(0));;
	//filter2D(srcImage, dstImage_oprncv, srcImage.depth(), Kernel_test_3_3);
	////imshow("filter2D卷积图", dstImage_oprncv);
	////imwrite("1.jpg", dstImage_oprncv);

	////自定义卷积
	//Mat dstImage_mycov(srcImage.rows, srcImage.cols, CV_8UC1, Scalar(0));
	//Convlution(srcImage, dstImage_mycov, Kernel_test_3_3);
	//imshow("卷积图3", dstImage_mycov);
	//imwrite("2.jpg", dstImage_mycov);

}
//for (vector<int>::iterator iter = v.begin(); iter != v.end(); iter++)
//*iter = 0;