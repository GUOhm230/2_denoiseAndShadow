//#include <fmt/core.h>                    // for print
//#include <opencv2/core/hal/interface.h>  // for CV_8U, CV_32FC1
//#include <opencv2/core.hpp>              // for normalize, merge, split, dft
//#include <opencv2/highgui.hpp>           // for imshow, waitKey, namedWindow
//#include <opencv2/imgproc.hpp>           // for circle
//#include<iostream>
//using namespace std;
//using namespace cv;
//namespace db {
//    void imshow(const cv::String& winName_, const cv::Mat& img_) {
//#ifdef DEBUG
//        if (img_.depth() == CV_8U) {
//            cv::imshow(winName_, img_);
//        }
//        else {
//            double min_ = 0, max_ = 0;
//            cv::Point2i min_pt, max_pt;
//            cv::Mat temp;
//            cv::minMaxLoc(img_, &min_, &max_, &min_pt, &max_pt, cv::noArray());
//            cv::normalize(img_, temp, 0, 255, cv::NORM_MINMAX, CV_8U, {});
//
//            cv::imshow(winName_, temp);
//            if (min_ < 0 || max_ > 1) {
//                fmt::print("[DEBUG] {} is not of any showing formats, a makeshift image "
//                    "is create.\nOriginal states:\n",
//                    winName_);
//                fmt::print("minVal: {} at point ({},{})\n", min_, min_pt.x, min_pt.y);
//                fmt::print("maxVal: {} at point ({},{})\n\n", max_, max_pt.x, max_pt.y);
//            }
//        }
//#endif // DEBUG
//    }
//}; // namespace db
//
//
//
//static std::tuple<cv::Mat, cv::Mat> calcPSD(const cv::Mat& input_image);
//static cv::Mat remove_periodic_noise(cv::Mat&& complex_, const cv::Point2i& noise_spike);
//
//int main() {
//   
//    cv::Mat input_image = cv::imread("C:/Users/14302/Desktop/12.jpeg", cv::IMREAD_GRAYSCALE);
//    if (input_image.empty()) {
//        //fmt::print("Error reading image: {}", argv[1]);
//        return 1;
//    }
//    //cv::imshow("input_image", input_image);
//    cv::Rect roi(0, 0, input_image.cols & -2, input_image.rows & -2);
//    input_image = input_image(roi);
//
//    auto a = calcPSD(input_image);
//    cv:: Mat magnitude = std::get<0>(a);
//    cv::Mat complex = std::get<1>(a);
//
//    const char* winName = "magnitude";
//
//
//    cv::Point2i noise_spike;
//    //cv::namedWindow(winName, WINDOW_NORMAL);
//
//    cv::setMouseCallback(winName,
//        [](int event, int x, int y, int flags, void* pt) {
//            if (event == cv::EVENT_LBUTTONDOWN) {
//                auto spike_ptr = static_cast<cv::Point2i*>(pt);
//                spike_ptr->x = x;
//                spike_ptr->y = y;
//                cout<<x<<y<<endl;
//                //fmt::print("Selected frequency spike at ({},{}), now press any key to continue.\n", x, y);
//            }
//        },
//        &noise_spike);
//    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX, CV_8U, {});
//    //cv::imshow("magnitude", magnitude);
//    //cv::waitKey(0);
//    
//    cv::Mat recovered_image = remove_periodic_noise(std::move(complex), noise_spike);
//    cv::normalize(recovered_image, recovered_image, 0, 255, cv::NORM_MINMAX, CV_8U, {});
//    //cv::imshow("recovered image", recovered_image);
//    cv::waitKey(0);
//    return 0;
//}
//
//std::tuple<cv::Mat, cv::Mat> calcPSD(const cv::Mat& input_image) {
//    std::vector<cv::Mat> channels{ cv::Mat_<float>(input_image),
//                                  cv::Mat::zeros(input_image.size(), CV_32FC1) };
//    cv::Mat composite;
//    cv::merge(channels, composite);
//    cv::Mat complex;
//    cv::dft(composite, complex, cv::DFT_COMPLEX_OUTPUT);
//    cv::split(complex, channels);
//    cv::Mat magnitude;
//    cv::magnitude(channels[0], channels[1], magnitude);
//    cv::pow(magnitude, 2, magnitude);
//    magnitude.at<float>(0) = 0; //NOTE: very important, value at (0,0) is the main frequency of the whole image. For visualization reason, it must set to zero, otherwise it will dwarf all the noise spikes.
//    return { magnitude, complex };
//}
//
//cv::Mat remove_periodic_noise(cv::Mat&& complex_, const cv::Point2i& noise_spike) {
//    cv::Point2i c2, c3, c4;
//    c2.x = noise_spike.x;
//    c2.y = complex_.rows - noise_spike.y;
//    c3.x = complex_.cols - noise_spike.x;
//    c3.y = noise_spike.y;
//    c4.x = complex_.cols - noise_spike.x;
//    c4.y = complex_.rows - noise_spike.y;
//    cv::Mat anti_spike = cv::Mat::ones(complex_.size(), CV_32FC1);
//    for (const auto& pt : { noise_spike, c2, c3, c4 }) {
//        cv::circle(anti_spike, pt, 10, { 0 }, -1);
//    }
//    std::vector<cv::Mat> channels{
//        std::move(anti_spike),
//        cv::Mat::zeros(complex_.size(), CV_32FC1) };
//    cv::Mat anti_spike_composite;
//    cv::merge(channels, anti_spike_composite);
//    cv::mulSpectrums(complex_, anti_spike_composite, complex_, cv::DFT_ROWS);
//    cv::idft(complex_, complex_);
//    cv::split(complex_, channels);
//    return channels[0];
//}
/*eclipse cdt, gcc 4.8.1*/

//#include <iostream>  
//#include <vector>  
//#include <string>  
//#include <tuple>  
//
//using namespace std;
//
//std::tuple<std::string, int>
//giveName(void)
//{
//    std::string cw("Caroline");
//    int a(2013);
//    std::tuple<std::string, int> t = std::make_tuple(cw, a);
//    return t;
//}
//
//int main()
//{
//    std::tuple<int, double, std::string> t(64, 128.0, "Caroline");
//    std::tuple<std::string, std::string, int> t2 =
//        std::make_tuple("Caroline", "Wendy", 1992);
//
//    //返回元素个数  
//    size_t num = std::tuple_size<decltype(t)>::value;
//    std::cout << "num = " << num << std::endl;
//
//    //获取第1个值的元素类型  
//    std::tuple_element<1, decltype(t)>::type cnt = std::get<1>(t);
//    std::cout << "cnt = " << cnt << std::endl;
//
//    //比较  
//    std::tuple<int, int> ti(24, 48);
//    std::tuple<double, double> td(28.0, 56.0);
//    bool b = (ti < td);
//    std::cout << "b = " << b << std::endl;
//
//    //tuple作为返回值  
//    auto a = giveName();
//    std::cout << "name: " << get<0>(a)
//        << " years: " << get<1>(a) << std::endl;
//
//    return 0;
//}
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

static void help(char* progName)
{
        cout << endl<< "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
        << "The dft of an image is taken and it's power spectrum is displayed." << endl
        << "Usage:" << endl
        << progName << " [image_name -- default ../data/lena.jpg] " << endl << endl;
        
}

int main3()
{

    const char* filename = "C:/Users/14302/Desktop/12.jpeg";
    Mat input = imread(filename, IMREAD_GRAYSCALE);

    int w = getOptimalDFTSize(input.cols);
    int h = getOptimalDFTSize(input.rows);//获取最佳尺寸，快速傅立叶变换要求尺寸为2的n次方
    Mat padded;     //将输入图像延扩到最佳的尺寸  在边缘添加0
    copyMakeBorder(input, padded, 0, h - input.rows, 0, w - input.cols, BORDER_CONSTANT, Scalar::all(0));//填充图像保存到padded中
    Mat plane[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F) };//创建通道
    Mat complexIm;
    merge(plane, 2, complexIm);//为延扩后的图像增添一个初始化为0的通道
    dft(complexIm, complexIm);//进行傅立叶变换，结果保存在自身
    split(complexIm, plane);//分离通道
    magnitude(plane[0], plane[1], plane[0]);//获取幅度图像，0通道为实数通道，1为虚数，因为二维傅立叶变换结果是复数
    plane[0] = plane[0](Rect(0, 0, plane[0].cols & -2, plane[0].rows & -2));
    int cx = padded.cols / 2; int cy = padded.rows / 2;//一下的操作是移动图像，左上与右下交换位置，右上与左下交换位置
    Mat temp;
    Mat part1(plane[0], Rect(0, 0, cx, cy));
    Mat part2(plane[0], Rect(cx, 0, cx, cy));
    Mat part3(plane[0], Rect(0, cy, cx, cy));
    Mat part4(plane[0], Rect(cx, cy, cx, cy));


    part1.copyTo(temp);
    part4.copyTo(part1);
    temp.copyTo(part4);

    part2.copyTo(temp);
    part3.copyTo(part2);
    temp.copyTo(part3);
    //*******************************************************************

    Mat _complexim;
    complexIm.copyTo(_complexim);//把变换结果复制一份，进行逆变换，也就是恢复原图
    Mat iDft[] = { Mat::zeros(plane[0].size(),CV_32F),Mat::zeros(plane[0].size(),CV_32F) };//创建两个通道，类型为float，大小为填充后的尺寸
    idft(_complexim, _complexim);//傅立叶逆变换
    split(_complexim, iDft);//结果貌似也是复数
    magnitude(iDft[0], iDft[1], iDft[0]);//分离通道，主要获取0通道
//    normalize(iDft[0],iDft[0],1,0,CV_MINMAX);//归一化处理，float类型的显示范围为0-1,大于1为白色，小于0为黑色
    normalize(iDft[0], iDft[0], 0, 1, NORM_MINMAX);
    //imshow("idft", iDft[0]);//显示逆变换
//*******************************************************************
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    plane[0] += Scalar::all(1);//傅立叶变换后的图片不好分析，进行对数处理，结果比较好看
    log(plane[0], plane[0]);
    normalize(plane[0], plane[0], 0, 1, NORM_MINMAX);

    //imshow("dft", plane[0]);
    waitKey();
    return 0;
}
