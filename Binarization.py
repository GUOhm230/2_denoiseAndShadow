import cv2 as cv

# 图像二值化  全局阈值化方法
def threshold(self):
    src = cv.imread(self)
    if src is None:
        return

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # 这个函数的第一个参数就是原图像，原图像应该是灰度图。
    # 第二个参数就是用来对像素值进行分类的阈值。
    # 第三个参数就是当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
    # 第四个参数来决定阈值方法，见threshold_simple()
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    ret, dst = cv.threshold(gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.namedWindow("二值化", cv.WINDOW_NORMAL)
    cv.imshow("二值化", dst)
    cv.waitKey(0)


# 图像二值化之局部阈值化方法
def jubuThreshold(input_img_file):
    image = cv.imread(input_img_file)
    cv.imshow("image", image)  # 显示二值化图像
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 8)  # 自适应阈值算法取平均值：THRESH_BINARY；高斯法：  ADAPTIVE_THRESH_GAUSSIAN_C
    cv.imshow("binary", binary)
    cv.waitKey(0)
    cv.destroyAllWindows()

import numpy as np
imgPath = r"E:\workHandover\2_denoiseAndShadow\data\20220424105436.jpg"
# jubuThreshold(imgPath)


kernel = np.ones((3, 3), np.uint8)
original_img = cv.imread(imgPath)
# 计算灰白色部分像素的均值-----------------------------去阴影操作 后续可调节参数
pixel = int(np.mean(original_img[original_img > 140]))
# 把灰白色部分修改为与背景接近的颜色
original_img[original_img > 140] = pixel
#----------------------------------------------------------------------------
gray_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)  # 灰度处理

cv.imshow('Gray_img', gray_img)
# 二值化阈值处理(CV2.THRESH_BINARY)
# 反二值化阈值处理(CV2.THRESH_BINARY_INV)
# 截断阈值化处理(CV2.THRESH_TRUNC)
# 超阈值零处理(CV2.THRESH_TOZERO_INV)
# 低阈值零处理(CV2.THRESH_TOZERO)
ret, th1 = cv.threshold(gray_img, 140, 255, cv.THRESH_BINARY)
cv.imshow('th1', th1)

# 开运算
# erosion = cv2.erode(th1, kernel, iterations=1)  # 腐蚀
# dilation = cv2.dilate(erosion, kernel, iterations=1)  #膨胀

# 闭运算
dilation = cv.dilate(th1, kernel, iterations=3)  #膨胀
erosion = cv.erode(dilation, kernel, iterations=1)  # 腐蚀

cv.imshow('erosion', th1)
cv.imshow('dilation', dilation)
cv.imshow('image', pixel)
cv.waitKey(0)

