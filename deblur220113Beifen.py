# -*- coding: utf-8 -*-
# @Author : Alvin
# @File : ShadowsRemoval_V2_1.py
# @CreateTime : 2022.01.13
# @Update: 1. 优化图像模糊度判断
#          2. 优化滑窗过程
#          3. 优化插图缺失情况
#          4. 优化字符残影

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from PIL import Image

# from edgeExtract import normalEdge


class ShadowsRemoval:

    def __init__(self, rawImg, dispersionRadius=0.5, debugMode=0) -> None:  # 无需返回值。其中表示定义的返回值类型。
        self.rawImg = rawImg
        self.patchSize = 40
        self.DispersionRadius = dispersionRadius
        self.debugMode = debugMode

        self.process = None
        self.title = None
        self.needEnhance = self.intensityJudgment()

    def intensityJudgment(self):
        """
        图像模糊度判断
        """
        # 统计亮度
        # print(self.rawImg.shape)
        imgData = copy.deepcopy(self.rawImg).ravel()  # 二维数组拉成一维ndarray数组
        # print(type(imgData))
        imgData.sort()

        n, bins, patches = plt.hist(imgData, bins=200, range=(0, 255))  # 直方图,n的长度为200,指的就是当前分组下的数据个数（normed为默认值）
        # bins表示分组的情况 每个分组的范围上下界，是一个长度为201的数组。其中宽度=255/200=1.275。按照bins的大小进行平均分分组宽度
        # patches表示分组内的具体数值
        # cv2.calcHist()也是直方图处理。直方图均衡化，是增强图像对比度的一个方法---自适应直方图均衡化
        # print(bins)
        # print(patches)
        # plt.show()
        # 统计模糊幅度：实则统计12.75:242.5之间像素点个数所占百分比就是不是全黑或者全白部分
        blurAmplitude = np.sum(n[10:190])  #
        # print(blurAmplitude)
        blurRation = blurAmplitude/imgData.shape[0] # 10:190之间总共多少个像素点。统计在此之间的像素点个数
        if blurRation > 0.2:
            return 1
        else:
            return 0

    def enhanceImg(self):

        self.process = self.rawImg
        self.title = np.ones((100, self.rawImg.shape[1]), dtype=np.uint8)*255
        cv2.putText(self.title, 'oriImg', (int(self.rawImg.shape[1]*0.45), 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 5)
        self.slideStep = self.__slidSizeCal()

        edgeImg = self.__step0()
        edgeMask = self.__step1(edgeImg)
        edgeImg[edgeMask==0] = 0
        # cv2.imwrite("edgeImg_s1_s2.jpg",edgeImg)
        noshadow = self.__step2(edgeMask)
        noshadow = self.__step3(edgeImg, noshadow)
        sharpenImg = self.__step4(noshadow)
        # finalImg = self.__step5(sharpenImg)
        if self.debugMode:
            self.process = np.concatenate((self.title, self.process), axis=0)

        return sharpenImg

    def __slidSizeCal(self):
        return (self.rawImg.shape[1]//self.patchSize, self.rawImg.shape[0]//self.patchSize)

    def __step0(self):
        """
        使用canny算子提取边缘(强特征高频信息？)。实际上后面结果并没用。但是不用则会导致一些阴影部分无法检测
        1.高斯滤波去除噪声
        2.图像梯度计算：sobel算子计算梯度和方向
        3.极大值抑制：对整幅图进行扫描，去除非边界上的点。对每个像素进行检查。得到细边的二进制图像
        4.滞后阈值：确定真正的边界。设置两个阈值：maxval:检测图像中明显的边缘和minval：将间断的边缘连接起来
        """
        # if self.needEnhance:
        #     # 处理摩尔纹
        #     cannyImg = edgeCal(copy.deepcopy(self.rawImg))
        # else:
        # cannyImg = normalEdge(copy.deepcopy(self.rawImg))
        blurImg = cv2.medianBlur(copy.deepcopy(self.rawImg), 55)  # 中值滤波,获得白色前景,去除椒盐噪声
        # cv2.imwrite("medianBlur__step0.jpg", blurImg)
        cannyImg = cv2.divide(blurImg, copy.deepcopy(self.rawImg), scale=200)  # 调节图像的明亮度。两个图像相除
        # cv2.imwrite("divide__step0.jpg", cannyImg)
        cannyImg = cv2.normalize(cannyImg, None, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite("normalize__step0.jpg", cannyImg)
        cannyImg[cannyImg != 255] = 0
        gaussImg = cv2.GaussianBlur(copy.deepcopy(self.rawImg), (3, 3), 0)  # 高斯模糊处理，在做边缘检测之前可以用于去噪
        # cv2.imwrite("gaussImg.jpg", gaussImg)
        cannyImg = cv2.Canny(gaussImg, 50, 150)
        if self.debugMode:
            # 字符边缘图
            self.__debugImg(cannyImg, "cannyImg")
        # cv2.imwrite("step0R.jpg", cannyImg)
        return cannyImg

    def __step1(self, edgeImg):
        """
        提取字符mask:选定一个区域，对图片进行遮挡，以控制图像处理的区域或处理过程
        """
        # 1.0 取出梯度mask（通过两次腐蚀扩大字符mask）
        edgeM = ~edgeImg
        kernelsize1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 返回指定形状和尺寸的结构元素，其实也就是滤波器
        # print("腐蚀所用的结构类型数据：\n", kernelsize1)
        # print("腐蚀所用的结构类型数据：", kernelsize1.shape)
        # 第一个参数表示内核形状矩形：MORPH_RECT;交叉形：MORPH_CROSS;椭圆形：MORPH_ELLIPSE;第二个表示内核尺寸以及锚点位置（默认为（-1，-1）表示中心点）
        edgeM = cv2.erode(edgeM, kernelsize1, iterations=4)  # 腐蚀操作.kernelsize1:表示腐蚀操作时所才用的结构类型。腐蚀4次

        if self.debugMode:
            # 字符掩码图
            self.__debugImg(edgeM, "edgeM")

        # 1.1 通过mask取出字符边缘轮廓
        edgeM1 = copy.deepcopy(~edgeM)
        if self.debugMode:
            # 字符轮廓图
            self.__debugImg(~edgeImg, "textEdgeM")
        # cv2.imwrite("step1R.jpg", edgeM1)
        return edgeM1

    def __step2(self, edgeMask):
        """
        获取低频信息并去阴影：低频信息就是灰度变化较为缓慢的地方，也可以说是边缘以外的区域
        """
        # 2.0 缩小mask提取原图字符纹理信息
        kernelsize2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edgeMask = cv2.morphologyEx(edgeMask, cv2.MORPH_CLOSE, kernel=kernelsize2)   # 填充空洞，为形态学滤波
        # cv2.imwrite("morphologyEx__step21.jpg", edgeMask)
        edgeMask = cv2.erode(edgeMask, kernelsize2, iterations=2)   # 缩小mask区域

        # 2.1 滑窗统计模糊度并自适应增强sled
        noshadowImg = self.__slideProcess() #copy.deepcopy(self.rawImg) 此处开始变成的线不太好
        # cv2.imwrite("noshadowImg__step22.jpg", noshadowImg)
        if self.debugMode:
            # 无阴影字符纹理图
            self.__debugImg(noshadowImg, "noshadowImg")

        textureImg = copy.deepcopy(noshadowImg)  # 深复制类似于深拷贝。完全复刻一个相同的变量，无论怎么改变新变量，原有变量的值都不会受到影响
        textureImg[edgeMask == 0] = 255
        # cv2.imwrite("textureImg__step23.jpg", textureImg)
        if self.debugMode:
            # 字符纹理图
            self.__debugImg(textureImg, "textureImg")

        # 2.2 纹理整体增强
        textureImg = textureImg / 255.0
        textureImg = np.power(textureImg, 2) * 255.0
        textureImg = textureImg.astype(np.uint8)  # 变成uint8类型：0-255范围内的整数
        if self.debugMode:
            # 无阴影强化字符纹理图
            self.__debugImg(textureImg, "textureImgPlus")
        # cv2.imwrite("step2R.jpg", textureImg)
        return textureImg

    def __step3(self, edgeImg, noshadowImg):
        """
        高低频信息拼接。高频信息：边缘处
        """
        highEdge = copy.deepcopy(~edgeImg)
        # 3.0 缩进边缘特征
        highEdge1 = cv2.resize(highEdge, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) # 使用像素区域关系进行重采样
        # cv2.imwrite("highEdge1_step31.jpg", highEdge1)
        highEdge2 = cv2.resize(highEdge1, dsize=(noshadowImg.shape[1], noshadowImg.shape[0]), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite("highEdge2_step32.jpg", highEdge2)
        noshadowImg[highEdge2==0] = 0
        if self.debugMode:
            # 字符复原图
            self.__debugImg(noshadowImg, "realTextImg")
        # cv2.imwrite("step3R.jpg", noshadowImg)
        return noshadowImg

    def __step4(self, noshadow):  # noshadow的图片是在mask处包含阴影的图片
        """
        锐化
        """
        noshadow = cv2.GaussianBlur(noshadow, (3, 3), 0)
        # cv2.imwrite("noshadow_step41.jpg", noshadow)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        sharpenImg = cv2.filter2D(noshadow, -1, kernel)  # 卷积滤波
        # cv2.imwrite("sharpenImg_step42.jpg", sharpenImg)

        # 去除字符阴影:方法掌握。
        n, _, _ = plt.hist(sharpenImg.ravel(), bins=255, range=(0, 255))
        # plt.show()
        satlowBkg1 = copy.deepcopy(sharpenImg)
        satlowBkg1 = satlowBkg1.ravel()
        valid_pixels = satlowBkg1[satlowBkg1<255]
        valid_max_pixel = np.max(valid_pixels)
        # print("所计算", valid_max_pixel)
        valid_max_pixel = 150           # 应计算
        sharpenImg[sharpenImg > valid_max_pixel] = valid_max_pixel  # 目的是将字符阴影部分的像素值进行修改
        # sharpenImg = cv2.normalize(sharpenImg, None, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite("normalize_step43.jpg", sharpenImg)
        if self.debugMode:
            self.__debugImg(sharpenImg, "sharpenImg")
        # cv2.imwrite("step4R.jpg", sharpenImg)
        return sharpenImg

    def __step5(self, wholeImg):
        """
        填补漏洞
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        graphImg = cv2.morphologyEx(copy.deepcopy(self.rawImg), cv2.MORPH_CLOSE, kernel, iterations=11)
        if self.debugMode:
            self.__debugImg(graphImg, "graphImg")
        # cv2.imwrite("graphImg.jpg", graphImg)
        holeMask = cv2.threshold(copy.deepcopy(graphImg), 0, 255, cv2.THRESH_OTSU)[1]
        kernelsize3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        holeMask = cv2.morphologyEx(holeMask, cv2.MORPH_CLOSE, kernel=kernelsize3, iterations=5)
        contours1st, _ = cv2.findContours(~holeMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c, contour in enumerate(contours1st):
            x, y, w, h = cv2.boundingRect(contour)
            if not ((10 < x and x+w < holeMask.shape[1]-10) and (10 < y and y+h < holeMask.shape[0]-10)):
                cv2.drawContours(holeMask, contours1st, c, 255, -1)
        backgroundImg = cv2.bitwise_and(wholeImg, wholeImg, mask=holeMask)
        holeMaskInv = ~holeMask
        holeImg = copy.deepcopy(self.rawImg)
        foregroundImg = cv2.bitwise_and(holeImg, holeImg, mask=holeMaskInv)
        dst = cv2.add(backgroundImg, foregroundImg)
        if self.debugMode:
            self.__debugImg(dst, "wholeImg")
        # cv2.imwrite("dst_step5.jpg", dst)
        return dst

    def __slideProcess(self):

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blackHatImg = cv2.morphologyEx(copy.deepcopy(self.rawImg), cv2.MORPH_BLACKHAT, kernel, iterations=20)

        blackHatImg = ~blackHatImg  # 取反操作。正负数取反 - ( a + 1 )，或者bool类型取反
        if self.debugMode:
            self.__debugImg(blackHatImg, "blackHatImg")

        x, y = 0, 0
        winW, winH = self.patchSize, self.patchSize
        blankImg = np.zeros(blackHatImg.shape, dtype=np.uint8)
        while y < blackHatImg.shape[0]:
            if (blackHatImg.shape[0] - y - winH) < winH:
                swinH = blackHatImg.shape[0] - y
            else:
                swinH = winH
            while x < blackHatImg.shape[1]:
                if (blackHatImg.shape[1] - x - winW) < winW:
                    swinW = blackHatImg.shape[1] - x
                else:
                    swinW = winH
                window = blackHatImg[y:y+swinH, x:x+swinW]

                swImg = cv2.normalize(window, None, 0, 255, cv2.NORM_MINMAX)
                blankImg[y:y+swinH, x:x+swinW] = swImg
                x += swinW
            x = 0
            y += swinH

        return blankImg

    def __debugImg(self, contentImg, titleText="blank"):
        self.process = np.concatenate((self.process, contentImg), axis=1)
        titleImg = np.ones((100, self.rawImg.shape[1]), dtype=np.uint8)*255
        cv2.putText(titleImg, titleText, (int(self.rawImg.shape[1]*0.45), 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 5)
        self.title = np.concatenate((self.title, titleImg), axis=1)


def singleTest(file, debug=0):
    import os
    import time
    t0 = time.time()
    postfix = os.path.splitext(file)[1]
    newname = file.replace(postfix, "-s2s.png")
    src = cv2.imdecode(np.fromfile(file,dtype=np.uint8), 0)
    removeShadow = ShadowsRemoval(src, debugMode=debug)

    if removeShadow.needEnhance:
        print("need enhance")
    else:
        print("don't need enhance")
    display = removeShadow.enhanceImg()
    # print("slideStep:", removeShadow.slideStep)
    t1 = time.time()
    # print("spend on:", t1 - t0)
    if debug:
        process = removeShadow.process
        dst = Image.fromarray(process)
        dst.save(newname.replace(".png", "_debugs.png"))
    # result = Image.fromarray(display)
    # result.save(newname)
    return display


def singleTest_add(src, debug=0):
    import os
    import time
    t0 = time.time()
    # postfix = os.path.splitext(file)[1]
    # newname = file.replace(postfix, "-s2s.png")
    # src = cv2.imdecode(np.fromfile(file,dtype=np.uint8), 0)
    removeShadow = ShadowsRemoval(src, debugMode=debug)

    if removeShadow.needEnhance:
        print("need enhance")
    else:
        print("don't need enhance")
    display = removeShadow.enhanceImg()
    print("slideStep:", removeShadow.slideStep)
    t1 = time.time()
    # print("spend on:", t1 - t0)
    # if debug:
    #     process = removeShadow.process
    #     dst = Image.fromarray(process)
        # dst.save(newname.replace(".png", "_debugs.png"))
    # result = Image.fromarray(display)
    # result.save(newname)
    return display


def batchTest(rootPath, postfixs):
    import os
    from glob import glob
    import sys
    all_files = []
    for postfix in postfixs:
        all_file = glob(os.path.join(rootPath, "*" + postfix))  # 匹配模式，返回所有后缀名为指定名字的文件
        all_files += all_file
    print("文件通配符", all_files)
    total_file = len(all_files)
    for i, file in enumerate(all_files):
        newname = file.replace("up_final_data", "up_de_final_data")
        print("-------------------------no.{}------------------------".format(i))
        print(file)
        # newname = file.replace(postfix, "_s2_gt" + postfix)
        src = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 0)
        removeShadow = ShadowsRemoval(src)
        display = removeShadow.enhanceImg()
        # display = np.concatenate((src, display), axis=1)
        dst = Image.fromarray(display)

        cv2.imwrite(newname, display)
    #     dst.save(newname)
    #     sys.stdout.write('\r>> generate gt_image %d/%d\t' % (i + 1, total_file))
    #     print(newname)
    #     sys.stdout.flush()
    # sys.stdout.write('\n')
    # sys.stdout.flush()


if __name__ == '__main__':

    # img = cv2.imread(r'D:\11\updown_data\pic\1639982613923.png',1)
    # display = singleTest(r"D:\11\updown_data\pic\1639982613923.png", 1)

    # cv2.imwrite("display.jpg",display)
    # singleTest("I:/【测试数据】/去阴影 问题图片/questionImg/869.png", 1)
    batchTest(r"/home/lq4/SSDProject/workHandover/5_updownJudgement/code/iphoneData/train/up_final_data", [".JPG", ".png", ".jpg"])  # 得到的是一个二值化的图像。在原图中进行变化。

