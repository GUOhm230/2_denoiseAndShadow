乐谱拍摄功能中的第二步
对图片进行去阴影并二值化
但是对于低分辨率的图片，效果则较差
直接运行：deblur220113.py。修改    
batchTest(r"/home/lq4/SSDProject/ImageGAN/image_superResolution/NAFNet/infer_result/xiaoyezi_deblur", [".JPG", ".png", ".jpg"])  # 得到的是一个二值化的图像。在原图中进行变化。
以及
newname = file.replace("xiaoyezi_deblur", "xiaoyezi_deblur_denoise")

