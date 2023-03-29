import cv2
import numpy as np
from matplotlib import pyplot as plt

imgTest = r"E:\workHandover\2_denoiseAndShadow\data\20220427121622.jpg"
img = cv2.imread(imgTest)
edges = cv2.Canny(img, 100, 200, L2gradient=False)
edges2 = cv2.Canny(img, 100, 200, L2gradient=True)
cv2.imwrite("canny_L2F.jpg", edges)
cv2.imwrite("canny_L2T.jpg", edges2)
# 绘图学习一下
plt.subplot(131),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(edges, cmap = 'gray')
plt.title('Edge Image1'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(edges2, cmap = 'gray')
plt.title('Edge Image2'), plt.xticks([]), plt.yticks([])
plt.show()
