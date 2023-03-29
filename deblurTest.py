import deblur220113
import cv2
import numpy as np
import matplotlib as plt
imgPath = r"E:\workHandover\2_denoiseAndShadow\data\20220424105436.jpg"

# src = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), 0)
# removeShadow = deblur220113.ShadowsRemoval(src)
a = np.ones([3, 4])
print(a)
a[a==1] = 0
print(a)