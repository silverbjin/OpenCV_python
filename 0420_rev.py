# 0420.py
import cv2
import numpy as np

src = cv2.imread('./data/lena.jpg', cv2.IMREAD_GRAYSCALE)

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src)
print('src:', minVal, maxVal, minLoc, maxLoc)

dst = cv2.normalize(src, None, 100, 200, cv2.NORM_MINMAX)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dst)
print('dst:', minVal, maxVal, minLoc, maxLoc)

cv2.imshow('src',  src)
cv2.imshow('dst',  dst)


#-- 직접 연산한 정규화
img_f = src.astype(np.float32)
img_norm = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min()))
img_norm = img_norm.astype(np.uint8)
#-- OpenCV API를 이용한 정규화
img_norm2 = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX)
#-- 히스토그램 계산
hist = cv2.calcHist([src], [0], None, [256], [0, 255])
hist_norm = cv2.calcHist([img_norm], [0], None, [256], [0, 255])
hist_norm2 = cv2.calcHist([img_norm2], [0], None, [256], [0, 255])

cv2.imshow('Before', src)
cv2.imshow('Manual', img_norm)
cv2.imshow('cv2.normalize()', img_norm2)
cv2.waitKey(500)    

import matplotlib.pylab as plt
#histogram
hist_src = cv2.calcHist([src], [0], None, [256], [0, 255])
hist_dst = cv2.calcHist([dst], [0], None, [256], [0, 255])
hists = {'src' : hist_src, 'dst':hist_dst, 'Before' : hist, 'Manual':hist_norm, 'cv2.normalize()':hist_norm2}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1, len(hists.items()) ,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()

cv2.waitKey()    
cv2.destroyAllWindows()
