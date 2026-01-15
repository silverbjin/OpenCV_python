# 0709.py
import cv2
import numpy as np

#1
# src = np.zeros(shape=(512,512), dtype=np.uint8)
# cv2.rectangle(src, (50, 200), (450, 300), (255, 255, 255), -1)
src = np.zeros(shape=(5,5), dtype=np.uint8)
cv2.rectangle(src, (1, 1), (3, 3), 255, -1)

#2
dist  = cv2.distanceTransform(src, distanceType=cv2.DIST_L1, maskSize=3)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
# dist8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # for visualization
# cv2.imshow('dist', dist.astype(np.uint8)) #error # dist의 type은 float32임으로 uint8로 변경해서 arg로 넣어야 함.
# cv2.imshow('dist8', dist8) # dist의 type은 float32임으로 uint8로 변경해서 arg로 넣어야 함.
print('src:\n', src)
print('dist:\n', dist)
print('src_info:', minVal, maxVal, minLoc, maxLoc)

# dist_transform  함수를 사용하면 실수 타입(float32)의 이미지가 생성. 
# 화면 출력을 위해 normalize 함수를 사용
dst = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
ret, dst2 = cv2.threshold(dist, maxVal-1, 255, cv2.THRESH_BINARY)
# cv2.circle(dst, maxLoc, 1, 0, 1)

#3 
gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize = 3)
gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize = 3)
mag   = cv2.magnitude(gx, gy)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mag)
print('src:', minVal, maxVal, minLoc, maxLoc)
ret, dst3 = cv2.threshold(mag, maxVal-3, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('src',  src)
cv2.imshow('dst',  dst)
cv2.imshow('dst2',  dst2)
cv2.imshow('dst3',  dst3)
cv2.waitKey()
cv2.destroyAllWindows()
