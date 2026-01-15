# 0603.py
import cv2
import numpy as np

#src = cv2.imread('./data/rect.jpg', cv2.IMREAD_GRAYSCALE)
src = np.array([[1, 1, 100, 1],
                [1, 1, 100, 1],
                [10, 10, 100, 10],
                [1, 1, 100, 1]], dtype=np.uint8)
#1
gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize = 1)
gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize = 1)
print('gx:\n', gx)
print('gy:\n', gy)

#2
dstX = cv2.sqrt(np.abs(gx))
print('dstX:\n', dstX)
dstX = cv2.normalize(dstX, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
print('normalized dstX:\n', dstX)

#3
dstY = cv2.sqrt(np.abs(gy))
print('dstY:\n', dstY)
dstY = cv2.normalize(dstY, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
print('normalized dstY:\n', dstY)

#4
mag   = cv2.magnitude(gx, gy)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mag)
print('minVal, maxVal, minLoc, maxLoc of mag:', minVal, maxVal, minLoc, maxLoc)

dstM = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# cv2.imshow('src',  src)
# cv2.imshow('dstX',  dstX)    
# cv2.imshow('dstY',  dstY)
# cv2.imshow('dstM',  dstM)
cv2.waitKey()
cv2.destroyAllWindows()
