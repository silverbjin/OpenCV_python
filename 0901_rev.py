# 0901.py
import cv2
import numpy as np
 
src = cv2.imread('./data/chessBoard.jpg')
gray= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

#1
##fastF = cv2.FastFeatureDetector_create()
##fastF =cv2.FastFeatureDetector.create()
fastF =cv2.FastFeatureDetector_create(threshold=30) # 100
kp = fastF.detect(gray) 
print('type(kp): ', type(kp))
print('len(kp): ', len(kp))
print('kp: ', kp)
print('type(kp[0]): ', type(kp[0]))
print('kp[0]: ', kp[0])
print('kp[0].class_id: ', kp[0].class_id)
print('kp[0].octave: ', kp[0].octave)
print('kp[0].pt (feature points): ', kp[0].pt)
print('kp[0].response (feature value): ', kp[0].response)
print('kp[0].size (feature neighbor diameter): ', kp[0].size)
print('kp[0].angle (gradient): ', kp[0].angle)
dst = cv2.drawKeypoints(gray, kp, None, color=(0,0,255))
print('len(kp)=', len(kp))
cv2.imshow('dst',  dst)

#2
fastF.setNonmaxSuppression(False)
kp2 = fastF.detect(gray)
dst2 = cv2.drawKeypoints(src, kp2, None, color=(0,0,255))
print('len(kp2)=', len(kp2))
cv2.imshow('dst2',  dst2)

#3
point = []
dst3 = src.copy()
points = cv2.KeyPoint_convert(kp)
print('-'*50)
print('type(kp): ', type(kp))
print('len(kp): ', len(kp))
#print('kp: ', kp)
print('+'*50)
print('type(kp[0]): ', type(kp[0]))
print('kp[0]: ', kp[0])
#print('kp[0].class_id: ', kp[0].class_id)
#print('kp[0].octave: ', kp[0].octave)
print('kp[0].pt (feature points): ', kp[0].pt)
print('kp[0].size (feature neighbor diameter): ', kp[0].size)
print('*'*50)
print('type(points): ', type(points))
print('points.shape: ', points.shape)
# print('points: \n', points)
#print('points: ', points)
print('/'*50)
print('type(points[0]): ', type(points[0]))
print('points[0].shape: ', points[0].shape)
print('points[0]: ', points[0])
print('-'*50)

for cx, cy in points:
    cv2.circle(dst3, (int(cx), int(cy)), 3, color=(255, 0, 0), thickness=1)
cv2.imshow('dst3',  dst3)
cv2.waitKey()
cv2.destroyAllWindows()
