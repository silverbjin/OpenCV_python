# 0809.py
import cv2
import numpy as np

#1
# src = cv2.imread('./data/momentTest.jpg')
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# ret, bImage = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# for analysis
src = np.zeros(shape=(5,5), dtype=np.uint8)
cv2.rectangle(src, (1, 1), (3, 3), 255, -1)
bImage = src
print('src: \n', src)

#2
##M = cv2.moments(bImage)   
M = cv2.moments(bImage, True)
for key, value in M.items():
    print('{}={}'.format(key, value))
    
#3
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print('cx: ', cx, 'cy: ', cy)
dst = src.copy()
cv2.circle(dst, (cx, cy), 5, (0,0,255), 2)

cv2.imshow('dst',  dst)
cv2.waitKey()
cv2.destroyAllWindows()
