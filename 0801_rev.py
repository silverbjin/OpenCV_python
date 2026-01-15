# 0801.py
import cv2
import numpy as np
#1
def findLocalMaxima(src):
    # kernel= cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(11,11))
    kernel= cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))        # for analysis
    dilate = cv2.dilate(src,kernel)# local max if kernel=None, 3x3 
    localMax = (src == dilate)
    print('src: \n', src)               # for analysis
    print('kernel: \n', kernel)         # for analysis
    print('dilate: \n', dilate)         # for analysis
    print('localMax: \n', localMax)     # for analysis
    
    erode = cv2.erode(src,kernel) #  local min if kernel=None, 3x3 
    localMax2 = src > erode      
    localMax &= localMax2
    points = np.argwhere(localMax == True)
    points[:,[0, 1]] = points[:,[1, 0]] # switch x, y 
    print('erode: \n', erode)               # for analysis
    print('localMax2: \n', localMax2)       # for analysis
    print('points: \n', points)             # for analysis
    return points

# for analysis
res0 = np.array([[0, 0, 0, 4, 0, 0, 0],
                 [0, 1, 4, 4, 4, 0, 0],
                 [0, 2, 4, 4, 4, 0, 0],
                #  [0, 1, 4, 5, 5, 0, 0],
                 [0, 1, 4, 5, 6, 0, 0],
                 [0, 6, 1, 1, 3, 0, 0],
                 [4, 4, 4, 1, 2, 0, 0],
                 [4, 4, 4, 1, 2, 0, 0]
              ], dtype=np.uint8)
ret, res1 = cv2.threshold(res0, 4, 0, cv2.THRESH_TOZERO)
corners1 = findLocalMaxima(res1)

#2
res0 = np.array([[255, 255, 255, 255, 255, 255, 255],
                 [255, 255, 255, 255, 255, 255, 255],
                 [255, 255, 0, 0, 0, 255, 255],  
                 [255, 255, 0, 0, 0, 255, 255],
                 [255, 255, 0, 0, 0, 255, 255],
                 [255, 255, 255, 255, 255, 255, 255],
                 [255, 255, 255, 255, 255, 255, 255]
              ], dtype=np.uint8)

src = cv2.imread('./data/CornerTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
res = cv2.preCornerDetect(gray, ksize=3)
res_test = cv2.preCornerDetect(res0, ksize=3)
print("input: \n", res0)
print("res_test: \n", res_test.astype(np.uint8))
ret, res2 = cv2.threshold(np.abs(res), 0.1, 0, cv2.THRESH_TOZERO)
corners = findLocalMaxima(res2)
print('corners.shape=', corners.shape)
print('corners: ', corners)

#3
dst = src.copy()  
for x, y in corners:    
    cv2.circle(dst, (x, y), 5, (0,0,255), 2)

cv2.imshow('dst',  dst)
cv2.waitKey()
cv2.destroyAllWindows()
