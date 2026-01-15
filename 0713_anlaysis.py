# 0713_anlaysis.py
import cv2
import numpy as np
#1
# def floodFillPostProcess(src, diff=(2,2,2)):
def floodFillPostProcess(src, diff=(0,0,0)):
    img = src.copy()
    rows, cols = img.shape[:2]
    mask   = np.zeros(shape=(rows+2, cols+2), dtype=np.uint8)
    print('img', img)       # for analysis
    print('mask', mask)     # for analysis
    for y in range(rows):
        for x in range(cols):
            if mask[y+1, x+1] == 0:
                r = np.random.randint(256)
                g = np.random.randint(256)
                b = np.random.randint(256)
                cv2.floodFill(img,mask,(x,y),(b,g,r),diff,diff)
                print('img', img)       # for analysis
                print('mask', mask)     # for analysis
    return img

# for analysis
src1 = np.array([[0, 0, 0, 0],
                [1, 1, 3, 5],
                [6, 1, 1, 3],
                [4, 3, 1, 7]
              ], dtype=np.uint8)
dst1  = floodFillPostProcess(src1)

#2
src = cv2.imread('./data/flower.jpg')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
dst  = floodFillPostProcess(src)
dst2 = floodFillPostProcess(hsv)

cv2.imshow('src',src)
cv2.imshow('hsv',hsv)
cv2.imshow('dst',dst)
cv2.imshow('dst2',dst2)
#3
res = cv2.pyrMeanShiftFiltering(src, sp=5, sr=20, maxLevel=4)
dst3 = floodFillPostProcess(res)

#4
term_crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 2)
res2=cv2.pyrMeanShiftFiltering(hsv,sp=5,sr=20,maxLevel=4, termcrit=term_crit)
dst4 = floodFillPostProcess(res2)

cv2.imshow('res',res)
cv2.imshow('res2',res2)
cv2.imshow('dst3',dst3)
cv2.imshow('dst4',dst4)
cv2.waitKey()
cv2.destroyAllWindows()
