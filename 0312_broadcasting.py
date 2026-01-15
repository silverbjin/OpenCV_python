# 0312.py
import numpy as np
import cv2

def onChange(pos): # 트랙바 핸들러
    global img
    r = cv2.getTrackbarPos('R','img')
    g = cv2.getTrackbarPos('G','img')
    b = cv2.getTrackbarPos('B','img')                       
    img[:] = (b, g, r)
    cv2.imshow('img', img)

img = np.zeros((512, 512, 3), np.uint8)

test_img = np.zeros((3, 3, 2), np.uint8)

print("*"*20,"slicing" ,"*"*20)
print(test_img[:])

print("*"*20,"broadcasting(same axis(=2) length)" ,"*"*20)
test_img[:] = (1, 2)
print(test_img[:])

print("*"*20,"broadcasting(one element)" ,"*"*20)
test_img[:] = 3
print(test_img[:])

