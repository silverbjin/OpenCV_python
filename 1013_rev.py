# 1013.py
import cv2
import numpy as np

#1
src1 = cv2.imread('./data/book3.jpg')
img1= cv2.cvtColor(src1,cv2.COLOR_BGR2GRAY)

#F = cv2.ORB_create(scoreType=1)  # FAST_SCORE
# F = cv2.SIFT_create()
F = cv2.xfeatures2d.SIFT_create()
kp1, des1 = F.detectAndCompute(img1, None) 
flan = cv2.FlannBasedMatcher_create()

#2
cap = cv2.VideoCapture('./data/book3.mp4')
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("frame_size =", frame_size)

#3
nndrRatio = 0.45  # 0.45
h, w = img1.shape
t = 0
while True:    
    retval, frame = cap.read() # 프레임 획득
    if not retval: break
    t+=1
##    print('t=',t)
#3-1
    src2 = frame.copy()
    img2= cv2.cvtColor(src2,cv2.COLOR_BGR2GRAY)
    kp2, des2 = F.detectAndCompute(img2, None)
    matches = flan.knnMatch(np.float32(des1), np.float32(des2), k=2)

#3-2
    good_matches = [f1 for f1, f2 in matches
                       if f1.distance < nndrRatio*f2.distance]
    dst = cv2.drawMatches(src1,kp1,src2,kp2, good_matches, None,
                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    if len(good_matches) < 5:
        print('sorry, too small good matches', len(good_matches))
        cv2.imshow('dst',  dst)
        key = cv2.waitKey(50)
        if key == 27: break
        continue
#3-3    
    src1_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches])
    src2_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches])
    H, mask=cv2.findHomography(src1_pts,src2_pts, cv2.RANSAC, 2.0)
    mask_matches = mask.ravel().tolist() # list(mask.flatten())

#3-4
    if H is None:
        print('sorry, no H')
        continue   
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    pts2 = cv2.perspectiveTransform(pts, H)
    src2 = cv2.polylines(src2,[np.int32(pts2)],True,(255,0, 0),2)
            
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = None,
                       matchesMask = mask_matches, flags = 2)                 
    dst = cv2.drawMatches(src1,kp1,src2,kp2, good_matches, None,**draw_params)
    cv2.imshow('dst',  dst)
        
    key = cv2.waitKey(25)
    if key == 27: # Esc
        break
cap.release()
cv2.destroyAllWindows()
