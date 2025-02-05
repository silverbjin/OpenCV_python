# 1109.py
import cv2
import numpy as np
np.set_printoptions(precision=2, suppress=True)

#1: open video capture
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./data/chess1.wmv')
if (not cap.isOpened()): 
     print('Error opening video')
     import sys
     sys.exit()
     
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
imageSize = width, height

#2
patternSize = (6, 3)
def FindCornerPoints(src_img, patternSize):
    found, corners = cv2.findChessboardCorners(src_img, patternSize)
    if not found:
        return found, corners
    
    term_crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), term_crit)
    corners = corners[::-1] # reverse order, in this example, to set origin to (left-upper)
    return found, corners

#3: set world(object) coordinates to Z = 0
xN, yN = patternSize # (6, 3)
mW = np.zeros((xN*yN, 3), np.float32) # (18, 3)
mW[:, :2] = np.mgrid[0:xN, 0:yN].T.reshape(-1, 2) # mW points on Z = 0
mW[:, :2]+= 1 # (1, 1, 0): coord of the start corner point in the pattern
   
#4: load camera matrix K
with np.load('./data/calib_1104.npz') as X:
    K = X['K']
    dists = X['dists']  
#dists = np.zeros(5)
print("K=\n", K)
print("dists=", dists)


#5: calculate K from obj_points, img_points
##t = 0
##count = 0
##N_FRAMES = 10
##obj_points = [ ]
##img_points = [ ]
##
##while True:
###5-1: build obj_points, img_points 
##    ret, frame = cap.read()
##    if not ret:
##        break
##    found, corners = FindCornerPoints(frame, patternSize)
##    if not found:
##        cv2.imshow('frame',frame)
##        key = cv2.waitKey(20)
##        if key == 27:  break
##        continue
##    
##    t+= 1
##    if t%10 != 0: # sample 
##        continue
##    
##    if found and count<N_FRAMES:
##        obj_points.append(mW)
##        img_points.append(corners)
##        cv2.drawChessboardCorners(frame, patternSize, corners, found)
##        count += 1
##    else:
##        break       
##    cv2.imshow('frame',frame)  
##    key = cv2.waitKey(20)
##    if key == 27:
##        break
##
###5-2: calibrate camera matrix
##K = cv2.initCameraMatrix2D(obj_points, img_points, imageSize)
##errors, K, dists, _, _= cv2.calibrateCamera(
##                                obj_points,img_points, imageSize, None, None)
##np.savez('./data/calib_1109.npz', K=K, dists= dists)
##print("K=\n", K)
##print("dists=", dists)

#6: calibrate rvec and tvec, draw axis, object, errors 
index = [0, 5, 17, 12] # 4-corner index
axis3d = np.float32([[0,0,0], [3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

t = 0 # frame counter
while True:
#6-1
    ret, frame = cap.read()
    if not ret:
        break

    found, corners = FindCornerPoints(frame, patternSize)
    if not found:
        cv2.imshow('frame',frame)
        key = cv2.waitKey(20)
        if key == 27:  break
        continue
    
    ret, rvec, tvec = cv2.solvePnP(mW, corners, K, dists)

#6-2: display axis
    axis_2d, _ = cv2.projectPoints(axis3d, rvec, tvec, K, dists)
    axis_2d    = np.int32(axis_2d).reshape(-1,2)
    cv2.line(frame, tuple(axis_2d[0]), tuple(axis_2d[1]),(255, 0, 0),3)
    cv2.line(frame, tuple(axis_2d[0]), tuple(axis_2d[2]),(0, 255, 0),3)
    cv2.line(frame, tuple(axis_2d[0]), tuple(axis_2d[3]),(0, 0, 255),3)
    
#6-3: display pW on Z = 0
    pW = mW[index]  # 4-corners' coord (x, y, 0)
    p1, _ = cv2.projectPoints(pW, rvec, tvec, K, dists)
    p1    = np.int32(p1)
    
    cv2.drawContours(frame, [p1],-1,(0,255,255), -1)
    cv2.polylines(frame,[p1],True,(0,255,0), 2)

#6-4: display pW on Z = -2    
    pW[:, 2] = -2 # 4-corners' coord (x, y, -2)   
    p2, _ = cv2.projectPoints(pW, rvec, tvec, K, dists)
    p2    = np.int32(p2)
    cv2.polylines(frame,[p2],True,(0,0,255), 2)

#6-5: display edges between two rectangles 
    for j in range(4):
        x1, y1 = p1[j][0] # Z = 0
        x2, y2 = p2[j][0] # Z = -2
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
#6-6: re-projection errors
    pts, _ = cv2.projectPoints(mW, rvec, tvec, K, dists) 
    errs = cv2.norm(corners, np.float32(pts))
    #print("errs[{}]={:.2f}".format(t, errs))
    t += 1
    cv2.imshow('frame',frame)  
    key = cv2.waitKey(20) 
    if key == 27:
        break
#7
if cap.isOpened(): cap.release()
cv2.destroyAllWindows()
 





