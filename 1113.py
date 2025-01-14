# 1113.py
import cv2
import numpy as np
np.set_printoptions(precision=2, suppress=True)

#1: open video 
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./data/chess1.wmv')
if (not cap.isOpened()): 
     print('Error opening video')
     import sys
     sys.exit()
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

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
with np.load('./data/calib_1104.npz') as X: # './data/calib_1109.npz'
    K = X['K']
    #dists = X['dists']
dists = np.zeros(5)
print("K=\n", K)
print("dists=", dists)

#5: camera displacement, homography, plane normal and origin
#5-1: the camera displacement
def computeC2MC1(rvec1, t1, rvec2, t2):
    R1 = cv2.Rodrigues(rvec1)[0] # vector to matrix
    R2 = cv2.Rodrigues(rvec2)[0]

    R_1to2 = np.dot(R2, R1.T)
    r_1to2 = cv2.Rodrigues(R_1to2)[0]
    
    t_1to2 = np.dot(R2, np.dot(-R1.T, t1)) + t2
    return r_1to2, t_1to2

#5-2:
def computeHomography(rvec_1to2, tvec_1to2, d, normal):
    R_1to2 = cv2.Rodrigues(rvec_1to2)[0] # vector to matrix    
    homography = R_1to2 + np.dot(tvec_1to2, normal.T)/d
    return homography

#5-3: the plane normal and origin on calibration pattern
normal = np.array([0., 0.,  1.]).reshape(3, 1)  # +Z
origin = np.array([0., 0., 0.]).reshape(3, 1)

#6: pose estimation from H, project, and re-projection errors
index = [0, 5, 17, 12] # 4-corner index
axis3d = np.float32([[0,0,0], [3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
method = cv2.LMEDS # cv2.RANSAC

t = 1 # frame counter
bInit = True
while True:
#6-1
    ret, frame = cap.read()
    if not ret:
        break
    found, corners2 = FindCornerPoints(frame, patternSize)
    #cv2.drawChessboardCorners(frame, patternSize, corners2, found)

    if not found:
        cv2.imshow('frame',frame)
        key = cv2.waitKey(20)
        if key == 27:  break
        bInit = True
        continue
#6-2: 
    if bInit: # find (rvec1, tvec1) : mW->corners, in 1st frame
        print("Initialize.......")
        corners1 = corners2.copy()
        ret1, rvec1, tvec1 = cv2.solvePnP(mW, corners1, K, dists)
        bInit = False
        #continue

#6-3: the plane distance to the camera 1
    R1 = cv2.Rodrigues(rvec1)[0] # vector to matrix
    normal1 = np.dot(R1, normal)
    origin1 = np.dot(R1, origin) + tvec1
    d1 = np.sum(normal1*origin1)
    #print("d1=", d1)
    
#6-4: homography from the camera displacement
    ret2, rvec2, tvec2 = cv2.solvePnP(mW, corners2, K, dists)    
    rvec_1to2, tvec_1to2 = computeC2MC1(rvec1, tvec1, rvec2, tvec2)
    homography_euclidean = computeHomography(rvec_1to2, tvec_1to2, d1, normal1)
    H = np.dot(np.dot(K, homography_euclidean), np.linalg.inv(K))
    H /= H[2,2]
    #print("H=", H)

#6-5: homography decomposition and filtering  
    ret, Rs_decomp, ts_decomp, ns_decomp = cv2.decomposeHomographyMat(H, K)
    if ret == 1:
        min_i = 0 # solutions = [0]
    else:
        solutions = cv2.filterHomographyDecompByVisibleRefpoints(Rs_decomp, ns_decomp, corners1, corners2)
        if solutions is None:
            print("----------no solutions! ----------")
            print("ret=", ret)
            #cv2.waitKey(0)
            #bInit = False
            continue
        else:
            solutions = solutions.flatten()

            #find a solution with minimum re-projection errors
            min_errors = 1.0E5
            for i in solutions:
                rvec_decomp = cv2.Rodrigues(Rs_decomp[i])[0]
                tvec_decomp = ts_decomp[i]*d1 # scale by the plane distance to the camera 1
                
                # re-projection errors
                rvec3, tvec3 = cv2.composeRT(rvec1, tvec1, rvec_decomp, tvec_decomp)[:2]
                pts, _ = cv2.projectPoints(mW, rvec3, tvec3, K, dists) 
                errs = cv2.norm(corners2, np.float32(pts))
                #print("errs[{}]={}".format(i, errs))
                if errs < min_errors:
                    min_errors = errs
                    min_i = i

    # the decomposition with minimum errors
    rvec_decomp = cv2.Rodrigues(Rs_decomp[min_i])[0]    
    tvec_decomp = ts_decomp[min_i]*d1 # scale by the plane distance to the camera 1
    nvec_decomp = ns_decomp[min_i]

#6-6: pose estimation
    rvec, tvec = cv2.composeRT(rvec1, tvec1, rvec_decomp, tvec_decomp)[:2]
    
    # copy for next frame
    corners1 = corners2.copy()
    rvec1 = rvec  #rvec2
    tvec1 = tvec  #tvec2
    
#6-7: display axis and cube
    axis_2d, _ = cv2.projectPoints(axis3d, rvec, tvec, K, dists)
    axis_2d    = np.int32(axis_2d).reshape(-1,2)
    cv2.line(frame, tuple(axis_2d[0]), tuple(axis_2d[1]),(255, 0, 0),3)
    cv2.line(frame, tuple(axis_2d[0]), tuple(axis_2d[2]),(0, 255, 0),3)
    cv2.line(frame, tuple(axis_2d[0]), tuple(axis_2d[3]),(0, 0, 255),3)
    
    #display pW on Z = 0
    pW = mW[index]  # 4-corners' coord (x, y, 0)
    p1, _ = cv2.projectPoints(pW, rvec, tvec, K, dists)
    p1    = np.int32(p1)
    
    cv2.drawContours(frame, [p1],-1,(0,255,255), -1)
    cv2.polylines(frame,[p1],True,(0,255,0), 2)

    #display pW on Z = -2    
    pW[:, 2] = -2 # 4-corners' coord (x, y, -2)   
    p2, _ = cv2.projectPoints(pW, rvec, tvec, K, dists)
    p2    = np.int32(p2)
    cv2.polylines(frame,[p2],True,(0,0,255), 2)

    #display edges between two rectangles 
    for j in range(4):
        x1, y1 = p1[j][0] # Z = 0
        x2, y2 = p2[j][0] # Z = -2
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
#6-8: re-projection errors
    pts, _ = cv2.projectPoints(mW, rvec, tvec, K, dists) 
    errs = cv2.norm(corners2, np.float32(pts))
    #print("errs[{}]={:.2f}".format(t, errs))
    t += 1
     
    cv2.imshow('frame',frame)  
    key = cv2.waitKey(20)
    if key == 27:
        break     
#7
if cap.isOpened(): cap.release()
cv2.destroyAllWindows()
 





