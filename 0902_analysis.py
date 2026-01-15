# 0902.py
import cv2
import numpy as np
 
src = cv2.imread('./data/chessBoard.jpg')
gray= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

#1
fastF = cv2.FastFeatureDetector_create()
kp = fastF.detect(gray) 
dst = cv2.drawKeypoints(gray, kp, None, color=(255,0,0))
print('len(kp)=', len(kp))

#2
kp = sorted(kp, key=lambda f: f.response, reverse=True)
# cv2.drawKeypoints(gray, kp[:10], dst, color=(0,0,255),
cv2.drawKeypoints(gray, kp[:38], dst, color=(0,0,255),
                 flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
cv2.imshow('dst',  dst)

#3
kp2 = list(filter(lambda f: f.response>50, kp))
print('len(kp2)=', len(kp2))
for f in kp2:
    print(f.response)

dst2 = cv2.drawKeypoints(gray, kp2, None, color=(0,0,255))   
cv2.imshow('dst2',  dst2)

#4
def distance(f1, f2):    
    x1, y1 = f1.pt
    x2, y2 = f2.pt
    return np.sqrt((x2 - x1)**2+ (y2 - y1)**2)

def filteringByDistance(kp, distE=0.5):
    size = len(kp)    
    mask = np.arange(1,size+1).astype(np.bool8) # all True   
    for i, f1 in enumerate(kp):
        if i//10 == 0: print('mask ', i, ' th: ', mask) # for analysis
        # if i//10 == 0: print('mask ', i, ' th: ', list(map(int, mask))) # for analysis
        #if i%10 == 0: print('mask ', i, ' th: ', list(map(int, mask))) # for analysis
        if not mask[i]:
            continue
        else: # True
            for j, f2 in enumerate(kp):
                if i == j:
                    continue
                if distance(f1, f2)<distE:
                    mask[j] = False
    np_kp = np.array(kp)
    # print('list(np_kp[mask])', np_kp[mask])
    return list(np_kp[mask])

kp3 = filteringByDistance(kp2, 30)
print('len(kp3)=', len(kp3))
dst3 = cv2.drawKeypoints(gray, kp3, None, color=(0,0,255))
cv2.imshow('dst3',  dst3)
cv2.waitKey()
cv2.destroyAllWindows()
