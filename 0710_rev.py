# 0710.py
import cv2
import numpy as np

#1
src = cv2.imread('./data/hand.jpg')
##src = cv2.imread('./data/flower.jpg')
mask   = np.zeros(shape=src.shape[:2], dtype=np.uint8)
markers= np.zeros(shape=src.shape[:2], dtype=np.int32)
print('src shape: ', src.shape[:])
print('src shape[:2]: ', src.shape[:2])
dst = src.copy()
cv2.imshow('dst',dst)

#2
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(param[0], (x, y), 10, (255, 255, 255), -1)
            cv2.circle(param[1], (x, y), 10, (255, 255, 255), -1) 
    cv2.imshow('mask', param[0])    
    cv2.imshow('dst', param[1])    
##cv2.setMouseCallback('dst', onMouse, [mask, dst])

#3
mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
while True:
    cv2.setMouseCallback('dst', onMouse, [mask, dst]) #3-1
    key = cv2.waitKey(30) # cv2.waitKeyEx(30)
    
    if key == 0x1B: 
        break
    elif key == ord('r'): #3-2
        mask[:,:] = 0        
        dst = src.copy()
        cv2.imshow('dst',dst)        
    elif key == ord(' '): #3-3
        image, contours, hierarchy = cv2.findContours(mask, mode, method)
        # contours, hierarchy = cv2.findContours(mask, mode, method)
        print('len(contours)=', len(contours))
        
        # print('contours=', contours)
        print('contours[0].shape=', contours[0].shape)
        # print('contours[0]=', contours[0])
        print('contours[1].shape=', contours[1].shape)
        
        markers[:,:] = 0  
        dst_markers = src.copy()        # for analysis        
        for i, cnt in enumerate(contours):
            cv2.drawContours(markers, [cnt], 0, i+1, -1)
            cv2.drawContours(dst_markers, [cnt], 0, (255,0,0), 2)   # for analysis
        # cv2.imshow('markers', markers.astype(np.uint8))           # for analysis
        markers_analysis = cv2.convertScaleAbs(markers)             # for analysis
        markers_analysis = cv2.normalize(markers_analysis, None, 0, 255, cv2.NORM_MINMAX)   # for analysis
        cv2.imshow('markers_analysis', markers_analysis)             # for analysis
        
        cv2.watershed(src,  markers)

        #3-4        
        dst = src.copy()
        dst[markers == -1] = [0,0,255] # 경계선
        dst_markers[markers == -1] = [0,0,255] # 경계선         # for analysis
        cv2.imshow('dst_markers', dst_markers)                 # for analysis
        for i in range(len(contours)): # 분할영역 
          r = np.random.randint(256)
          g = np.random.randint(256)
          b = np.random.randint(256)
          dst[markers == i+1] = [b, g, r]

        dst = cv2.addWeighted(src, 0.4, dst, 0.6, 0) # 합성
        cv2.imshow('dst',dst)        
cv2.destroyAllWindows()
