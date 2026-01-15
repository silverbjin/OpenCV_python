# 0422.py
import cv2
import numpy as np
import time

dst = np.full((512,512,3), (255, 255, 255), dtype= np.uint8)
nPoints = 100
pts = np.zeros((1, nPoints, 2), dtype=np.uint16)

cv2.setRNGSeed(int(time.time()))
cv2.randn(pts, mean=(256, 256), stddev=(50, 50))

X, Y = [], []
# draw points
for k in range(nPoints):
    x, y = pts[0][k, :] # pts[0, k, :]
    X.append(x)
    Y.append(y)
    cv2.circle(dst,(x,y),radius=5,color=(0,0,255),thickness=-1)

cv2.imshow('dst',  dst)
cv2.waitKey(100)    

#plot
#print(X)
#print(Y)
import matplotlib.pylab as plt
bins = np.arange(-50,562,1)
hist_src, bins = np.histogram(X, bins)
hist_src2, bins = np.histogram(Y, bins)
#hist_dst = cv2.calcHist([dst], [0], None, [256], [0, 255])

fig, ax = plt.subplots(1,2)
fig.canvas.set_window_title("norminal pdf")
ax[0].plot(hist_src, 'bo', ms=3, label = 'x-axis')
ax[0].set_ylim([0.1,5])
#plt.show()

ax[1].plot(hist_src2, 'r*', ms=3, label = 'Y-axis')
ax[1].set_ylim([0.1,5])
plt.show()

    
cv2.waitKey()    
cv2.destroyAllWindows()
