# 0507.py
import cv2
import numpy as np

src = np.array([[0, 0, 0, 0],
                [1, 1, 3, 5],
                [6, 1, 1, 3],
                [4, 3, 1, 7]
              ], dtype=np.uint8)

#1
dst = cv2.equalizeHist(src)
print('dst =', dst)

#2
'''
ref: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
'''
##hist = cv2.calcHist(images = [src], channels = [0], mask = None,
##                    histSize = [256], ranges = [0, 256])
hist,bins = np.histogram(src.flatten(),10,[0,10])     #for analysis
# hist,bins = np.histogram(src.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_m = np.ma.masked_equal(cdf, 0) # cdf에서 0을 True 마스킹  
T = (cdf_m - cdf_m.min())*10/(cdf_m.max()-cdf_m.min()) #for analysis of the LookUpTable
# T = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
T = np.ma.filled(T, 0).astype('uint8') # 마스킹을 0으로 채우기 
img_equal = T[src] # img_equal == dst
print('img_equal =', img_equal)


#3 cf. nomalization
img_f = src.astype(np.float32)
img_norm = ((img_f - img_f.min()) * (10) / (img_f.max() - img_f.min()))
img_norm = img_norm.astype(np.uint8)
print('img_norm = ', img_norm)

#4 hist visualization
hist = cv2.calcHist([src], [0], None, [10], [0, 10])
hist_norm = cv2.calcHist([img_norm], [0], None, [10], [0, 10])
hist_equal = cv2.calcHist([img_equal], [0], None, [10], [0, 10])
print('src hist = ', hist)
print('nomalization hist = ', hist_norm)
print('equalization hist = ', hist_equal)
# cv2.imshow('img_src', src)
# cv2.imshow('img_normalization', img_norm)
# cv2.imshow('img_equalization', img_equal)

# import matplotlib.pylab as plt
from   matplotlib import pyplot as plt
hists = {'img_src' : hist, 'img_normalization':hist_norm, 'img_equalization':hist_equal}
color = {'b', 'g', 'r'}
# for i, (k, v) in enumerate(hists.items()):
#     # plt.subplot(1,3,i+1)
#     # plt.title(k)
#     # plt.plot(v)
#     plt.plot(binX, v, color='k')
#     # plt.bar(binX, v.flatten(), width=8, color=color[i])    
# plt.show()
binX = np.arange(10)
plt.title('img_src hist')
plt.bar(binX, hist.flatten(), width=1, color='b') 
plt.show()
plt.title('img_normalization hist')
plt.bar(binX, hist_norm.flatten(), width=1, color='b') 
plt.show()
plt.title('img_equalization hist')
plt.bar(binX, hist_equal.flatten(), width=1, color='b') 
plt.show()
# binX = np.arange(32)*8
# plt.plot(binX, hist1, color='r')
# plt.bar(binX, hist1, width=8, color='b')
# plt.show()