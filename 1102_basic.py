# 1102_basic.py
# np.mgrid[0:xN, 0:yN].T.reshape(-1, 2) 

import numpy as np
import matplotlib.pylab as plt

# 1 ~ 6 까지 6개로, 11 ~ 13까지 3개로 → meshgrid로 인해 Broadcasting된 후 각각 (3, 6) array
## (가로, 세로)
x = np.linspace(1,6,6)
y = np.linspace(11,13,3)
X,Y = np.meshgrid(x,y)
print('x: \n', x)
print('y: \n', y)
print('X: \n', X)
print('Y: \n', Y)

xN, yN = (6, 3) # (6, 3)
mW = np.zeros((xN*yN, 3)) # (18, 3)
#code analysis
temp1 = np.mgrid[0:xN, 0:yN]
temp2 = np.mgrid[0:xN, 0:yN].T
temp3 = np.mgrid[0:xN, 0:yN].T.reshape(-1, 2)
print('temp1: \n', temp1)
print('temp2: \n', temp2)
print('temp3: \n', temp3)

plt.scatter(X,Y)
plt.scatter(temp1[0],temp1[1])
plt.grid()
    
plt.show()

# ref. numpy mgrid 의미
# https://namyoungkim.github.io/python/numpy/2017/10/01/python_1/
# https://seong6496.tistory.com/129
# https://tbr74.tistory.com/entry/numpy-meshgrid-%ED%95%A8%EC%88%98-%EC%95%8C%EC%95%84%EB%B3%B4%EA%B8%B0
