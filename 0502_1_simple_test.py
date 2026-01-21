# 0502_1_simple_test
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 10x10 example image
src = np.array([
    [120,130,125,128,129,131,120,110,100, 90],
    [122,135,130,129,128,132,118,112,102, 92],
    [121,134,132,131,130,133,119,113,104, 94],
    [119,133,129,127,125,130,121,114,105, 96],
    [118,132,128,126,124,129,122,115,106, 97],
    [117,131,127,125,123,128,123,116,108, 98],
    [116,130,126,124,122,126,124,118,110, 99],
    [115,129,124,123,121,125,125,120,112,100],
    [114,128,123,122,120,124,126,122,115,105],
    [113,127,122,121,119,123,127,124,118,108],
], dtype=np.float32)

# Parameters
C = 5
block = 3  # 3x3

# Mean adaptive threshold
mean_thresh = np.zeros_like(src)
for y in range(src.shape[0]):
    for x in range(src.shape[1]):
        y0 = max(0, y-block//2)
        y1 = min(src.shape[0], y+block//2+1)
        x0 = max(0, x-block//2)
        x1 = min(src.shape[1], x+block//2+1)
        w = src[y0:y1, x0:x1]
        T = w.mean() - C
        mean_thresh[y,x] = 255 if src[y,x] > T else 0

# Gaussian adaptive threshold using gaussian filter
gauss = gaussian_filter(src, sigma=0.8)  # approximate gaussian window
gaussian_thresh = np.where(src > (gauss - C), 255, 0)

# Plotting
fig, ax = plt.subplots(1,3, figsize=(9,3))
ax[0].imshow(src, cmap='gray'); ax[0].set_title('src (10x10)')
ax[1].imshow(mean_thresh, cmap='gray'); ax[1].set_title('Mean')
ax[2].imshow(gaussian_thresh, cmap='gray'); ax[2].set_title('Gaussian')
for a in ax: a.axis('off')
plt.show()

src, mean_thresh, gaussian_thresh
