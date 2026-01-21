# 0502_2_application_ocr
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Generate synthetic OCR-like image
img = np.ones((200, 600), dtype=np.uint8) * 255
cv2.putText(img, "Adaptive Thresholding", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,), 3, cv2.LINE_AA)
cv2.putText(img, "Mean vs Gaussian", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,), 3, cv2.LINE_AA)

# Add uneven illumination
gradient = np.tile(np.linspace(50, 255, img.shape[1], dtype=np.uint8), (img.shape[0], 1))
src = cv2.addWeighted(img, 0.7, gradient, 0.3, 0)

# Apply adaptive thresholds
dst_mean = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 51, 7)
dst_gaussian = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, 7)

# Plot results
plt.figure(figsize=(12,6))
plt.subplot(1,3,1); plt.title('Source'); plt.imshow(src, cmap='gray'); plt.axis('off')
plt.subplot(1,3,2); plt.title('Mean'); plt.imshow(dst_mean, cmap='gray'); plt.axis('off')
plt.subplot(1,3,3); plt.title('Gaussian'); plt.imshow(dst_gaussian, cmap='gray'); plt.axis('off')
plt.show()
