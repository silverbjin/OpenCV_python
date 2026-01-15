# 0712_ref.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 읽기
img = cv2.imread('./data/lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Gaussian Pyramid 생성 (이미지 축소)
layer = img.copy()
gaussian_pyramid = [layer]

for i in range(6):  # 6단계로 피라미드를 축소합니다.
    layer = cv2.pyrDown(layer)  # 이미지를 축소
    gaussian_pyramid.append(layer)

# 3. Laplacian Pyramid 생성 (Gaussian Pyramid 간의 차이)
laplacian_pyramid = []

for i in range(5, 0, -1):  # Gaussian Pyramid 역순으로 진행
    gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])  # 이미지를 확대
    laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)  # 차이 계산
    laplacian_pyramid.append(laplacian)

# 4. 결과 시각화 (Gaussian Pyramid)
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 6, i+1)
    plt.imshow(gaussian_pyramid[i])
    plt.title(f'Gaussian Level {i}')
    plt.axis('off')

# 5. 결과 시각화 (Laplacian Pyramid)
for i in range(5):
    plt.subplot(2, 6, i+7)
    plt.imshow(np.clip(laplacian_pyramid[i], 0, 255))
    plt.title(f'Laplacian Level {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()
