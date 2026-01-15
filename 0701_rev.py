# 0701_rev.py
import cv2
import numpy as np
from matplotlib import pyplot as plt

# gray scale 이미지 읽기
img = cv2.imread('./data/lena.jpg', cv2.IMREAD_GRAYSCALE)

# 1. 노이즈 제거 - Gaussian 블러링 적용
blurred_img = cv2.GaussianBlur(img, (5, 5), 1.4)

# 2. 그레디언트 계산 - Sobel 필터를 이용한 엣지 검출
sobelx = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)  # X축 방향
sobely = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)  # Y축 방향

# 그레디언트 크기 계산
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
gradient_direction = np.arctan2(sobely, sobelx)

# 3. 비최대 억제 - 그레디언트 방향에 따른 최대값 남기기 (이 부분은 캐니 함수가 내장함)

# 4, 5. 이중 임계값 적용 및 엣지 연결
edges = cv2.Canny(blurred_img, threshold1=100, threshold2=200)

# 결과 출력
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
