# 0713_other.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 읽기
img = cv2.imread('./data/flower.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. pyrMeanShiftFiltering 적용
# sp: 공간 윈도우 반경, sr: 색상 윈도우 반경
spatial_radius = 20  # 공간 반경 (이 값을 높이면 더 넓은 영역에서의 필터링)
color_radius = 30  # 색상 반경 (이 값을 높이면 색상 차이가 큰 경우에만 그룹화)
mean_shift_img = cv2.pyrMeanShiftFiltering(img, sp=spatial_radius, sr=color_radius)

# 3. 결과를 출력
plt.figure(figsize=(10, 5))

# 원본 이미지 출력
plt.subplot(121)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# Mean Shift 필터링 적용 후 이미지 출력
mean_shift_img_rgb = cv2.cvtColor(mean_shift_img, cv2.COLOR_BGR2RGB)
plt.subplot(122)
plt.imshow(mean_shift_img_rgb)
plt.title('Mean Shift Filtered Image')
plt.axis('off')

plt.tight_layout()
plt.show()
