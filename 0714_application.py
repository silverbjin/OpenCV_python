# 0714_application.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. 이미지 읽기 (OpenCV는 BGR 형식으로 이미지를 불러옵니다)
# img = cv2.imread('./data/hand.jpg')
img = cv2.imread('./data/flower.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB 형식으로 변환

# 2. 이미지 데이터를 2차원 배열로 변환 (각 픽셀은 3개의 값: R, G, B를 가집니다)
pixel_values = img_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)  # float32 형식으로 변환

# 3. KMeans 알고리즘 적용
k = 4  # K개의 색상 클러스터로 나눔
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixel_values)

# 4. 각 픽셀의 클러스터 할당 (predict 사용)
labels = kmeans.predict(pixel_values)

# 5. 클러스터의 중심을 기반으로 새로운 이미지 생성
centers = np.uint8(kmeans.cluster_centers_)  # 각 클러스터의 중심 색상
segmented_image = centers[labels]  # 각 픽셀을 클러스터의 중심 값으로 대체
segmented_image = segmented_image.reshape(img_rgb.shape)  # 원본 이미지와 같은 형식으로 변환

# 6. 결과 시각화
plt.figure(figsize=(10, 5))

# 원본 이미지 출력
plt.subplot(121)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# K-means로 압축된 이미지 출력
plt.subplot(122)
plt.imshow(segmented_image)
plt.title(f'K-means Image (K={k})')
plt.axis('off')

plt.tight_layout()
plt.show()
