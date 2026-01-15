# 0714_application_implement.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(data, k):
    # 데이터에서 임의의 k개의 포인트를 선택하여 초기 중심으로 사용
    np.random.seed(42)
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    # 각 데이터 포인트와 각 중심점 사이의 거리를 계산하여 가장 가까운 클러스터에 할당
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(data, labels, k):
    # 각 클러스터에 속한 포인트들의 평균을 계산하여 새로운 중심을 갱신
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans_clustering(data, k, max_iter=100, tol=1e-4):
    # 1. 중심 초기화
    centroids = initialize_centroids(data, k)
    for i in range(max_iter):
        # 2. 클러스터 할당
        labels = assign_clusters(data, centroids)
        
        # 3. 중심 갱신
        new_centroids = update_centroids(data, labels, k)
        
        # 4. 중심이 거의 변하지 않으면 종료
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        
        centroids = new_centroids
    return labels, centroids

# 1. 이미지 읽기 (RGB 변환)
# img = cv2.imread('./data/hand.jpg')
img = cv2.imread('./data/flower.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. 이미지 데이터를 2차원 배열로 변환 (각 픽셀을 하나의 데이터로 간주)
pixel_values = img_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# 3. KMeans 클러스터링 적용
k = 4  # 4개의 색상 클러스터로 나눕니다.
labels, centers = kmeans_clustering(pixel_values, k)

# 4. 각 픽셀을 새로운 중심 값으로 변환하여 이미지 재구성
centers = np.uint8(centers)  # float32를 uint8로 변환
segmented_image = centers[labels]
segmented_image = segmented_image.reshape(img_rgb.shape)

# 5. 결과 시각화
plt.figure(figsize=(10, 5))

# 원본 이미지 출력
plt.subplot(121)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# K-means로 클러스터링된 이미지 출력
plt.subplot(122)
plt.imshow(segmented_image)
plt.title(f'K-means Image (K={k})')
plt.axis('off')

plt.tight_layout()
plt.show()
