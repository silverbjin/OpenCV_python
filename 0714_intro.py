# 0714_intro.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. 예제 데이터를 생성합니다.
# make_blobs: 군집을 시뮬레이션한 데이터를 생성하는 함수
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 2. KMeans 알고리즘을 적용합니다.
k = 4  # 클러스터 개수
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X)

# 3. 군집의 중심 좌표를 얻습니다.
centers = kmeans.cluster_centers_

# 4. 각 데이터 포인트의 클러스터 할당을 얻습니다.
y_kmeans = kmeans.predict(X)

# 5. 결과 시각화
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')  # 군집 중심
plt.title('K-means Clustering')
plt.show()
