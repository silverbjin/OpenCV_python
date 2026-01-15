# 0714_application_class.py
import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        """
        KMeans 클래스 생성자
        :param n_clusters: 클러스터의 수 (K)
        :param max_iter: 최대 반복 횟수
        :param tol: 중심 값 변화에 대한 허용 오차
        :param random_state: 난수 시드 설정 (재현 가능성을 위해)
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None

    def fit(self, X):
        """
        KMeans 학습 메서드
        :param X: 입력 데이터 (n_samples, n_features)
        """
        # 1. 난수 시드 설정
        if self.random_state:
            np.random.seed(self.random_state)
        
        # 2. 무작위로 K개의 초기 중심점 선택
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]

        # 3. 클러스터링 알고리즘 반복 수행
        for i in range(self.max_iter):
            # 4. 각 샘플을 가장 가까운 클러스터에 할당
            labels = self.predict(X)

            # 5. 새로운 클러스터 중심 계산
            new_centers = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])

            # 6. 중심 값의 변화가 tol보다 작으면 알고리즘 종료
            if np.all(np.abs(new_centers - self.cluster_centers_) < self.tol):
                break

            # 7. 클러스터 중심 업데이트
            self.cluster_centers_ = new_centers

    def predict(self, X):
        """
        예측 메서드: 각 데이터 포인트가 속하는 클러스터 할당
        :param X: 입력 데이터 (n_samples, n_features)
        :return: 각 데이터 포인트가 속하는 클러스터의 레이블
        """
        # 각 데이터 포인트와 클러스터 중심 사이의 거리 계산
        distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
        
        # 가장 가까운 클러스터의 인덱스 반환
        return np.argmin(distances, axis=0)


# 샘플 데이터 생성
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 랜덤 데이터셋 생성 (3개의 클러스터)
X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

# KMeans 객체 생성 및 학습
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 예측 (각 데이터 포인트의 클러스터 할당)
labels = kmeans.predict(X)

# 결과 시각화
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title('K-means Clustering')
plt.show()
