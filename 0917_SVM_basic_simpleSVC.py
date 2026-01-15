# 0917_SVM_basic_simpleSVC.py
import numpy as np

class SimpleSVC:
    def __init__(self, C=1.0, kernel='linear'):
        self.C = C
        self.kernel = kernel
        self.support_vectors_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        SVM 모델을 학습합니다.
        X: 입력 데이터 (n_samples, n_features)
        y: 레이블 (-1 또는 1로 구성된 벡터)
        """
        # 선형 커널만 처리하도록 구현 (실제 커널은 다룰 수 없음)
        if self.kernel == 'linear':
            # 간단하게 선형 SVM을 학습하는 과정 구현 (여기서는 선형 분리를 가정)
            n_samples, n_features = X.shape

            # 가상의 방법으로 계수(coef_)와 절편(intercept_) 결정
            # 실제로는 쌍대형 문제를 풀어야 하지만, 여기에선 매우 단순한 모델을 사용
            self.coef_ = np.dot(np.linalg.pinv(X), y)  # 의사역행렬을 사용해 계수 구함
            self.intercept_ = np.mean(y - np.dot(X, self.coef_))  # 평균 차이로 절편 구함

            # 서포트 벡터 (학습된 초평면에 가까운 벡터, 여기선 임의로 설정)
            margin = 1 / np.sqrt(np.sum(self.coef_ ** 2))
            decision_values = np.dot(X, self.coef_) + self.intercept_
            self.support_vectors_ = X[np.abs(decision_values) <= margin]

    def predict(self, X):
        """
        학습된 모델을 사용하여 입력 데이터 X를 분류합니다.
        X: 입력 데이터 (n_samples, n_features)
        """
        return np.sign(np.dot(X, self.coef_) + self.intercept_)

# 샘플 데이터
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
y = np.array([1, 1, 1, -1, -1])

# SVC 클래스 사용 예
svc = SimpleSVC(C=1.0, kernel='linear')
svc.fit(X, y)

# 예측
print("계수 (coef_):", svc.coef_)
print("절편 (intercept_):", svc.intercept_)
print("서포트 벡터 (support_vectors_):", svc.support_vectors_)

# 새 데이터로 예측
X_new = np.array([[3, 3], [1, 1]])
print("예측 결과:", svc.predict(X_new))
