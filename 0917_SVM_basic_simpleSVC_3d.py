# 0917_SVM_basic_simpleSVC_3d.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SimpleSVC:
    def __init__(self, C=1.0, kernel='linear'):
        self.C = C
        self.kernel = kernel
        self.support_vectors_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        if self.kernel == 'linear':
            n_samples, n_features = X.shape
            # 간단하게 의사역행렬을 사용해 계수 구함
            self.coef_ = np.dot(np.linalg.pinv(X), y)
            self.intercept_ = np.mean(y - np.dot(X, self.coef_))

            # 서포트 벡터: 결정 함수 값이 마진 내에 있는 벡터들
            margin = 1 / np.sqrt(np.sum(self.coef_ ** 2))
            decision_values = np.dot(X, self.coef_) + self.intercept_
            self.support_vectors_ = X[np.abs(decision_values) <= margin]

    def predict(self, X):
        return np.sign(np.dot(X, self.coef_) + self.intercept_)

# 3차원 데이터 시각화 함수
def plot_svc_decision_boundary_3d(model, X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 데이터 포인트 그리기
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='coolwarm', s=50)

    # 초평면을 그리기 위해 그리드 생성
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # 그리드 좌표 계산
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    
    # 초평면 계산
    coef = model.coef_
    intercept = model.intercept_
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

    # 초평면 그리기
    ax.plot_surface(xx, yy, zz, alpha=0.5, color='gray')

    # 서포트 벡터 표시
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
               model.support_vectors_[:, 2], s=100, facecolors='none', edgecolors='k')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    plt.show()

# 3차원 데이터 생성
np.random.seed(0)
X = np.random.randn(20, 3)
y = np.array([1] * 10 + [-1] * 10)

# 데이터 조정 (간단한 선형 분리를 만들기 위한 조작)
X[:10, :] += 1
X[10:, :] -= 1

# SimpleSVC 학습
svc = SimpleSVC(C=1.0, kernel='linear')
svc.fit(X, y)

# 시각화
plot_svc_decision_boundary_3d(svc, X, y)
