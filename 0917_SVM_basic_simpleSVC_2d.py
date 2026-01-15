# 0917_SVM_basic_simpleSVC_2d.py
import numpy as np
import matplotlib.pyplot as plt

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

# SVM 모델 시각화 함수
def plot_svc_decision_boundary(model, X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', s=50)

    # 초평면 그리기
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 초평면의 기울기 및 절편 계산
    w = model.coef_
    slope = -w[0] / w[1]
    intercept = -model.intercept_ / w[1]

    # 경계선 그리기
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = slope * xx + intercept
    plt.plot(xx, yy, 'k-')

    # 서포트 벡터 마진 계산 및 그리기
    margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + slope ** 2) * margin
    yy_up = yy + np.sqrt(1 + slope ** 2) * margin

    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    # 서포트 벡터 표시
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k')

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

# 데이터 생성
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
y = np.array([1, 1, 1, -1, -1])

# SimpleSVC 클래스 사용
svc = SimpleSVC(C=1.0, kernel='linear')
svc.fit(X, y)

# 시각화
plot_svc_decision_boundary(svc, X, y)
