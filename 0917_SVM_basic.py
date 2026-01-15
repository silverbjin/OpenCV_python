# 0917_SVM_basic.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# 데이터 생성 - 두 개의 클래스를 구분하는 샘플 데이터
X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=6)

# SVM 모델 생성 및 학습
model = SVC(kernel='linear')
model.fit(X, y)

# 초평면 그리기 위한 함수
def plot_hyperplane(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

    # 초평면의 경계선 그리기
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 초평면의 기울기와 절편 계산
    w = model.coef_[0]
    slope = -w[0] / w[1]
    intercept = -model.intercept_[0] / w[1]

    # 경계선 그리기
    xx = np.linspace(xlim[0], xlim[1])
    yy = slope * xx + intercept
    plt.plot(xx, yy, 'k-')

    # 서포트 벡터의 마진 선 그리기
    margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + slope ** 2) * margin
    yy_up = yy + np.sqrt(1 + slope ** 2) * margin
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                s=100, facecolors='none', edgecolors='k')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

# 학습된 SVM 모델에 대한 초평면 및 서포트 벡터 시각화
plot_hyperplane(X, y, model)
