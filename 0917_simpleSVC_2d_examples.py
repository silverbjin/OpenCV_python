# 0917_simpleSVC_2d_examples.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

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
            self.coef_ = np.dot(np.linalg.pinv(X), y)
            self.intercept_ = np.mean(y - np.dot(X, self.coef_))
            margin = 1 / np.sqrt(np.sum(self.coef_ ** 2))
            decision_values = np.dot(X, self.coef_) + self.intercept_
            self.support_vectors_ = X[np.abs(decision_values) <= margin]

    def predict(self, X):
        return np.sign(np.dot(X, self.coef_) + self.intercept_)

# 2D 데이터 시각화 함수
def plot_svc_decision_boundary(model, X, y, title, xlabel, ylabel):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', s=50)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    w = model.coef_
    slope = -w[0] / w[1]
    intercept = -model.intercept_ / w[1]
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = slope * xx + intercept
    plt.plot(xx, yy, 'k-')
    margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + slope ** 2) * margin
    yy_up = yy + np.sqrt(1 + slope ** 2) * margin
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# 1. 학생 성적 예측 (Attendance vs Homework Submission)
X1 = np.array([[0.9, 0.8], [0.7, 0.6], [0.85, 0.9], [0.5, 0.4], [0.3, 0.5], [0.6, 0.3]])
y1 = np.array([1, 1, 1, -1, -1, -1])

svc1 = SimpleSVC(C=1.0, kernel='linear')
svc1.fit(X1, y1)
plot_svc_decision_boundary(svc1, X1, y1, 
                           title='Grade Prediction (Pass vs Fail)', 
                           xlabel='Attendance Rate', 
                           ylabel='Homework Submission Rate')

# 2. 학생 그룹 분류 (Academic Performance vs Study Hours)
X2 = np.array([[85, 10], [90, 12], [80, 9], [60, 4], [55, 5], [50, 3]])
y2 = np.array([1, 1, 1, -1, -1, -1])

svc2 = SimpleSVC(C=1.0, kernel='linear')
svc2.fit(X2, y2)
plot_svc_decision_boundary(svc2, X2, y2, 
                           title='Student Group Classification (High vs Low Achievers)', 
                           xlabel='Academic Performance (Scores)', 
                           ylabel='Study Hours per Week')

# 3. 동아리 참여와 성적 분류 (Club Participation vs Academic Performance)
X3 = np.array([[9, 85], [8, 90], [10, 88], [4, 60], [5, 55], [3, 50]])
y3 = np.array([1, 1, 1, -1, -1, -1])

svc3 = SimpleSVC(C=1.0, kernel='linear')
svc3.fit(X3, y3)
plot_svc_decision_boundary(svc3, X3, y3, 
                           title='Club Participation vs Academic Success', 
                           xlabel='Club Participation (Hours)', 
                           ylabel='Academic Performance (Scores)')
