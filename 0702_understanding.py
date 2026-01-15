# 0702_understanding.py
import numpy as np
import matplotlib.pyplot as plt

# 직교 좌표계에서의 직선을 정의합니다.
def plot_line(m, b, x_range=(-10, 10)):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = m * x + b
    plt.plot(x, y, label=f'y = {m}x + {b}')

# 극좌표계에서 허프 공간을 시각화합니다.
def plot_hough_line(x, y):
    thetas = np.deg2rad(np.arange(-90, 90))  # 각도 범위 -90도에서 90도까지
    rhos = x * np.cos(thetas) + y * np.sin(thetas)
    return thetas, rhos

# 직선 그리기
plt.figure(figsize=(12, 6))

# 1. 직교 좌표계에서 직선을 그립니다.
plt.subplot(121)
plt.title('Cartesian Coordinate (y = mx + b)')
plot_line(1, 0)  # 기울기 1, 절편 0인 직선
plot_line(-0.5, 2)  # 기울기 -0.5, 절편 2인 직선
plot_line(0, 1)  # 수평선 y = 1
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True)
plt.legend()

# 2. 허프 변환을 적용하여 극좌표계에서 직선을 표현합니다.
plt.subplot(122)
plt.title('Hough Space (ρ = x cos(θ) + y sin(θ))')
for x, y in [(2, 2), (3, -1), (4, 0)]:  # 몇 개의 점을 예시로 들겠습니다.
    thetas, rhos = plot_hough_line(x, y)
    plt.plot(np.rad2deg(thetas), rhos, label=f'Point ({x},{y})')

plt.xlabel('Theta (degrees)')
plt.ylabel('Rho (distance)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
