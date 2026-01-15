# 0702_rev_2410_statistical.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 읽기 및 그레이스케일 변환
# img = cv2.imread('./data/rect.jpg')   #original
img = cv2.imread('./data/solidWhiteCurve.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 캐니 엣지 검출 적용
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 3. 확률적 허프 변환을 적용하여 직선 검출
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

# 4. 검출된 직선 그리기
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 결과 출력
plt.subplot(121), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Hough Line Detection'), plt.xticks([]), plt.yticks([])

plt.show()