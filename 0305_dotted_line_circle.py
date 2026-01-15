import cv2
import numpy as np

# 두 점 설정
start_point = (50, 50)
end_point = (300, 300)
dot_spacing = 10  # 점 사이의 간격

# 빈 이미지 생성
image = np.ones((400, 400, 3), dtype=np.uint8) * 255

# 두 점 사이의 거리 및 방향 벡터 계산
distance = np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
num_dots = int(distance // dot_spacing)  # 필요한 점 개수
direction = ((end_point[0] - start_point[0]) / distance, (end_point[1] - start_point[1]) / distance)

# 점 그리기
for i in range(num_dots + 1):
    x = int(start_point[0] + direction[0] * i * dot_spacing)
    y = int(start_point[1] + direction[1] * i * dot_spacing)
    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

# 이미지 표시
cv2.imshow("Dotted Line", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
