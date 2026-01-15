import cv2
import numpy as np

# 두 점 설정 hyper params.
start_point = (50, 50)
end_point = (300, 300)
dot_length = 5  # 점선의 각 점의 길이
dot_spacing = 10  # 점 사이의 간격

# 빈 이미지 생성
image = np.ones((400, 400, 3), dtype=np.uint8) * 255

# 두 점 사이의 거리 및 방향 벡터 계산 
distance = np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
num_dots = int(distance // (dot_length + dot_spacing))  # 필요한 점 개수
direction = ((end_point[0] - start_point[0]) / distance, (end_point[1] - start_point[1]) / distance)

# 점선 그리기
for i in range(num_dots + 1):
    # 시작 위치 계산
    start_x = int(start_point[0] + direction[0] * i * (dot_length + dot_spacing))
    start_y = int(start_point[1] + direction[1] * i * (dot_length + dot_spacing))
    
    # 끝 위치 계산
    end_x = int(start_x + direction[0] * dot_length)
    end_y = int(start_y + direction[1] * dot_length)
    
    # 점선 그리기
    cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)

# 이미지 표시
cv2.imshow("Dotted Line using line function", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
