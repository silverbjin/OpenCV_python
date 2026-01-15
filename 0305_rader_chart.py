import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 데이터 설정
labels = ["외모", "성격", "학력", "자산", "직업", "집안"]
scores = [3, 5, 4, 3, 3, 2]
max_score = 5  # 최대 점수

# 기본 설정
size = 400  # 이미지 크기
center = (size // 2, size // 2)  # 중심점
radius = 150  # 그래프 반지름
num_vars = len(labels)  # 꼭지점 개수

# 빈 이미지 생성
image = np.ones((size, size, 3), dtype=np.uint8) * 255

# 꼭지점 좌표 계산
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
points = [
    (int(center[0] + radius * np.cos(angle)), int(center[1] + radius * np.sin(angle)))
    for angle in angles
]

# 스코어 분류를 위한 점선 원 그리기
for i in range(1, max_score + 1):
    level_radius = radius * i / max_score
    level_points = [
        (int(center[0] + level_radius * np.cos(angle)), int(center[1] + level_radius * np.sin(angle)))
        for angle in angles
    ]
    for j in range(num_vars):
        next_point = level_points[(j + 1) % num_vars]
        cv2.line(image, level_points[j], next_point, (200, 200, 200), 1, lineType=cv2.LINE_AA)

# 점선으로 각 범례의 0점에서 5점까지 연결
for i in range(num_vars):
    for j in range(1, max_score + 1):
        inner_point = (
            int(center[0] + (radius * j / max_score) * np.cos(angles[i])),
            int(center[1] + (radius * j / max_score) * np.sin(angles[i]))
        )
        if j > 1:
            cv2.line(image, prev_point, inner_point, (150, 150, 150), 1, lineType=cv2.LINE_8)            
        prev_point = inner_point

# 점수에 따라 선 연결
score_points = [
    (int(center[0] + (radius * score / max_score) * np.cos(angle)),
     int(center[1] + (radius * score / max_score) * np.sin(angle)))
    for score, angle in zip(scores, angles)
]

# 내부 음영 그리기 (투명도 0.5)
overlay = image.copy()
pts = np.array(score_points, np.int32)
cv2.fillPoly(overlay, [pts], (100, 150, 255))
image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

# 스코어를 선으로 연결하여 외곽선 그리기
for i in range(num_vars):
    next_point = score_points[(i + 1) % num_vars]
    cv2.line(image, score_points[i], next_point, (0, 100, 255), 2, lineType=cv2.LINE_AA)

# OpenCV 이미지를 PIL 이미지로 변환
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil_image)

# 한글 폰트 경로 설정 (예: 나눔고딕)
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font = ImageFont.truetype(font_path, 12)

# 한글 라벨 표시
for i, label in enumerate(labels):
    label_pos = (
        int(center[0] + (radius + 20) * np.cos(angles[i])),
        int(center[1] + (radius + 20) * np.sin(angles[i])),
    )
    draw.text(label_pos, label, font=font, fill=(0, 0, 0))

# PIL 이미지를 다시 OpenCV 이미지로 변환
image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# 이미지 표시
cv2.imshow("Radar Chart with Dashed Ranges", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
