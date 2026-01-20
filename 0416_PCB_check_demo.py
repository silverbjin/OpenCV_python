import cv2
import numpy as np

# 원본 이미지 로딩 (물체가 있다고 가정)
# src = cv2.imread('./data/pcb_component.png')
src = cv2.imread('./data/pcb_component_test.png')
rows, cols = src.shape[:2]

# 회전/스케일 변환 정의 (실제 공정에서 센서로 측정되는 값에 해당)
angle = 18.2         # 부품이 18.2도 돌아가 있음
scale = 1/0.8        # 부품이 80% 축소된 상태
center = (cols//2, rows//2)

# Affine 변환 계산
M = cv2.getRotationMatrix2D(center, angle, scale)
dst = cv2.warpAffine(src, M, (cols, rows))

# 시각적 비교용 화면 구성
merged = np.hstack((src, dst))

# 중심 좌표 및 좌표계 표시
cv2.circle(merged, center, 5, (0,0,255), -1)
cv2.putText(merged, f"angle={angle}, scale={scale}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

cv2.imshow('Alignment Demo', merged)
cv2.waitKey()
cv2.destroyAllWindows()
