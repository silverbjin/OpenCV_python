# 0910_principal.py
import cv2
import matplotlib.pyplot as plt

# 이미지를 읽어오기 (흑백 이미지로)
img = cv2.imread('./data/CornerTest.jpg', cv2.IMREAD_GRAYSCALE)

# KAZE 객체 생성
kaze = cv2.KAZE_create()

# 특징점과 기술자 검출
keypoints, descriptors = kaze.detectAndCompute(img, None)

# 특징점 그리기
img_kaze = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 결과 시각화
plt.figure(figsize=(10, 10))
plt.imshow(img_kaze, cmap='gray')
plt.title('KAZE KeyPoints')
plt.show()
