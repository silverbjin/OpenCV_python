# 0911_principal.py
import cv2

# SIFT 객체 생성
sift = siftF = cv2.xfeatures2d.SIFT_create()

# 이미지 로드
img = cv2.imread('./data/CornerTest.jpg', cv2.IMREAD_GRAYSCALE)

# 특징점과 디스크립터 추출
keypoints, descriptors = sift.detectAndCompute(img, None)

# 특징점 그리기
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)

cv2.imshow('SIFT Keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
