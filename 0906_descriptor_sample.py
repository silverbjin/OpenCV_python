# 0906_descriptor_sample.py
import cv2
import numpy as np

# 이미지 불러오기 (특징점 검출에 사용할 이미지)
img = cv2.imread('./data/CornerTest.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('./data/chessBoard.jpg', cv2.IMREAD_GRAYSCALE)

# 1단계: 키포인트를 검출할 알고리즘 선택 (예: FAST)
fast = cv2.FastFeatureDetector_create()

# 2단계: FAST를 사용해 키포인트 검출
keypoints = fast.detect(img, None)

# 3단계: BRIEF 디스크립터 생성기 초기화
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# brief = cv2.ORB_create()

# 4단계: 검출된 키포인트로부터 BRIEF 디스크립터 계산
keypoints, descriptors = brief.compute(img, keypoints)

# 5단계: 결과 출력 (키포인트 수와 디스크립터)
print(f"Number of keypoints: {len(keypoints)}")
print(f"Descriptor shape: {descriptors.shape}")
print(f"Descriptor: {descriptors}")
# 디스크립터는 이진 형태이므로, 각 디스크립터는 128 또는 256 비트 크기의 이진 문자열을 나타냄
