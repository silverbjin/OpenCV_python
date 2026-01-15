# 0906_descriptor_128.py
import cv2
import numpy as np

# 이미지 불러오기 (특징점 검출에 사용할 이미지)
img = cv2.imread('./data/CornerTest.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('./data/chessBoard.jpg', cv2.IMREAD_GRAYSCALE)

# 1단계: 키포인트를 검출할 알고리즘 선택 (FAST를 사용)
# fast = cv2.FastFeatureDetector.create()
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(img, None)

# 2단계: 하나의 키포인트 주변에서 패치를 추출 (중심을 키포인트로 설정)
keypoint = keypoints[0]  # 첫 번째 키포인트 선택
patch_size = 31  # 패치 크기
x, y = int(keypoint.pt[0]), int(keypoint.pt[1])  # 키포인트의 좌표
patch = img[y - patch_size//2 : y + patch_size//2, x - patch_size//2 : x + patch_size//2]

# 3단계: 픽셀 쌍을 선택하고, BRIEF 디스크립터를 수동으로 생성
def generate_brief_descriptor(patch, num_bits=128):
    height, width = patch.shape
    descriptor = []

    # num_bits만큼 랜덤한 픽셀 쌍 선택
    for _ in range(num_bits):
        # 패치 내에서 두 픽셀 좌표를 무작위로 선택
        p1 = (np.random.randint(0, height), np.random.randint(0, width))
        p2 = (np.random.randint(0, height), np.random.randint(0, width))

        # 두 픽셀의 밝기 비교
        if patch[p1] < patch[p2]:
            descriptor.append(0)
        else:
            descriptor.append(1)

    return np.array(descriptor, dtype=np.uint8)

# 4단계: BRIEF 디스크립터 생성
brief_descriptor = generate_brief_descriptor(patch)

# 5단계: 디스크립터 출력
print("Generated BRIEF Descriptor (128-bit):")
print(brief_descriptor)
