# 0911_descriptor.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1
def distance(f1, f2):    
    x1, y1 = f1.pt
    x2, y2 = f2.pt
    return np.sqrt((x2 - x1)**2+ (y2 - y1)**2)

def filteringByDistance(kp, distE=0.5):
    size = len(kp)
    mask = np.arange(1,size+1).astype(np.bool8) # all True   
    for i, f1 in enumerate(kp):
        if not mask[i]:
            continue
        else: # True
            for j, f2 in enumerate(kp):
                if i == j:
                    continue
                if distance(f1, f2)<distE:
                    mask[j] = False
    np_kp = np.array(kp)
    return list(np_kp[mask])
  

# 이미지 로드 (흑백으로 로드)
img = cv2.imread('./data/CornerTest.jpg', cv2.IMREAD_GRAYSCALE)

# SIFT 객체 생성
sift = cv2.xfeatures2d.SIFT_create()

# 특징점과 디스크립터 추출
# keypoints, descriptors = sift.detectAndCompute(img, None)
kp = sift.detect(img)
kp = sorted(kp, key=lambda f: f.response, reverse=True)   
filtered_kp = filteringByDistance(kp, 10)    
keypoints, descriptors = sift.compute(img, filtered_kp)

# 첫 번째 특징점의 디스크립터 확인
first_keypoint = keypoints[0]
first_descriptor = descriptors[0]

print(f"첫 번째 특징점의 위치: {first_keypoint.pt}")
print(f"첫 번째 특징점의 크기: {first_keypoint.size}")
print(f"첫 번째 특징점의 각도: {first_keypoint.angle}")
print(f"첫 번째 디스크립터(128차원): \n{first_descriptor}")

# 특징점 그리기
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)

# 디스크립터 히스토그램 시각화 함수
def plot_descriptor_histogram(descriptor):
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(descriptor)), descriptor)
    plt.title('SIFT Descriptor Histogram')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.show()

# 첫 번째 디스크립터 시각화
plot_descriptor_histogram(first_descriptor)

# 특징점과 디스크립터가 그려진 이미지 출력
plt.figure(figsize=(10, 10))
plt.imshow(img_with_keypoints, cmap='gray')
plt.title('SIFT Keypoints')
plt.show()
