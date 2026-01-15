# 0909_detector_implement.py
import numpy as np
import cv2
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

class KAZE:
    def __init__(self):
        # 초기 설정값 (비선형 확산 관련 파라미터 등)
        self.octaves = 4  # 예시: 옥타브 수
        self.threshold = 0.001  # 예시: 검출 임계값
        self.descriptor_size = 64  # 예시: 기술자 벡터 크기
        # 실제 OpenCV는 더 많은 매개변수를 사용함

    def nonlinear_scale_space(self, img):
        # 비선형 스케일 공간 생성 함수 (단순화된 형태)
        # 실제 KAZE에서는 복잡한 확산 방정식을 사용하나,
        # 여기서는 간단하게 Gaussian blur로 예시를 대체함
        scales = []
        for i in range(self.octaves):
            sigma = 1.6 * (2 ** i)  # 스케일 별 sigma 값 변화
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
            scales.append(blurred)
        return scales

    def detect_keypoints(self, scales):
        # 스케일 공간에서 특징점을 검출하는 함수 (단순화)
        keypoints = []
        for i, scale in enumerate(scales):
            # 특징점 검출 (단순한 방법으로 예시 구현)
            fast = cv2.FastFeatureDetector_create()
            kp = fast.detect(scale, None)            
            keypoints.extend(kp)
            keypoints = sorted(keypoints, key=lambda f: f.response, reverse=True)
        # return keypoints
        return filteringByDistance(keypoints, 10)

    def compute_descriptors(self, img, keypoints):
        # 특징점에 대한 기술자 생성 (단순화된 예시)
        # 실제 KAZE는 비선형 확산에 기반한 독특한 기술자를 사용함
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(self.descriptor_size)
        keypoints, descriptors = brief.compute(img, keypoints)
        return keypoints, descriptors

    def detectAndCompute(self, img, mask=None):
        # 전체 파이프라인
        scales = self.nonlinear_scale_space(img)
        keypoints = self.detect_keypoints(scales)
        keypoints, descriptors = self.compute_descriptors(img, keypoints)
        return keypoints, descriptors


# KAZE 객체 생성
kaze = KAZE()

# 이미지 불러오기
# img = cv2.imread('./data/chessBoard.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('./data/CornerTest.jpg', cv2.IMREAD_GRAYSCALE)

# 특징점과 기술자 검출
keypoints, descriptors = kaze.detectAndCompute(img)

# 결과 출력
print(f"검출된 특징점 수: {len(keypoints)}")
print(f"기술자 크기: {descriptors.shape}")

# 특징점 그리기
img_kaze = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 결과 시각화
plt.figure(figsize=(10, 10))
plt.imshow(img_kaze, cmap='gray')
plt.title('KAZE keypoints')
plt.show()