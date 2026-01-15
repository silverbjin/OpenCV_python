# 0911_detector_parameter.py
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
    
#2
##siftF = cv2.xfeatures2d.SIFT_create()
# siftF = cv2.xfeatures2d.SIFT_create(edgeThreshold = 80)
# kp= siftF.detect(gray)
# print('len(kp)=', len(kp))

# 이미지 로드
img = cv2.imread('./data/CornerTest.jpg', cv2.IMREAD_GRAYSCALE)

# SIFT 생성 함수
def create_sift(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
    return cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures, 
                           nOctaveLayers=nOctaveLayers, 
                           contrastThreshold=contrastThreshold, 
                           edgeThreshold=edgeThreshold, 
                           sigma=sigma)

# 파라미터 테스트 함수
def test_sift_params(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma):
    # SIFT 객체 생성
    sift = create_sift(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
    
    # 특징점과 디스크립터 추출
    # keypoints, descriptors = sift.detectAndCompute(img, None)
    kp = sift.detect(img)
    kp = sorted(kp, key=lambda f: f.response, reverse=True)   
    filtered_kp = filteringByDistance(kp, 10)    
    keypoints, descriptors = sift.compute(img, filtered_kp)
    
    # 특징점 그리기
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)
    
    return img_with_keypoints, len(keypoints)

# 파라미터 설정 및 테스트
test_cases = [
    {'nfeatures': 0, 'nOctaveLayers': 3, 'contrastThreshold': 0.04, 'edgeThreshold': 10, 'sigma': 1.6},
    {'nfeatures': 100, 'nOctaveLayers': 3, 'contrastThreshold': 0.04, 'edgeThreshold': 10, 'sigma': 1.6},
    {'nfeatures': 0, 'nOctaveLayers': 5, 'contrastThreshold': 0.04, 'edgeThreshold': 10, 'sigma': 1.6},
    {'nfeatures': 0, 'nOctaveLayers': 3, 'contrastThreshold': 0.1, 'edgeThreshold': 10, 'sigma': 1.6},
    {'nfeatures': 0, 'nOctaveLayers': 3, 'contrastThreshold': 0.04, 'edgeThreshold': 5, 'sigma': 1.6},
    {'nfeatures': 0, 'nOctaveLayers': 3, 'contrastThreshold': 0.04, 'edgeThreshold': 10, 'sigma': 2.0},
]

# 결과 시각화
plt.figure(figsize=(15, 10))

for i, params in enumerate(test_cases):
    img_with_keypoints, num_keypoints = test_sift_params(**params)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title(f"nfeatures={params['nfeatures']}, nOctaveLayers={params['nOctaveLayers']}\n"
              f"contrastThreshold={params['contrastThreshold']}, edgeThreshold={params['edgeThreshold']}, sigma={params['sigma']}\n"
              f"Keypoints: {num_keypoints}")
    plt.axis('off')

plt.tight_layout()
plt.show()
