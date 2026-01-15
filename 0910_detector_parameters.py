# 0909_detector_parameters.py
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
    def __init__(self, extended=False, upright=False, threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=1):
        # KAZE 객체 생성
        self.kaze = cv2.KAZE_create(extended=extended, upright=upright, threshold=threshold, 
                                     nOctaves=nOctaves, nOctaveLayers=nOctaveLayers, diffusivity=1)

    def detectAndCompute(self, img):
        # 특징점 및 기술자 검출        
        # keypoints, descriptors = self.kaze.detectAndCompute(img, None)        
        kp = self.kaze.detect(img)
        kp = sorted(kp, key=lambda f: f.response, reverse=True)
        filtered_kp = filteringByDistance(kp, 10) #30
        keypoints, descriptors = self.kaze.compute(img, filtered_kp)
        
        return keypoints, descriptors

    def draw_keypoints(self, img, keypoints):
        # 특징점을 이미지에 그리기
        return cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def test_kaze_parameters(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 다양한 파라미터 조합으로 KAZE 테스트
    params = [
        {'extended': True, 'upright': False, 'threshold': 0.001, 'nOctaves': 4, 'nOctaveLayers': 4, 'diffusivity': 1},
        {'extended': False, 'upright': False, 'threshold': 0.005, 'nOctaves': 3, 'nOctaveLayers': 3, 'diffusivity': 1},
        {'extended': True, 'upright': True, 'threshold': 0.002, 'nOctaves': 5, 'nOctaveLayers': 5, 'diffusivity': 2}
    ]

    for param in params:
        kaze = KAZE(**param)
        keypoints, descriptors = kaze.detectAndCompute(img)
        img_kaze = kaze.draw_keypoints(img, keypoints)

        plt.figure(figsize=(10, 10))
        plt.imshow(img_kaze, cmap='gray')
        plt.title(f"KAZE Parameters: {param}")
        plt.axis('off')
        plt.show()

# 이미지 경로에 따라 테스트 함수 호출
# test_kaze_parameters('./data/chessBoard.jpg')
test_kaze_parameters('./data/CornerTest.jpg')
