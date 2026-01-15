# 0910_descriptor.py
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


def display_keypoints_with_descriptors(img, keypoints, descriptors):
    # 특징점과 기술자를 이미지에 시각화
    img_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # 기술자 시각화 (간단한 방법)
    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if descriptors is not None:
            # 기술자의 첫 번째 값을 기준으로 원의 색상을 다르게 함
            color = (0, int(descriptors[i][0] * 255), 255 - int(descriptors[i][0] * 255))
            cv2.circle(img_keypoints, (x, y), 5, color, -1)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_keypoints, cmap='gray')
    plt.title('KAZE Keypoints and Descriptors')
    plt.axis('off')
    plt.show()

def kaze_feature_extraction(img_path):
    # 이미지를 불러오고 그레이스케일로 변환
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # KAZE 객체 생성
    kaze = cv2.KAZE_create()

    # 특징점 및 기술자 검출
    # keypoints, descriptors = kaze.detectAndCompute(img, None)    
    kp = kaze.detect(img)
    kp = sorted(kp, key=lambda f: f.response, reverse=True)
    filtered_kp = filteringByDistance(kp, 10) #30
    keypoints, descriptors = kaze.compute(img, filtered_kp)

    # 특징점과 기술자 시각화
    display_keypoints_with_descriptors(img, keypoints, descriptors)

    # 결과 출력
    print(f"검출된 특징점 수: {len(keypoints)}")
    if descriptors is not None:
        print(f"기술자 크기: {descriptors.shape}")

# 이미지 경로에 따라 함수 호출
# kaze_feature_extraction('./data/chessBoard.jpg')
kaze_feature_extraction('./data/CornerTest.jpg')