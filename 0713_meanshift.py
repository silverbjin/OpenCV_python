# 0713_meanshift.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_shift_filtering(img, spatial_radius, color_radius, max_iter=10, epsilon=1):
    # 1. 이미지 크기 및 초기화
    rows, cols, channels = img.shape
    output_img = np.copy(img)
    
    # 2. 각 픽셀에 대해 Mean Shift 알고리즘 적용
    for r in range(rows):
        for c in range(cols):
            current_center = img[r, c].astype(np.float32)  # 픽셀의 초기 중심값
            
            for _ in range(max_iter):
                # 주변 픽셀 좌표 탐색 (spatial_radius 내)
                min_r = max(r - spatial_radius, 0)
                max_r = min(r + spatial_radius, rows - 1)
                min_c = max(c - spatial_radius, 0)
                max_c = min(c + spatial_radius, cols - 1)
                
                # 색상 차이와 공간 거리 기반으로 윈도우 안의 픽셀 계산
                window_pixels = []
                for i in range(min_r, max_r + 1):
                    for j in range(min_c, max_c + 1):
                        pixel = img[i, j].astype(np.float32)
                        color_diff = np.linalg.norm(pixel - current_center)
                        
                        if color_diff < color_radius:
                            window_pixels.append(pixel)
                
                # 새로운 중심 계산 (윈도우 내 평균값)
                if len(window_pixels) == 0:
                    break
                new_center = np.mean(window_pixels, axis=0)
                
                # 중심이 수렴했는지 확인
                if np.linalg.norm(new_center - current_center) < epsilon:
                    break
                
                current_center = new_center
            
            # 새로운 중심값을 출력 이미지에 할당
            output_img[r, c] = current_center
    
    return output_img.astype(np.uint8)

# 3. 이미지 로드
img = cv2.imread('./data/flower.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# random image for processing speed test
random_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

# 이미지 축소
# small_img = cv2.resize(random_img, (img_rgb.shape[1] // 4, img_rgb.shape[0] // 4))
small_img = cv2.resize(img_rgb, (img_rgb.shape[1] // 4, img_rgb.shape[0] // 4))

# 4. Mean Shift 필터링 적용
# spatial_radius = 20  # 공간 반경
# color_radius = 40    # 색상 반경
# filtered_img = mean_shift_filtering(img_rgb, spatial_radius, color_radius)

# 파라미터 조정
spatial_radius = 10  # 공간 반경을 줄여서 더 작은 영역에서 군집화
color_radius = 20    # 색상 반경도 줄여서 더 세밀한 군집화
epsilon = 5          # 수렴 조건을 더 크게 설정하여 빠르게 수렴하게
filtered_img = mean_shift_filtering(small_img, spatial_radius, color_radius, epsilon)

# 5. 결과 출력
plt.figure(figsize=(10, 5))

# 원본 이미지 출력
plt.subplot(121)
plt.imshow(img_rgb)
# plt.imshow(random_img)
plt.title('Original Image')
plt.axis('off')

# 필터링 적용 후 이미지 출력
plt.subplot(122)
plt.imshow(filtered_img)
plt.title('Mean Shift Filtered Image (Custom Implementation)')
plt.axis('off')

plt.tight_layout()
plt.show()
