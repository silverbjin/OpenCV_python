# 0701_rev_Non-Max_Sup.py
import cv2
import numpy as np

def non_max_suppression(gradient_magnitude, gradient_direction):
    # 그레디언트 크기와 동일한 크기의 배열을 0으로 초기화
    rows, cols = gradient_magnitude.shape
    suppressed_img = np.zeros((rows, cols), dtype=np.int32)

    # 그레디언트 방향을 0, 45, 90, 135도로 정규화
    angle = gradient_direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            q = 255
            r = 255

            # 그레디언트 방향에 따라 인접 픽셀 선택
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]

            # 현재 픽셀 값이 인접한 두 픽셀보다 크면 유지
            if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                suppressed_img[i, j] = gradient_magnitude[i, j]
            else:
                suppressed_img[i, j] = 0

    return suppressed_img

# gray scale 이미지 읽기
img = cv2.imread('./data/lena.jpg', cv2.IMREAD_GRAYSCALE)

# Gaussian 블러링으로 노이즈 제거
blurred_img = cv2.GaussianBlur(img, (5, 5), 1.4)

# Sobel 필터를 사용하여 그레디언트 계산
sobelx = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)  # X 방향 그레디언트
sobely = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)  # Y 방향 그레디언트

# 그레디언트 크기와 방향 계산
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
gradient_direction = np.arctan2(sobely, sobelx)

# 비최대 억제 적용
suppressed_img = non_max_suppression(gradient_magnitude, gradient_direction)

# 결과 출력
cv2.imshow('Original', img)
cv2.imshow('Suppressed Image', suppressed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
