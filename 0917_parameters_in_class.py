# 0917_parameters_in_class.py
import numpy as np
import cv2

class CustomHOGDescriptor:
    def __init__(self, winStride=(8, 8), padding=(8, 8), scale=1.05):
        # HOGDescriptor 기본 설정
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.winStride = winStride
        self.padding = padding
        self.scale = scale

    def detect_people(self, image):
        """
        주어진 이미지에서 사람을 검출합니다.
        """
        # 이미지 피라미드와 윈도우 탐색을 이용한 사람 검출
        (rects, weights) = self.hog.detectMultiScale(image, 
                                                     winStride=self.winStride, 
                                                     padding=self.padding, 
                                                     scale=self.scale)
        return rects, weights

    def apply_padding(self, image):
        """
        패딩을 적용한 이미지를 반환합니다.
        """
        return cv2.copyMakeBorder(image, self.padding[1], self.padding[1], self.padding[0], self.padding[0], cv2.BORDER_CONSTANT)

    def process_pyramid(self, image):
        """
        이미지 피라미드와 스케일 변환을 적용합니다.
        """
        original_shape = image.shape[:2]
        while min(original_shape) >= self.scale * min(original_shape):
            yield image
            # 이미지 크기를 스케일 비율에 맞춰 축소
            image = cv2.resize(image, (int(image.shape[1] / self.scale), int(image.shape[0] / self.scale)))

# 테스트 코드
if __name__ == "__main__":
    # 테스트 이미지 불러오기
    # image = cv2.imread('./data/people1.png')
    image = cv2.imread('./data/people.png')
    
    # Custom HOGDescriptor 객체 생성
    custom_hog = CustomHOGDescriptor(winStride=(8, 8), padding=(8, 8), scale=1.05)

    # 패딩이 적용된 이미지 출력
    padded_image = custom_hog.apply_padding(image)
    cv2.imshow('Padded Image', padded_image)

    # 피라미드 변환을 확인하면서 출력
    for idx, img in enumerate(custom_hog.process_pyramid(image)):
        cv2.imshow(f'Scaled Image {idx}', img)
        cv2.waitKey(500)

    # 사람 검출
    rects, weights = custom_hog.detect_people(image)

    # 검출된 사람을 그리기
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow("Detected People", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
