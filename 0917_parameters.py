# 0917_parameters.py
import cv2

# HOGDescriptor 객체 생성 및 사람 검출기 설정
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 테스트할 이미지 불러오기
# image = cv2.imread('./data/people1.png')
image = cv2.imread('./data/people.png')

# HOG 검출기 파라미터 설정
winStride_values = [(4, 4), (8, 8), (16, 16)]  # 작은 값일수록 더 촘촘하게 탐색
padding_values = [(4, 4), (8, 8), (16, 16)]    # 경계 근처에서의 패딩
scale_values = [1.05, 1.2, 1.5]                # 이미지 스케일 피라미드 비율

# 다양한 파라미터 조합으로 사람 검출
for winStride in winStride_values:
    for padding in padding_values:
        for scale in scale_values:
            # 사람 검출
            (rects, weights) = hog.detectMultiScale(image, winStride=winStride, 
                                                    padding=padding, 
                                                    scale=scale)
            
            # 결과 이미지 복사본 생성
            output_image = image.copy()

            # 검출된 사람을 사각형으로 그리기
            for (x, y, w, h) in rects:
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 파라미터 출력 및 결과 이미지 표시
            print(f'winStride={winStride}, padding={padding}, scale={scale}, people detected={len(rects)}')
            cv2.imshow(f"People Detection (winStride={winStride}, padding={padding}, scale={scale})", output_image)
            cv2.waitKey(0)

cv2.destroyAllWindows()
