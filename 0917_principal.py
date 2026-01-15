# 0917_principal.py
import cv2

# HOGDescriptor 객체 생성 및 사전 학습된 사람 검출기 설정
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 이미지 불러오기
# image = cv2.imread('./data/people1.png')
image = cv2.imread('./data/people.png')

# 사람 검출
(rects, weights) = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)

# 검출된 객체에 사각형 그리기
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 결과 출력
cv2.imshow("Detected People", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
