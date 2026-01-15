# 0917_SVMDetector.py
import cv2
import numpy as np

# HOGDescriptor 객체 생성
hog = cv2.HOGDescriptor()

# SVM 학습을 위한 샘플 데이터 생성 (여기서는 가상의 데이터 사용)
# X는 HOG 특징 벡터, y는 해당 데이터가 사람인지 여부 (1: 사람, 0: 사람 아님)
X = np.random.rand(10, 3780).astype(np.float32)  # 3780은 기본 HOG 벡터 크기
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)

# SVM 분류기 설정 및 학습
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(0.01)
svm.train(X, cv2.ml.ROW_SAMPLE, y)

# 학습된 SVM 모델을 이용하여 HOGDescriptor에 적용
svm_detector = svm.getSupportVectors().reshape(-1).astype(np.float32)
hog.setSVMDetector(svm_detector)

# 테스트할 이미지 불러오기
# image = cv2.imread('./data/people1.png')
image = cv2.imread('./data/people.png')

# 이미지에서 사람을 검출
(rects, weights) = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)

# 검출된 사람에 사각형 그리기
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 결과 이미지 출력
cv2.imshow("Detected People", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
