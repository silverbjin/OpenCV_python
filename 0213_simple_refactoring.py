# 0213.py
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cv2
import matplotlib.pyplot as plt


class SimpleAnimation:
    def __init__(self, interval=50):
        self.cap = cv2.VideoCapture(0)
        self.interval = interval

        fig = plt.figure(figsize=(10, 6))
        fig.canvas.set_window_title('Video Capture')
        plt.axis('off')

        # init 단계에서 이미지 객체 생성
        self.init(fig)

        # update 루프 실행
        self.run()

    def init(self, fig):
        retval, frame = self.cap.read()
        if retval:
            # 멤버 변수 self.im 로 저장 → global 필요 없음
            self.im = plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def update(self, k):
        retval, frame = self.cap.read()
        if retval:
            # 기존 im 객체에 픽셀만 갱신
            self.im.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def run(self):
        import time
        k = 0
        while True:
            self.update(k)
            k += 1
            time.sleep(self.interval / 1000)


# 실행
SimpleAnimation(interval=50)

