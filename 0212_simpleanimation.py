# 0212.py
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class SimpleAnimation:
    def __init__(self, fig, init_func, update_func, interval=50):
        self.fig = fig
        self.init = init_func
        self.update = update_func
        self.interval = interval
        
        # 초기화 함수 1회 실행
        self.init()
        
        # 영상 재생 루프
        self.run()

    def run(self):
        import time
        k = 0
        while True:
            self.update(k)
            k += 1
            time.sleep(self.interval / 1000)

def init():
    global im
    retval, frame = cap.read()
    im = plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def updateFrame(k): 
    retval, frame = cap.read()
    if retval:
        im.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# 프로그램 시작    
cap = cv2.VideoCapture(0)
fig = plt.figure(figsize=(10, 6)) # fig.set_size_inches(10, 6)
fig.canvas.set_window_title('Video Capture')
plt.axis('off')

def init():
    global im
    retval, frame = cap.read() # 첫 프레임 캡처
    im = plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
##    return im,

def updateFrame(k): 
    retval, frame = cap.read()
    if retval:
        im.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# ani = animation.FuncAnimation(fig, updateFrame, init_func=init, interval=50)
ani = SimpleAnimation(fig, updateFrame, init_func=init, interval=50)

plt.show()
if cap.isOpened():
    cap.release()
