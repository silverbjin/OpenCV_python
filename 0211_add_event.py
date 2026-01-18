# webcam_events.py
import cv2
import matplotlib.pyplot as plt

# ------------------------------
# 1. 이벤트 핸들러용 전역 변수
# ------------------------------
circles = []  # 원 패치 저장
running = True  # ESC로 종료용

def handle_key_press(event):
    global running, circles

    if event.key == 'escape':
        running = False
        plt.close()

    elif event.key == ' ':
        # 원 삭제
        for c in circles:
            c.remove()
        circles = []
        fig.canvas.draw_idle()


def handle_mouse(event):
    # 마우스 좌표가 axes 영역 안일 때만 동작
    if event.inaxes != ax:
        return
    
    x, y = event.xdata, event.ydata
    circle = plt.Circle((x, y), radius=10, color='red', fill=False, linewidth=2)
    ax.add_patch(circle)
    circles.append(circle)
    fig.canvas.draw_idle()


def handle_close(evt):
    global running
    running = False
    cap.release()
    print("Window closed.")


# ------------------------------
# 2. 프로그램 시작
# ------------------------------
cap = cv2.VideoCapture(0)

plt.ion()
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
plt.axis('off')

# fig.canvas.set_window_title('Video Capture + Events')

fig.canvas.mpl_connect('key_press_event', handle_key_press)
fig.canvas.mpl_connect('button_press_event', handle_mouse)
fig.canvas.mpl_connect('close_event', handle_close)

retval, frame = cap.read()
im = plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# ------------------------------
# 3. Main Loop (실시간)
# ------------------------------
while running:
    retval, frame = cap.read()
    if not retval:
        break

    im.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

cap.release()
plt.close()
