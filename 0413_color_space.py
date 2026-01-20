import cv2
import numpy as np

window_count = 0

def create_info_window(bgr):
    global window_count

    b, g, r = map(int, bgr)
    color_patch = np.zeros((220, 350, 3), dtype=np.uint8)
    color_patch[:] = (b, g, r)

    # RGB 텍스트
    cv2.putText(color_patch, f"RGB: R={r}, G={g}, B={b}",
                (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255-b, 255-g, 255-r), 2)

    # HSV 변환
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = map(int, hsv)

    # HSV 텍스트
    cv2.putText(color_patch, f"HSV: H={h}, S={s}, V={v}",
                (10, 180), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255-b, 255-g, 255-r), 2)

    win_name = f"Measure-{window_count}"
    cv2.imshow(win_name, color_patch)

    window_count += 1


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param['frame']
        bgr = frame[y, x]
        create_info_window(bgr)


cap = cv2.VideoCapture(0)
cv2.namedWindow('Video')

mouse_param = {'frame': None}
cv2.setMouseCallback('Video', onMouse, mouse_param)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mouse_param['frame'] = frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
