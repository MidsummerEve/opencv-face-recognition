# 导入 cv 模块
import os
import sys

import cv2 as cv


# 检测函数
def face_detect_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 用 cv.data.haarcascades 动态取级联文件路径，避免写死本机绝对路径
    cascade_path = os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_alt2.xml')
    face_detect = cv.CascadeClassifier(cascade_path)

    face = face_detect.detectMultiScale(gray)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y, x + w, y + h), color=(0, 0, 255), thickness=2)

    cv.imshow('result', img)


# 视频源：默认用仓库自带的示例视频，也可以命令行指定，例如
#     python 07视频检测.py D:\my_video.mp4
# 传入 0 表示使用本机摄像头
src = sys.argv[1] if len(sys.argv) > 1 else 'samples/demo.avi'
cap = cv.VideoCapture(int(src) if src.isdigit() else src)

if not cap.isOpened():
    print('无法打开视频源：%s' % src)
    raise SystemExit(1)

print('正在处理：%s   按 q 退出' % src)
DISPLAY_MIN_WIDTH = 480

# 循环读取每一帧
while True:
    flag, frame = cap.read()
    if not flag or frame is None:
        print('视频播放完毕')
        break
    face_detect_demo(frame)
    # 画面太小的话放大后再显示，纯为看得清楚
    if DISPLAY_MIN_WIDTH and frame.shape[1] < DISPLAY_MIN_WIDTH:
        s = DISPLAY_MIN_WIDTH / frame.shape[1]
        cv.imshow('result', cv.resize(frame, None, fx=s, fy=s,
                                      interpolation=cv.INTER_NEAREST))
    if ord('q') == cv.waitKey(1):
        break

# 释放资源
cap.release()
cv.destroyAllWindows()
