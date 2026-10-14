"""
视频中的人脸检测：把单帧检测循环应用到每一帧。

学习路径的第三站。相比静态图片，视频多了一个「逐帧读取」的循环，
以及必须处理「读完了 / 摄像头断了」的退出条件。

运行（务必在项目根目录下执行）：
    python tutorials/03_video_detection.py                 # 用自带示例视频
    python tutorials/03_video_detection.py 0               # 用本机摄像头
    python tutorials/03_video_detection.py D:\\my_video.mp4  # 用指定视频

按 q 退出。
"""

import os
import sys

import cv2

CASCADE = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt2.xml')
detector = cv2.CascadeClassifier(CASCADE)
if detector.empty():
    raise SystemExit('加载级联分类器失败：%s' % CASCADE)

# 画面比这个小就放大后再显示，纯为看得清楚
DISPLAY_MIN_WIDTH = 480


def detect_and_draw(frame):
    """在一帧上检测人脸并画出红框。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    return len(faces)


# 视频源：默认用仓库自带的示例视频；传 0 表示摄像头
source = sys.argv[1] if len(sys.argv) > 1 else 'samples/demo.avi'
cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
if not cap.isOpened():
    raise SystemExit('无法打开视频源：%s' % source)

print('正在处理：%s   按 q 退出' % source)

frame_count = 0
face_count = 0

while True:
    ok, frame = cap.read()
    # 视频读完了，或者摄像头断开了
    if not ok or frame is None:
        print('视频播放完毕 / 摄像头断开')
        break

    face_count += detect_and_draw(frame)
    frame_count += 1

    if frame_count % 30 == 0:
        print('已处理 %d 帧，累计检出 %d 张人脸' % (frame_count, face_count))

    disp = frame
    if frame.shape[1] < DISPLAY_MIN_WIDTH:
        s = DISPLAY_MIN_WIDTH / frame.shape[1]
        disp = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('result', disp)

    if ord('q') == cv2.waitKey(1):
        break

cap.release()
cv2.destroyAllWindows()
print('共处理 %d 帧，检出 %d 张人脸' % (frame_count, face_count))
