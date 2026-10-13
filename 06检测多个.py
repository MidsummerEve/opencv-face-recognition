# 导入 cv 模块
import os

import cv2 as cv


# 检测函数
def face_detect_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 用 cv.data.haarcascades 动态取级联文件路径，避免写死本机绝对路径
    cascade_path = os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_detect = cv.CascadeClassifier(cascade_path)

    face = face_detect.detectMultiScale(gray, 1.32)
    print('检测到 %d 张人脸' % len(face))

    for x, y, w, h in face:
        cv.rectangle(img, (x, y, x + w, y + h), color=(0, 0, 255), thickness=2)

    cv.imshow('result', img)


# 读取图像：这张是多人合影（由 ORL 公开数据集的人脸横向拼接而成）
img1 = cv.imread('samples/group_faces.jpg')
if img1 is None:
    raise SystemExit('未找到 samples/group_faces.jpg，请在项目根目录下运行本脚本')

img = cv.resize(img1, dsize=(960, 560))

face_detect_demo(img)

# 等待，按 q 退出
while True:
    if ord('q') == cv.waitKey(0):
        break

# 释放窗口资源
cv.destroyAllWindows()
