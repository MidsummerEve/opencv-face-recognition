"""
图像基础操作：读取 → 灰度 → 缩放 → 绘制。

学习路径的第一站，四个 OpenCV 基础操作依次演示。
每个脚本都是独立可运行的，不依赖项目里的其它模块，方便单独复制出去用。

运行：
    python tutorials/01_image_basics.py

按任意键进入下一步，按 q 退出。
"""

import os

import cv2

# samples/ 相对项目根目录，而本脚本在 tutorials/ 下，所以要往上一级。
# 这样在 PyCharm 里运行时不依赖它的「工作目录」设置。
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE = 'samples/sample_face.jpg'

img = cv2.imread(SAMPLE)
if img is None:
    raise SystemExit('未找到 %s，检查一下 samples/ 目录是不是完整的' % SAMPLE)

# shape 的顺序是（高, 宽, 通道数）
print('原图尺寸:', img.shape)

# ---------------------------------------------------------------- 1. 读取并显示
cv2.imshow('1_read', img)

# ---------------------------------------------------------------- 2. 灰度转换
# 人脸识别的绝大多数算法都只在灰度图上工作，因为颜色对「是谁」没有帮助，
# 去掉颜色既能减少计算量，也能降低光照变化带来的干扰。
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('2_gray', gray)

# ---------------------------------------------------------------- 3. 修改尺寸
# 统一尺寸是人脸识别里很关键的一步，见 src/config.py 的 FACE_SIZE 说明。
resized = cv2.resize(img, dsize=(200, 200))
print('缩放后尺寸:', resized.shape)
cv2.imshow('3_resize', resized)

# ---------------------------------------------------------------- 4. 绘制图形
# 人脸检测的结果就是一张图上的一堆矩形框，所以先学会画框。
# 原图只有 92x112，先放大才能看清画出来的图形。
canvas = cv2.resize(img, dsize=(368, 448))
x, y, w, h = 100, 100, 100, 100

# 矩形：(左上角x, 左上角y), (右下角x, 右下角y)
cv2.rectangle(canvas, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
# 圆形：圆心 + 半径。注意 OpenCV 的颜色顺序是 BGR 不是 RGB
cv2.circle(canvas, center=(x + w, y + h), radius=100, color=(255, 0, 0), thickness=5)
cv2.imshow('4_draw', canvas)

print("四个窗口都已打开，按任意键继续，按 'q' 退出")
while True:
    if ord('q') == cv2.waitKey(0):
        break

cv2.destroyAllWindows()
