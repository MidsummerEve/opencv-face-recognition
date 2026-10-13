# 导入 cv 模块
import cv2 as cv

# 读取图片
img = cv.imread('samples/sample_face.jpg')
if img is None:
    raise SystemExit('未找到 samples/sample_face.jpg，请在项目根目录下运行本脚本')

# 为了看得清楚，先把小图放大一点
img = cv.resize(img, dsize=(368, 448))

# 矩形左上角坐标与宽高
x, y, w, h = 100, 100, 100, 100

# 绘制矩形（红框）
cv.rectangle(img, (x, y, x + w, y + h), color=(0, 0, 255), thickness=1)

# 绘制圆形（蓝圈）
cv.circle(img, center=(x + w, y + h), radius=100, color=(255, 0, 0), thickness=5)

# 显示
cv.imshow('re_img', img)

# 等待按键，按 q 退出
while True:
    if ord('q') == cv.waitKey(0):
        break

# 释放窗口资源
cv.destroyAllWindows()
