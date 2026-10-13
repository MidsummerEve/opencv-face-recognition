# 导入 cv 模块
import cv2 as cv

# 读取图片
img = cv.imread('samples/sample_face.jpg')
if img is None:
    raise SystemExit('未找到 samples/sample_face.jpg，请在项目根目录下运行本脚本')

# 修改尺寸
resize_img = cv.resize(img, dsize=(200, 200))

# 显示原图与缩放后的图
cv.imshow('img', img)
cv.imshow('resize_img', resize_img)

# 打印尺寸（shape 顺序为 高, 宽, 通道数）
print('未修改：', img.shape)
print('修改后：', resize_img.shape)

# 等待按键，按 q 退出
while True:
    if ord('q') == cv.waitKey(0):
        break

# 释放窗口资源
cv.destroyAllWindows()
