# 导入 cv 模块
import os

import cv2 as cv

# 读取图片
img = cv.imread('samples/sample_face.jpg')
if img is None:
    raise SystemExit('未找到 samples/sample_face.jpg，请在项目根目录下运行本脚本')

# 灰度转换
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 显示原图与灰度图
cv.imshow('read.jpg', img)
cv.imshow('gray_img', gray_img)

# 保存灰度图到 output/（该目录已在 .gitignore 中排除，不会污染仓库）
os.makedirs('output', exist_ok=True)
cv.imwrite(os.path.join('output', 'gray_img.jpg'), gray_img)
print('灰度图已保存到 output/gray_img.jpg')

# 等待按键
cv.waitKey(0)

# 释放窗口资源
cv.destroyAllWindows()
