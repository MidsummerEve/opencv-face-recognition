# 导入 cv 模块
import cv2 as cv

# 读取图片
# 示例图片 samples/sample_face.jpg 来自公开数据集 ORL/AT&T（见 README）
img = cv.imread('samples/sample_face.jpg')

# 读图失败时 cv.imread 不会报错，只会返回 None，这里显式检查一下
if img is None:
    raise SystemExit('未找到 samples/sample_face.jpg，请在项目根目录下运行本脚本')

# 显示图片
cv.imshow('read.jpg', img)

# 等待按键，按任意键关闭窗口
cv.waitKey(0)

# 释放窗口资源
cv.destroyAllWindows()
