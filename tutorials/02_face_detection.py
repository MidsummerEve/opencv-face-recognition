"""
人脸检测：用 Haar 级联分类器在图片里找脸。

包含两个场景：
    场景 A —— 单张人脸（samples/sample_face.jpg）
    场景 B —— 多人合影（samples/group_faces.jpg）

两个场景只差一个循环，所以合并在一个脚本里对比着看更清楚。

运行：
    python tutorials/02_face_detection.py
按任意键切换场景，按 q 退出。

关于 Haar 级联
--------------
它用一堆「像」和「不像」的简单特征层层筛选，速度快、不需要 GPU，
是深度学习普及之前的主流方案。缺点是对侧脸、遮挡、大角度旋转不敏感。
"""

import os

import cv2

# samples/ 相对项目根目录，而本脚本在 tutorials/ 下，所以要往上一级。
# 这样在 PyCharm 里运行时不依赖它的「工作目录」设置。
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 级联分类器文件随 OpenCV 一起安装，用 cv2.data.haarcascades 动态取路径，
# 不要写死本机绝对路径，否则换台电脑就跑不起来。
# （项目正式代码里这个文件名统一在 src/config.py 管理，这里为了脚本能
#   单独运行，写在本地。）
CASCADE = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt2.xml')
detector = cv2.CascadeClassifier(CASCADE)
if detector.empty():
    raise SystemExit('加载级联分类器失败：%s' % CASCADE)


def detect(gray, scale_factor=1.1, min_neighbors=5):
    """检测人脸，返回 [(x, y, w, h), ...]。

    scale_factor  —— 每次把检测窗口放大多少倍
    min_neighbors —— 一个框要被多少个邻近框「投票」才认定是人脸
                     调大 = 更严格（少误检）；调小 = 更宽松（可能误检）
    """
    return detector.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
        minSize=(30, 30))


def show_scene(title, path, scale_factor=1.1, min_neighbors=5):
    img = cv2.imread(path)
    if img is None:
        print('未找到 %s，跳过' % path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect(gray, scale_factor, min_neighbors)
    print('%-22s 检测到 %d 张人脸' % (title, len(faces)))

    # 把人脸框画回原图
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)

    cv2.imshow(title, img)


# ---------------------------------------------------------- 场景 A：单张人脸
# samples/sample_face.jpg 只有 92x112，放大后便于观察
single = cv2.imread('samples/sample_face.jpg')
if single is not None:
    big = cv2.resize(single, dsize=(480, 560))
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    faces = detect(gray)
    print('场景 A 单张人脸：检测到 %d 张' % len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(big, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    cv2.imshow('A_single_face', big)

# ---------------------------------------------------------- 场景 B：多人合影
# 这张合影由 ORL 公开数据集的人脸横向拼接而成
show_scene('B_group_faces', 'samples/group_faces.jpg', scale_factor=1.32)

print("按任意键继续，按 'q' 退出")
while True:
    if ord('q') == cv2.waitKey(0):
        break

cv2.destroyAllWindows()
