"""
训练 LBPH 人脸识别模型。

用法：
    1. 先把人脸图片放进 data/ 目录，文件名形如  1.xxx.jpg
       —— 点号前面的数字就是该图片的标签 id（同一个人的所有图片用同一个 id）
    2. 运行：python 09数据训练.py
    3. 训练结果保存在 trainer/trainer.yml

本仓库自带的 data/ 来自公开数据集 ORL/AT&T（40 人 × 10 张），
文件名形如 1.s01_1.jpg，标签即为被试编号。
"""
import json
import os

import cv2
import numpy as np
from PIL import Image

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.pgm', '.tif', '.tiff')

# 把人脸裁片统一缩放到这个尺寸再训练。
# 原因：LBPH 会把人脸切成固定的 8x8 网格来统计纹理，人脸大小不一致时
# 同一个人的网格内容会错位，导致误判。归一化尺寸后准确率明显提升。
FACE_SIZE = (100, 100)


def getImageAndLabels(path):
    """遍历目录下的人脸图片，返回（人脸数组, 标签数组）"""
    facesSamples = []
    ids = []

    # 级联分类器随 OpenCV 安装，用 cv2.data.haarcascades 动态取路径
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt2.xml')
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        raise SystemExit('加载级联分类器失败：%s' % cascade_path)

    imagePaths = [
        os.path.join(path, f) for f in sorted(os.listdir(path))
        if f.lower().endswith(IMG_EXTS)
    ]
    print('找到 %d 张待处理图片' % len(imagePaths))

    skipped = 0
    for imagePath in imagePaths:
        filename = os.path.split(imagePath)[1]
        # 文件名点号前的部分就是标签 id
        try:
            label = int(filename.split('.')[0])
        except ValueError:
            print('  跳过（文件名不是「数字.xxx」格式）：%s' % filename)
            skipped += 1
            continue

        # PIL 读图并转灰度（'L' 模式）
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        faces = face_detector.detectMultiScale(img_numpy)
        if len(faces) == 0:
            skipped += 1
            continue

        for (x, y, w, h) in faces:
            ids.append(label)
            face = img_numpy[y:y + h, x:x + w]
            face = cv2.resize(face, FACE_SIZE, interpolation=cv2.INTER_AREA)
            facesSamples.append(face)

    print('有效人脸样本 %d 个，跳过 %d 张（未检出人脸或文件名不合规）'
          % (len(facesSamples), skipped))
    return facesSamples, ids


if __name__ == '__main__':
    path = './data/'

    # 获取人脸数组和标签数组
    faces, ids = getImageAndLabels(path)
    if not faces:
        raise SystemExit('没有可用的训练样本，请检查 data/ 目录')

    # 创建 LBPH 识别器并训练
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))

    # 保存模型
    os.makedirs('trainer', exist_ok=True)
    recognizer.write('trainer/trainer.yml')
    print('模型已保存到 trainer/trainer.yml')

    # 顺便记录 id -> 姓名 的映射，供 10人脸识别.py 显示用
    unique_ids = sorted(set(ids))
    labels = {str(i): 'person%02d' % i for i in unique_ids}
    with open('trainer/labels.json', 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print('标签映射已保存到 trainer/labels.json（共 %d 个身份）' % len(unique_ids))
