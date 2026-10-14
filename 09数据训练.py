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

# 训练与推理必须共用的量统一放在 config.py：
# FACE_SIZE（裁片尺寸）、CASCADE_FILE（级联文件）、DETECT_PARAMS（检测参数）、
# 以及模型/标签的保存路径。具体原因见 config.py 开头的说明。
from config import (CASCADE_FILE, DETECT_PARAMS, FACE_SIZE, LABELS_PATH,
                    TRAINER_PATH)

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.pgm', '.tif', '.tiff')


def getImageAndLabels(path):
    """遍历目录下的人脸图片，返回（人脸数组, 标签数组）"""
    facesSamples = []
    ids = []

    # 级联分类器随 OpenCV 安装，用 cv2.data.haarcascades 动态取路径
    cascade_path = os.path.join(cv2.data.haarcascades, CASCADE_FILE)
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

        # 检测参数与 10人脸识别.py 共用 config.DETECT_PARAMS
        faces = face_detector.detectMultiScale(img_numpy, **DETECT_PARAMS)
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
    os.makedirs(os.path.dirname(TRAINER_PATH), exist_ok=True)
    recognizer.write(TRAINER_PATH)
    print('模型已保存到 %s' % TRAINER_PATH)

    # 顺便记录 id -> 姓名 的映射，供 10人脸识别.py 显示用
    unique_ids = sorted(set(ids))
    labels = {str(i): 'person%02d' % i for i in unique_ids}
    with open(LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print('标签映射已保存到 %s（共 %d 个身份）' % (LABELS_PATH, len(unique_ids)))
