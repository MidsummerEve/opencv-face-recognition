"""
LBPH 人脸识别器：训练、保存、加载、预测。

关于 LBPH
---------
LBPH（Local Binary Patterns Histograms）把人脸切成固定的网格，
统计每个格子的局部二值模式直方图作为特征。特点：

- 不需要 GPU，训练秒级完成；
- `predict()` 返回的 confidence 是**距离**——**越小表示越像**（这点和一般
  分类器的「置信度」方向相反，容易搞错）；
- 对成像条件（人脸尺寸、背景、光照、视频压缩）比较敏感，
  所以训练与推理必须共用 `src.config` 里的同一套参数。
"""

import json
import os

import cv2
import numpy as np

from src.config import LABELS_PATH, TRAINER_PATH


def train(faces, labels):
    """训练 LBPH 模型，返回识别器对象。"""
    if not faces:
        raise ValueError('没有可用的训练样本')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    return recognizer


def save(recognizer, labels):
    """保存模型与标签映射，返回 {标签id: 姓名} 字典。"""
    os.makedirs(os.path.dirname(TRAINER_PATH) or '.', exist_ok=True)
    recognizer.write(TRAINER_PATH)

    names = {str(i): 'person%02d' % i for i in sorted(set(labels))}
    with open(LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(names, f, ensure_ascii=False, indent=2)

    return names


def load_names():
    """读取标签映射；文件不存在时返回空字典。"""
    if not os.path.exists(LABELS_PATH):
        return {}
    with open(LABELS_PATH, encoding='utf-8') as f:
        return {int(k): v for k, v in json.load(f).items()}


def load():
    """加载已训练的模型，返回 (识别器, {标签id: 姓名})。"""
    if not os.path.exists(TRAINER_PATH):
        raise FileNotFoundError(
            '未找到模型 %s，请先运行 train.py 训练' % TRAINER_PATH)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
    return recognizer, load_names()


def predict(recognizer, face):
    """对单张已裁好的人脸做预测，返回 (标签id, 距离)。

    距离越小越像；超过 `src.config.UNKNOWN_THRESHOLD` 应判为未知人员。
    """
    return recognizer.predict(face)
