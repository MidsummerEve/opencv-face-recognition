"""
人脸检测模块：封装 Haar 级联分类器的加载与调用。

检测参数来自 `src.config.DETECT_PARAMS`，保证训练与推理用的是同一套标准。
"""

import os

import cv2

from src.config import CASCADE_FILE, DETECT_PARAMS, FACE_SIZE


def create_detector():
    """创建 Haar 级联人脸检测器。

    分类器文件随 OpenCV 一起安装，用 `cv2.data.haarcascades` 动态取路径，
    不写死本机绝对路径，换台电脑也能跑。
    """
    cascade_path = os.path.join(cv2.data.haarcascades, CASCADE_FILE)
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError('加载级联分类器失败：%s' % cascade_path)
    return detector


def detect_faces(detector, gray):
    """在灰度图上检测人脸，返回 [(x, y, w, h), ...]。"""
    return detector.detectMultiScale(gray, **DETECT_PARAMS)


def crop_faces(gray, faces, face_size=FACE_SIZE):
    """按检测框裁出人脸，并统一缩放到 face_size。

    统一尺寸很重要：LBPH 会把人脸切成固定的 8x8 网格统计纹理，
    人脸大小不一致时同一个人的网格内容会错位，导致误判。
    """
    return [
        cv2.resize(gray[y:y + h, x:x + w], face_size,
                   interpolation=cv2.INTER_AREA)
        for (x, y, w, h) in faces
    ]


def draw_faces(image, faces, labels=None, color=(0, 0, 255)):
    """在图上画出人脸框；labels 不为空时同时写出姓名。"""
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        if labels is not None and i < len(labels):
            cv2.putText(image, labels[i], (x + 10, max(y - 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return image
