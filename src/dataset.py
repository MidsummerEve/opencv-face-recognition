"""
训练数据加载：扫描 data/ 目录，产出（人脸数组, 标签数组）。

命名约定
--------
文件名**点号前的数字就是标签 id**，同一个人的所有图片用同一个数字：

    1.s01_1.jpg   -> 标签 1
    1.s01_2.jpg   -> 标签 1
    2.s02_1.jpg   -> 标签 2

本仓库自带的 data/ 来自公开数据集 ORL/AT&T（40 人 × 10 张），
文件名形如 1.s01_1.jpg，标签即为被试编号。
"""

import os

import numpy as np
from PIL import Image

from src.config import DATA_DIR
from src.detector import create_detector, crop_faces, detect_faces

# 允许读取的图片后缀
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.pgm', '.tif', '.tiff')


def load_samples(data_dir=DATA_DIR, verbose=True):
    """遍历目录下的人脸图片，返回 (人脸列表, 标签列表, 统计信息)。

    统计信息是一个字典，含 total / used / skipped_no_face / skipped_bad_name，
    方便调用方打印或做数据质量检查。
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError('找不到数据目录：%s' % os.path.abspath(data_dir))

    detector = create_detector()

    image_paths = [
        os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))
        if f.lower().endswith(IMG_EXTS)
    ]

    faces = []
    labels = []
    skipped_no_face = 0
    skipped_bad_name = 0

    for path in image_paths:
        filename = os.path.basename(path)

        # 文件名点号前的部分即标签 id
        try:
            label = int(filename.split('.')[0])
        except ValueError:
            skipped_bad_name += 1
            if verbose:
                print('  跳过（文件名不是「数字.xxx」格式）：%s' % filename)
            continue

        # PIL 读图并转灰度（'L' 模式）
        gray = np.array(Image.open(path).convert('L'), dtype='uint8')

        detected = detect_faces(detector, gray)
        if len(detected) == 0:
            skipped_no_face += 1
            continue

        for face in crop_faces(gray, detected):
            faces.append(face)
            labels.append(label)

    stats = {
        'total': len(image_paths),
        'used': len(faces),
        'skipped_no_face': skipped_no_face,
        'skipped_bad_name': skipped_bad_name,
    }

    if verbose:
        print('扫描 %d 张图片，得到 %d 个人脸样本'
              % (stats['total'], stats['used']))
        print('  未检出人脸跳过 %d 张，文件名不合规跳过 %d 张'
              % (skipped_no_face, skipped_bad_name))

    return faces, labels, stats
