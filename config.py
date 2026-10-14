"""
训练与推理的公共配置。

为什么需要这个文件
------------------
这些量**必须由训练（09数据训练.py）和推理（10人脸识别.py）共用同一份值**，
否则会出现"训练时裁到 100x100、推理时裁到别的尺寸"这类不一致——
**不会报错，但识别准确率会莫名其妙地崩掉**，而且极难排查。

所以凡是「两边必须一致」的量，都集中放在这里，改一处两边同时生效。

注意
----
本文件里的路径都是**相对项目根目录**的。请务必在项目根目录下运行脚本：

    cd opencv-face-recognition
    python 09数据训练.py
"""

# ---------------- 人脸处理 ----------------

# 人脸裁片统一缩放到的尺寸。训练与推理必须完全一致。
FACE_SIZE = (100, 100)

# 人脸检测用的 Haar 级联分类器文件名。
# 完整路径由 cv2.data.haarcascades 动态拼接（随 OpenCV 安装，不写死本机路径）。
CASCADE_FILE = 'haarcascade_frontalface_alt2.xml'

# 人脸检测参数，训练与推理必须完全一致。
# 用法：face_detector.detectMultiScale(gray, **DETECT_PARAMS)
#
# 各项含义：
#   scaleFactor  —— 每次把检测窗口放大多少倍（越小越慢越细）
#   minNeighbors —— 一个候选框要被多少个邻近框「投票」才认定是人脸。
#                   调大 = 更严格（少误检，可能漏检）；调小 = 更宽松。
#   minSize      —— 小于这个尺寸的框直接忽略，避免把噪点当人脸。
#
# 为什么统一到 minNeighbors=5：
#   推理端面对的是真实摄像头/杂乱背景，用 5 更稳健，不会把误检框当成人脸；
#   也避免训练时把误检样本喂进模型。在 ORL 留出集上代价约 1 个百分点。
DETECT_PARAMS = {
    'scaleFactor': 1.1,
    'minNeighbors': 5,
    'minSize': (30, 30),
}


# ---------------- 文件路径（相对项目根目录）----------------

# 训练产出的模型
TRAINER_PATH = 'trainer/trainer.yml'

# 标签 id -> 姓名 的映射，由 09数据训练.py 生成
LABELS_PATH = 'trainer/labels.json'
