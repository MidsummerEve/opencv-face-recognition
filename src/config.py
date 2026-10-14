"""
训练与推理的公共配置。

为什么需要这个文件
------------------
训练（train.py）和推理（recognize.py）之间是**通过文件耦合**的：
train.py 产出 trainer/trainer.yml，recognize.py 再读它。

这种耦合有个隐蔽的危险：**如果两边对「人脸长什么样」的假设不一致，
程序不会报错，但结果会悄悄变差**。比如训练时把人脸裁成 100x100 存进模型、
推理时裁成 80x80 去比对——代码能跑，没有任何报错，但 LBPH 的网格特征
全都对不上，识别率直接崩掉，而且从报错信息里看不出任何线索。

所以凡是「两边必须一致」的量，全部集中放在这里，改一处两边同时生效。
"""

# ---------------- 人脸处理 ----------------

# 人脸裁片统一缩放到的尺寸。训练与推理必须完全一致。
FACE_SIZE = (100, 100)

# 人脸检测用的 Haar 级联分类器文件名。
# 完整路径由 cv2.data.haarcascades 动态拼接（随 OpenCV 安装，不写死本机路径）。
CASCADE_FILE = 'haarcascade_frontalface_alt2.xml'

# 人脸检测参数，训练与推理必须完全一致。
# 用法：detector.detectMultiScale(gray, **DETECT_PARAMS)
#
# 各项含义：
#   scaleFactor  —— 每次把检测窗口放大多少倍（越小越慢越细）
#   minNeighbors —— 一个候选框要被多少个邻近框「投票」才认定是人脸。
#                   调大 = 更严格（少误检，可能漏检）；调小 = 更宽松。
#   minSize      —— 小于这个尺寸的框直接忽略，避免把噪点当人脸。
#
# 为什么取 minNeighbors=5：
#   推理端面对的是真实摄像头与杂乱背景，用 5 更不容易把误检框当人脸，
#   也避免训练时把误检样本喂进模型。在 ORL 留出集上代价约 1 个百分点。
DETECT_PARAMS = {
    'scaleFactor': 1.1,
    'minNeighbors': 5,
    'minSize': (30, 30),
}

# 置信度阈值：LBPH 的置信度是「距离」，越小越像，超过该值判为未知人员。
# 这个值随人脸尺寸、背景、视频编码方式变化很大，需要按自己的数据调，
# 详细说明见 README 的「置信度阈值怎么调」一节。
UNKNOWN_THRESHOLD = 80


# ---------------- 文件路径（相对项目根目录）----------------

# 数据目录
DATA_DIR = 'data'

# 训练产出的模型
TRAINER_PATH = 'trainer/trainer.yml'

# 标签 id -> 姓名 的映射，由 train.py 生成
LABELS_PATH = 'trainer/labels.json'
