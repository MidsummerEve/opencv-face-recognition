"""
训练与推理共用的配置。

train.py 训练出 trainer/trainer.yml，recognize.py 再读它——两个脚本之间
靠文件耦合。如果两边对「人脸长什么样」的假设不一致，程序不会报错，但识别率
会悄悄变差：比如训练时裁 100x100、推理时裁 80x80，LBPH 的网格特征全对不上，
看报错信息也找不到原因。

所以两边必须一致的东西都放这里，改一处两边生效。
"""

# 人脸裁片统一缩放到的尺寸，训练与推理必须一致
FACE_SIZE = (100, 100)

# Haar 级联分类器文件名。完整路径由 cv2.data.haarcascades 拼出来，
# 不写死本机路径
CASCADE_FILE = 'haarcascade_frontalface_alt2.xml'

# 人脸检测参数，训练与推理必须一致
# 用法：detectMultiScale(gray, **DETECT_PARAMS)
#   scaleFactor  每次把检测窗口放大多少倍
#   minNeighbors 一个候选框要被多少个邻近框投票才认定是人脸，调大更严格
#   minSize      小于这个尺寸的框直接忽略
# minNeighbors 取 5 是因为推理端面对的是真实摄像头和杂乱背景，5 不容易把
# 误检框当成人脸。在 ORL 留出集上比取 3 低约 1 个百分点。
DETECT_PARAMS = {
    'scaleFactor': 1.1,
    'minNeighbors': 5,
    'minSize': (30, 30),
}

# 判定陌生人的距离阈值。LBPH 的置信度是距离，越小越像。
#
# 这个数是从实测里挑的。登记照用每人前 5 张，查询时用同一人的后 5 张（熟人）
# 和另外 5 个没登记的人（陌生人），测出来：
#   熟人   50.0 ~ 90.5
#   陌生人 71.9 ~ 98.0
# 两段重叠在 72~91，没有完美阈值。几个候选：
#   60  -> 陌生人放行 0%，但熟人只通过 27%
#   74  -> 陌生人放行约 5%，熟人通过约 79%
#   80  -> 熟人通过 90%，陌生人放行 37%
# 漏放陌生人比误拒熟人严重，所以取 74。
#
# 换成自己的数据要重新标定：python recognize.py 视频 --truth <姓名>
UNKNOWN_THRESHOLD = 74

# 留出身份。这些人不参与训练，识别时会被判为陌生人——预警系统本来就只认识
# 名单内的人。同时也是 holdout：留一部分数据不训练，看模型对没见过的输入
# 是什么反应。samples/demo_stranger.avi 就是用这些人拼的。
HOLDOUT_SUBJECTS = {36, 37, 38, 39, 40}

# 陌生人累计多少次触发报警。计的是次数不是连续帧：只有识别出已知人员才清零，
# 这样有人反复快速经过镜头也躲不掉。
UNKNOWN_TRIGGER_FRAMES = 70

# 两次报警短信之间的最短间隔（秒）
SMS_COOLDOWN = 60


# 下面几个路径都相对项目根目录
DATA_DIR = 'data'
TRAINER_PATH = 'trainer/trainer.yml'
LABELS_PATH = 'trainer/labels.json'
