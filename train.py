"""
训练 LBPH 人脸识别模型。

用法：
    python train.py

前置：
    把带标签的人脸图片放进 data/。
    文件名点号前的数字就是标签 id，同一个人的所有图片用同一个数字：

        1.s01_1.jpg   1.s01_2.jpg   -> 标签 1
        2.s02_1.jpg                -> 标签 2

    仓库自带的 data/ 是公开数据集 ORL/AT&T（40 人 × 10 张），可直接跑，
    说明见 data/README.md。

产出：
    trainer/trainer.yml    训练好的模型（已在 .gitignore 中排除）
    trainer/labels.json    标签 id -> 姓名 的映射

关于参数：裁片尺寸、检测参数等都在 src/config.py，
训练与推理共用同一份，不要在脚本里另写一份。
"""

import os

from src import recognizer
from src.config import DATA_DIR, LABELS_PATH, TRAINER_PATH
from src.dataset import load_samples

# data/、trainer/ 这些路径都相对项目根目录。把工作目录切到脚本所在目录，
# 这样在 PyCharm 里运行时不依赖它的「工作目录」设置——PyCharm 默认用的是
# 外层工程目录，不切的话会去找 <工程目录>/data，报找不到数据目录。
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    print('从 %s/ 加载训练样本...' % DATA_DIR)
    faces, labels, _stats = load_samples(DATA_DIR)

    if not faces:
        raise SystemExit('没有可用的训练样本，请检查 data/ 目录')

    print('开始训练 LBPH 模型（%d 个样本）...' % len(faces))
    model = recognizer.train(faces, labels)

    names = recognizer.save(model, labels)
    print('模型已保存到 %s' % TRAINER_PATH)
    print('标签映射已保存到 %s（共 %d 个身份）' % (LABELS_PATH, len(names)))
    print()
    print('下一步：python recognize.py')


if __name__ == '__main__':
    main()
