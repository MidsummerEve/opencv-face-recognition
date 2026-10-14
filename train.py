"""
训练 LBPH 人脸识别模型。

用法：
    python train.py

前置：
    把带标签的人脸图片放进 data/。
    文件名**点号前的数字就是标签 id**，同一个人的所有图片用同一个数字：

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

from src import recognizer
from src.config import DATA_DIR, LABELS_PATH, TRAINER_PATH
from src.dataset import load_samples


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
