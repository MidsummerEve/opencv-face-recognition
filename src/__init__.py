"""
基于 OpenCV 的人脸识别系统 —— 核心模块。

模块划分
--------
- `config`     : 训练与推理必须共用的配置（裁片尺寸、检测参数、路径）
- `detector`   : 人脸检测（Haar 级联的加载、检测、裁片）
- `dataset`    : 训练数据加载（扫描 data/，解析标签）
- `recognizer` : LBPH 识别的训练、保存、加载、预测
- `alarm`      : 未知人员短信告警（凭据走环境变量）

被两个入口脚本共同使用：

    train.py      训练模型
    recognize.py  人脸识别与未知人员告警
"""

__all__ = ['config', 'detector', 'dataset', 'recognizer', 'alarm']
