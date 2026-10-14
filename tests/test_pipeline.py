"""
端到端冒烟测试：确认核心链路没有被改坏。

运行：
    pip install pytest
    pytest -v

这些测试都很轻量（不训练完整模型），几秒内跑完，
适合作为「改完代码之后、提交之前」的一道保险。
"""

import os
import sys

import numpy as np
import pytest

# 让测试能在项目根目录直接跑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import recognizer
from src.config import (CASCADE_FILE, DETECT_PARAMS, FACE_SIZE, TRAINER_PATH,
                        UNKNOWN_THRESHOLD)
from src.detector import create_detector, crop_faces, detect_faces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- 配置一致性
def test_face_size_is_2d_tuple():
    """裁片尺寸必须是二元组——训练与推理都靠它。"""
    assert isinstance(FACE_SIZE, tuple) and len(FACE_SIZE) == 2
    assert all(isinstance(v, int) and v > 0 for v in FACE_SIZE)


def test_detect_params_keys():
    """检测参数必须齐全，缺一个 OpenCV 会用到默认值，两边就可能不一致。"""
    assert set(DETECT_PARAMS) == {'scaleFactor', 'minNeighbors', 'minSize'}


def test_unknown_threshold_positive():
    assert UNKNOWN_THRESHOLD > 0


def test_holdout_subjects_are_valid():
    """留出身份必须是非空整数集合，且不能太大——否则训练样本会不够。"""
    from src.config import HOLDOUT_SUBJECTS
    assert HOLDOUT_SUBJECTS, '留出身份不能为空，否则演示不了陌生人报警'
    assert all(isinstance(s, int) and s > 0 for s in HOLDOUT_SUBJECTS)
    assert len(HOLDOUT_SUBJECTS) < 40


def test_trigger_frames_positive():
    from src.config import UNKNOWN_TRIGGER_FRAMES
    assert UNKNOWN_TRIGGER_FRAMES > 0


# ---------------------------------------------------------------- 数据集
def test_subject_of_parses_label():
    """文件名点号前的数字就是标签 id——这是训练能对上号的前提。"""
    from src.dataset import subject_of
    assert subject_of('1.s01_1.jpg') == 1
    assert subject_of('40.s40_10.jpg') == 40
    assert subject_of('readme.txt') is None
    assert subject_of('没有数字.jpg') is None


def test_holdout_not_in_dataset_labels():
    """训练产出的标签映射里，不应出现留出身份。

    这条直接检查训练产物：如果哪天忘了排除，陌生人演示会立刻失效。
    """
    from src.config import HOLDOUT_SUBJECTS
    names = recognizer.load_names()
    if not names:
        pytest.skip('还没有训练好的模型，跳过（先跑 python train.py）')
    assert not (set(names) & HOLDOUT_SUBJECTS), (
        '留出身份出现在了训练结果里：%s' % sorted(set(names) & HOLDOUT_SUBJECTS))


# ---------------------------------------------------------------- 检测模块
def test_create_detector():
    """级联分类器能正常加载（验证 cv2.data.haarcascades 路径可用）。"""
    detector = create_detector()
    assert not detector.empty()


def test_detect_and_crop_on_sample():
    """在示例图上应能检出人脸，且裁片尺寸等于 FACE_SIZE。"""
    import cv2

    img = cv2.imread(os.path.join(ROOT, 'samples', 'sample_face.jpg'),
                     cv2.IMREAD_GRAYSCALE)
    assert img is not None, '缺少 samples/sample_face.jpg'

    detector = create_detector()
    faces = detect_faces(detector, img)
    assert len(faces) >= 1, '示例图上没检出人脸'

    crops = crop_faces(img, faces)
    assert len(crops) == len(faces)
    for c in crops:
        assert c.shape == (FACE_SIZE[1], FACE_SIZE[0])


def test_crop_faces_shape_with_synthetic_input():
    """用构造数据验证裁片逻辑，不依赖具体图片。"""
    gray = np.full((200, 200), 128, np.uint8)
    faces = np.array([[10, 20, 60, 80]])
    crops = crop_faces(gray, faces)
    assert len(crops) == 1
    assert crops[0].shape == (FACE_SIZE[1], FACE_SIZE[0])


# ---------------------------------------------------------------- 识别模块
def test_train_predict_roundtrip():
    """同一个人脸训练后应能被认出，且置信度极低（LBPH 下距离越小越像）。"""
    rng = np.random.default_rng(0)
    face_a = rng.integers(0, 255, FACE_SIZE[::-1], dtype=np.uint8)
    face_b = rng.integers(0, 255, FACE_SIZE[::-1], dtype=np.uint8)

    model = recognizer.train([face_a, face_b], [1, 2])

    label, distance = recognizer.predict(model, face_a)
    assert label == 1
    # 与训练样本完全相同的输入，距离应当接近 0
    assert distance < 1e-6, '完全相同的输入本该得到 0 距离，实际 %.3f' % distance


def test_train_rejects_empty_input():
    with pytest.raises(ValueError):
        recognizer.train([], [])


def test_load_raises_when_no_model(tmp_path, monkeypatch):
    """模型不存在时应抛出 FileNotFoundError 并提示怎么处理。"""
    import src.recognizer as r
    monkeypatch.setattr(r, 'TRAINER_PATH', str(tmp_path / 'not_exist.yml'))
    with pytest.raises(FileNotFoundError):
        recognizer.load()


# ---------------------------------------------------------------- 告警模块
def test_alarm_not_configured(monkeypatch):
    """未配置环境变量时不应发送，也不应抛异常。"""
    for k in ('SMS_USER', 'SMS_PASSWORD', 'SMS_PHONE'):
        monkeypatch.delenv(k, raising=False)
    from src import alarm
    assert alarm.is_configured() is False
    assert alarm.send_alarm() is False


def test_alarm_handles_network_failure(monkeypatch):
    """凭据配好但网络不通时，只打印失败，不应把程序打断、也不应抛异常。

    这里把 urlopen 换成会抛错的假函数——测试不应该真的去访问外部服务。
    """
    monkeypatch.setenv('SMS_USER', 'dummy')
    monkeypatch.setenv('SMS_PASSWORD', 'dummy')
    monkeypatch.setenv('SMS_PHONE', '00000000000')

    from src import alarm

    def always_fail(*args, **kwargs):
        raise OSError('network unreachable')

    monkeypatch.setattr(alarm.urllib.request, 'urlopen', always_fail)

    assert alarm.is_configured() is True
    assert alarm.send_alarm() is False  # 失败时返回 False，而不是抛出


def test_alarm_success_path(monkeypatch):
    """模拟平台返回状态码 0（发送成功），应返回 True 且进入冷却。"""
    monkeypatch.setenv('SMS_USER', 'dummy')
    monkeypatch.setenv('SMS_PASSWORD', 'dummy')
    monkeypatch.setenv('SMS_PHONE', '00000000000')

    from src import alarm
    alarm._last_sms_time = 0.0  # 重置冷却状态，避免测试互相影响

    class FakeResponse:
        def read(self):
            return b'0'

    monkeypatch.setattr(alarm.urllib.request, 'urlopen',
                        lambda *a, **kw: FakeResponse())

    assert alarm.send_alarm() is True
    # 冷却期内再发应被拦下
    assert alarm.send_alarm() is False
