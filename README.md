# 基于 OpenCV 的人脸识别系统

用**经典计算机视觉方法**实现的完整人脸识别流程：
**人脸检测 → 特征训练 → 实时识别 → 未知人员短信告警**。

不依赖深度学习框架，纯 CPU 运行，从数据到告警每一步都可复现。

![识别效果](screenshots/demo_recognition.png)

*图为演示视频中的 4 帧：红框是检测到的人脸，绿字是识别结果与置信度（越小越像，这里都是 0）。
人脸来自 ORL 公开数据集。*

**技术栈** `Python` · `OpenCV (opencv-contrib)` · `NumPy` · `Pillow`

---

## 项目亮点

| 亮点 | 说明 |
|---|---|
| **完整链路** | 不只做检测或识别，而是贯穿「采集 → 训练 → 识别 → 告警」，含数据标注规范与评测脚本 |
| **量化评测** | 区分「演示集」与「留出集」两套指标，避免用训练数据自证；留出集准确率 **93.8%** |
| **工程规范** | 模块分层（检测 / 数据 / 识别 / 告警）；训练与推理共用的参数集中在 `src/config.py` |
| **凭据安全** | 短信平台账号密码走环境变量，仓库里不含任何敏感信息 |
| **记录局限** | 实测发现 LBPH 的置信度**无法干净区分「认对」与「认错」**，并解释了这正是行业转向深度嵌入的原因 |

---

## 快速开始

```bash
pip install -r requirements.txt

# 1. 训练（仓库自带 ORL 公开数据集，40 人 × 10 张，可直接跑）
python train.py

# 2. 识别（默认用自带的演示视频）
python recognize.py

# 也可以用自己的摄像头
python recognize.py 0

# 无图形界面 / 只想看统计
python recognize.py samples/demo.avi --no-display
```

> 安装时**必须用 `opencv-contrib-python`，不是 `opencv-python`**。
> `cv2.face.LBPHFaceRecognizer_create` 只存在于 contrib 版本，
> 装错会报 `module 'cv2' has no attribute 'face'`。

---

## 项目结构

```
.
├── train.py                    # 训练入口
├── recognize.py                # 识别入口（含未知人员告警）
│
├── src/                        # 核心模块（被两个入口共同导入）
│   ├── config.py               # 训练与推理必须共用的配置
│   ├── detector.py             # 人脸检测：Haar 级联的加载 / 检测 / 裁片
│   ├── dataset.py              # 训练数据加载：扫描 data/、解析标签
│   ├── recognizer.py           # LBPH 的训练 / 保存 / 加载 / 预测
│   └── alarm.py                # 未知人员短信告警（凭据走环境变量）
│
├── tutorials/                  # 学习路径（独立可运行，不依赖 src）
│   ├── 01_image_basics.py      # 读图 → 灰度 → 缩放 → 绘制
│   ├── 02_face_detection.py    # 单张人脸 / 多人合影检测
│   ├── 03_video_detection.py   # 视频逐帧检测
│   └── 04_face_capture.py      # 从摄像头采集训练素材
│
├── data/                       # 训练数据（ORL/AT&T 公开数据集）
│   └── README.md               # 数据集来源与命名规则
├── samples/                    # 演示素材（示例图 / 演示视频）
├── screenshots/                # 效果截图
├── trainer/                    # 训练产物（不入库，由 train.py 生成）
│
├── requirements.txt
├── .env.example                # 短信告警配置模板
└── LICENSE
```

**为什么这样分层**：`src/detector.py`、`src/recognizer.py`、`src/config.py`
被 `train.py` 和 `recognize.py` **共同导入**——这才是有 `src/` 的意义所在。
`tutorials/` 则是独立的学习脚本，刻意不依赖 `src/`，方便单独复制出去运行。

---

## 关于 `src/config.py`

这个项目有个隐蔽的风险：**训练与推理之间是通过"文件"耦合的**——
`train.py` 产出 `trainer/trainer.yml`，`recognize.py` 再读它，不是函数调用。

如果两边对「人脸长什么样」的假设不一致，**程序不会报错，但结果会悄悄变差**。
比如训练时把人脸裁成 100×100 存进模型、推理时裁成 80×80 去比对：
代码能跑，没有任何报错，但 LBPH 的网格特征全都对不上，识别率直接崩掉，
而且**从报错信息里看不出任何线索**。

所以凡是「两边必须一致」的量，全部集中放在 `src/config.py`：

| 配置项 | 不一致的后果 |
|---|---|
| `FACE_SIZE` | 裁片尺寸不同 → 特征网格错位，识别率静默崩坏 |
| `CASCADE_FILE` | 训练与推理「找脸」用的分类器不同 |
| `DETECT_PARAMS` | 同上。其中 `minNeighbors` 决定检测的严格程度 |
| `UNKNOWN_THRESHOLD` | 判定「未知人员」的置信度阈值 |
| `TRAINER_PATH` / `LABELS_PATH` | 找不到模型或标签文件 |

改一处，两边同时生效。

---

## 实测结果

以下数字都是在本仓库的数据上实跑得到的。

### 1. 演示视频端到端（`samples/demo.avi`）

| 指标 | 结果 |
|---|---|
| 人脸检出 | 64 / 64 帧（100%）|
| 正确识别 | **64 / 64（100%）** |
| 平均置信度 | 0.00 |

### 2. 泛化测试（模型没见过的照片）

用**每人前 5 张训练**、**后 5 张测试**，避免用训练数据自证：

| 方案 | 标签正确率 | 平均置信度 |
|---|---|---|
| 不做尺寸归一化 | 88.6%（156/176）| 93.0 |
| **加尺寸归一化**（本项目采用）| **93.8%（165/176）** | 67.0 |

尺寸归一化提升明显，因为 LBPH 会把脸切成固定 8×8 网格统计纹理，
人脸大小不一致时网格内容会错位。

### 3. 置信度能否区分「认对」和「认错」

| | 样本数 | 平均置信度 | 范围 |
|---|---|---|---|
| 正确匹配 | 165 | 66.1 | 50.0 ~ 90.2 |
| 错误匹配 | 11 | 81.1 | 71.5 ~ 90.5 |

**两个区间是重叠的**（约 71~90 内既有认对也有认错）。

**这意味着没有一个阈值能干净地把两类分开**——这是 LBPH 这类经典方法的固有限制，
也正是现代人脸识别转向深度嵌入（FaceNet / ArcFace 等）的核心原因。
本项目如实记录这一点，而不是用一个凑出来的阈值宣称「准确率 100%」。

---

## 置信度阈值怎么调

`src/config.py` 里的 `UNKNOWN_THRESHOLD`（默认 **80**）决定多大差异算「未知人员」。
**LBPH 的置信度是距离，越小越像**，但这个值会随以下因素大幅变化：

| 因素 | 影响 |
|---|---|
| 人脸在画面中的大小 | 与训练尺度差得越多，置信度越高 |
| 背景 | 换背景会改变检测框大小，进而改变裁片内容 |
| 视频压缩 | 有损编码（mp4v / H.264）能让人脸置信度整体抬高 20~60 |
| 光照 | 本文数据集背景光较暗，换环境需重新评估 |

**调法**：用带标注的视频跑 `python recognize.py 视频 --truth <姓名>`，
观察输出的置信度分布，再据此设阈值。

---

## 短信告警配置（可选）

程序**不会把账号密码写在代码里**。要启用告警，先设置三个环境变量：

```bash
# Windows（当前终端有效）
set SMS_USER=你的账号
set SMS_PASSWORD=你的密码
set SMS_PHONE=接收告警的手机号
```

或复制 `.env.example` 为 `.env` 后填写（`.env` 已在 `.gitignore` 中排除）。

**不配置也能正常跑**，只是检测到未知人员时只打印、不发短信。

> ⚠️ 不要把账号密码写进 `.py` 文件再提交——一旦进入 Git 历史，
> 事后删掉也仍然能被翻出来。这正是本项目把凭据抽成环境变量的原因。

---

## 学习路径（`tutorials/`）

四个脚本按难度递进，都是**独立可运行**的，不依赖项目其它模块：

| 脚本 | 内容 | 前置 |
|---|---|---|
| `01_image_basics.py` | 读图、灰度、缩放、绘制图形 | 无 |
| `02_face_detection.py` | Haar 级联检测单张 / 多人脸 | 01 |
| `03_video_detection.py` | 视频逐帧检测 | 02 |
| `04_face_capture.py` | 从摄像头采集训练素材 | 03 |

跑完这一圈，就能理解 `train.py` / `recognize.py` 里每一步在做什么。

---

## 已知限制

1. **LBPH 对成像条件敏感**：人脸尺度、背景、光照、视频编码都会显著抬高置信度。
   本项目在演示视频上用无损编码 + 原始尺度，就是为了绕开这一点。
2. **不是生产级方案**：Haar 级联对侧脸、遮挡、大角度旋转的检出率会明显下降；
   LBPH 的判别力也远不如深度模型。本项目的定位是**理解流程与经典基线**。
3. **评测是「同一数据集内的留出集」**：93.8% 是 ORL 内部的结果，
   换成真实监控场景表现会下降。
4. **未做活体检测**：对着照片也可能识别通过，不适合门禁等安全场景。

---

## 数据集

公开数据集 **ORL / AT&T Database of Faces**（又称 Olivetti Faces）：

- 40 位被试，每人 10 张，共 400 张
- 92 × 112 像素 8 位灰度图
- 1992–1994 年采集，同一人的 10 张在光照 / 表情 / 是否戴眼镜上有差异

原始数据来自 [AT&T Laboratories Cambridge](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)，
本仓库已转成 JPG 并重命名。**文件名点号前的数字即训练标签**，
详见 [`data/README.md`](data/README.md)。

> 换成自己的人脸：把图片放进 `data/`，命名成 `1.任意名.jpg`、`2.任意名.jpg`
> （同一个人的所有图片用同一个数字），然后重跑 `python train.py`。
> 也可以直接用 `tutorials/04_face_capture.py` 从摄像头采集。

---

## 参考

- F. Samaria and A. Harter, *Parameterisation of a stochastic model for human face
  identification*, 2nd IEEE Workshop on Applications of Computer Vision, 1994.
- [OpenCV](https://opencv.org/) —— Haar 级联分类器与 LBPH 实现

## License

[MIT](LICENSE)
