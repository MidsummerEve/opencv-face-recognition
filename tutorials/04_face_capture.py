"""
人脸录入：从视频 / 摄像头逐帧保存图片，作为后续训练的原始素材。

学习路径的最后一站。做完这一步，你就有了自己的数据集，
可以拿去跑 train.py 训练、再用 recognize.py 识别。

运行（务必在项目根目录下执行）：
    python tutorials/04_face_capture.py                  # 处理自带示例视频
    python tutorials/04_face_capture.py 0                # 用本机摄像头（推荐）
    python tutorials/04_face_capture.py D:\\my_video.mp4   # 处理指定视频

操作：
    按 s    保存当前帧
    按空格  退出

命名的讲究（很重要）
--------------------
保存的文件名形如  1.sample_1.jpg
                  ↑
                  点号前的数字就是训练时的标签 id

同一个人录十张，文件名都用同一个数字开头，例如 1.a.jpg、1.b.jpg ……
后一个人换成 2.xxx.jpg。train.py 就是靠点号前的数字来区分身份的。

（如果只是练习，data/ 里已经有 ORL 公开数据集；自己录的图会混在里面，
  但不影响训练——它只是多了几个身份。想清空就删掉 .sample_ 开头的文件。）
"""

import os
import sys

import cv2

# 保存目录（相对路径，换台电脑也能用）
SAVE_DIR = 'data'
os.makedirs(SAVE_DIR, exist_ok=True)

# 视频源
source = sys.argv[1] if len(sys.argv) > 1 else 'samples/demo.avi'
cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
if not cap.isOpened():
    raise SystemExit('无法打开视频源：%s' % source)

print('正在读取：%s' % source)
print("按 s 保存当前帧，按空格退出")

# 标签 id：你要录入的是第几个人就改成几。同一个人录多张时保持不变。
label = 1
num = 1

while cap.isOpened():
    ok, frame = cap.read()
    if not ok or frame is None:
        print('视频播放完毕')
        break

    # 在画面上提示当前标签，避免存错人
    shown = frame.copy()
    cv2.putText(shown, 'label=%d  saved=%d' % (label, num - 1), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Capture', shown)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('s'):
        filename = os.path.join(SAVE_DIR, '%d.sample_%d.jpg' % (label, num))
        cv2.imwrite(filename, frame)
        print('已保存：%s' % filename)
        num += 1
    elif key == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
print('共保存 %d 张' % (num - 1))
