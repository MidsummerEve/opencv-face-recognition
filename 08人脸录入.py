"""
人脸录入：从视频/摄像头逐帧保存图片，作为后续训练的原始素材。

用法：
    python 08人脸录入.py                  # 处理 samples/demo.avi
    python 08人脸录入.py 0                # 使用本机摄像头
    python 08人脸录入.py D:\\my_video.mp4  # 处理指定视频

操作：
    按 s 保存当前帧到 data/
    按空格 退出

注意：保存的文件名形如 1.sample_1.jpg，点号前面的数字就是训练时用的标签 id。
"""
import os
import sys

import cv2

# 图片保存目录（用相对路径，换台电脑也能跑）
SAVE_DIR = 'data'
os.makedirs(SAVE_DIR, exist_ok=True)

# 视频源
src = sys.argv[1] if len(sys.argv) > 1 else 'samples/demo.avi'
cap = cv2.VideoCapture(int(src) if src.isdigit() else src)

if not cap.isOpened():
    print('无法打开视频源：%s' % src)
    raise SystemExit(1)

print('正在读取：%s' % src)
print('按 s 保存当前帧，按空格退出')

num = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print('视频播放完毕')
        break

    cv2.imshow('Capture_Test', frame)

    k = cv2.waitKey(10) & 0xFF
    if k == ord('s'):
        filename = os.path.join(SAVE_DIR, '%d.sample_%d.jpg' % (num, num))
        cv2.imwrite(filename, frame)
        print('已保存：%s' % filename)
        num += 1
    elif k == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
