"""
人脸识别主程序：人脸检测 -> LBPH 识别 -> 未知人员短信告警。

用法：
    python recognize.py                          # 默认处理 samples/demo.avi
    python recognize.py 0                        # 使用本机摄像头
    python recognize.py D:\\my_video.mp4          # 处理指定视频
    python recognize.py samples/demo.avi --truth person01
                                                 # 追加精度统计（需指定画面里的人是谁）
    python recognize.py samples/demo.avi --no-display
                                                 # 不弹窗口（无图形界面 / 只想看统计）

前置：
    先运行 train.py 生成 trainer/trainer.yml

关于短信告警（可选功能）：
    程序不会把账号密码写进代码。要启用告警，先设置这三个环境变量：

        SMS_USER      短信平台账号
        SMS_PASSWORD  短信平台密码（明文即可，内部会做 md5）
        SMS_PHONE     接收告警的手机号

    没配置时程序照常运行，只是不会发短信。详见 src/alarm.py。
"""

import argparse
import os

import cv2

from src import alarm, recognizer
from src.config import FACE_SIZE, UNKNOWN_THRESHOLD
from src.detector import create_detector, crop_faces, detect_faces

# 连续多少帧检测到未知人员后才触发告警（避免偶发误检就发短信）
UNKNOWN_TRIGGER_FRAMES = 70
# 画面最大宽度，超过则等比缩小（None 表示不缩放）
MAX_WIDTH = 1200
# 显示时的最小宽度：画面比这个小就放大后再显示，纯为看得清楚，不影响识别结果
DISPLAY_MIN_WIDTH = 480
# 默认视频源
DEFAULT_SOURCE = 'samples/demo.avi'


def parse_args():
    parser = argparse.ArgumentParser(
        description='基于 OpenCV + LBPH 的人脸识别与未知人员告警')
    parser.add_argument(
        'source', nargs='?', default=DEFAULT_SOURCE,
        help='视频源：文件路径，或 0 表示本机摄像头（默认 %s）' % DEFAULT_SOURCE)
    parser.add_argument(
        '--truth', default=None,
        help='画面中真实身份的姓名。指定后会输出正确率 / 误识别率 / 拒识率')
    parser.add_argument(
        '--no-display', action='store_true',
        help='不弹出显示窗口，只输出统计（服务器或无图形界面时使用）')
    parser.add_argument(
        '--save', metavar='输出文件', default=None,
        help='把带标注的结果视频保存下来（例如 outputs/demo_out.avi），'
             '之后可以反复观看')
    parser.add_argument(
        '--hold', action='store_true',
        help='播放结束后保持最后一帧，按任意键才关闭窗口（便于观察结果）')
    return parser.parse_args()


def print_report(total, correct, wrong, reject, truth):
    print('\n===== 识别统计 =====')
    if total == 0:
        print('未检测到任何人脸，无统计数据')
        return
    print('总识别次数：%d' % total)
    if not truth:
        print('（加 --truth <姓名> 可输出正确率 / 误识别率 / 拒识率）')
        return
    print('真实身份：%s' % truth)
    print('正确识别率：%.2f%%' % (correct / total * 100))
    print('误识别率：  %.2f%%' % (wrong / total * 100))
    print('拒识率：    %.2f%%' % (reject / total * 100))


def main():
    args = parse_args()

    # 加载模型
    try:
        model, names = recognizer.load()
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    print('已加载模型，已知身份 %d 个' % len(names))
    if not alarm.is_configured():
        print('提示：未配置 SMS_* 环境变量，未知人员告警只打印不发短信')

    detector = create_detector()

    source = args.source
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit('无法打开视频源：%s' % source)

    # 按源帧率播放。
    # 如果只写死 waitKey(10)，因为人脸检测本身就要 20~40 毫秒，
    # 每帧实际耗时远大于 10 毫秒，视频会被「快放」一倍以上，窗口一闪就过去了。
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0 or source_fps > 120:
        source_fps = 25.0
    frame_delay = max(1, int(round(1000.0 / source_fps)))

    # 可选：把带标注的结果写成视频文件，便于反复观看
    writer = None
    if args.save:
        parent = os.path.dirname(args.save)
        if parent:
            os.makedirs(parent, exist_ok=True)
        ext = os.path.splitext(args.save)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*(('mp4v') if ext == '.mp4' else ('MJPG')))
        writer = cv2.VideoWriter(
            args.save, fourcc, source_fps,
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if not writer.isOpened():
            raise SystemExit('无法创建输出视频：%s' % args.save)
        print('结果将保存到 %s' % args.save)

    print("程序启动成功，按 'q' 退出")

    total = correct = wrong = reject = 0
    unknown_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print('视频播放完毕 / 摄像头断开')
            break
        if frame.size == 0:
            continue

        # 等比缩小到 MAX_WIDTH 以内（固定尺寸拉伸会让画面变形）
        if MAX_WIDTH and frame.shape[1] > MAX_WIDTH:
            scale = MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray)

        for (x, y, w, h), face in zip(faces, crop_faces(gray, faces)):
            label, distance = recognizer.predict(model, face)
            total += 1

            # distance 是「距离」：越小越像，超过阈值判为未知人员
            if distance <= UNKNOWN_THRESHOLD and label in names:
                name = names[label]
                unknown_frames = 0
                if args.truth:
                    if name == args.truth:
                        correct += 1
                    else:
                        wrong += 1
            else:
                name = 'unknown'
                unknown_frames += 1
                if args.truth:
                    reject += 1
                if unknown_frames > UNKNOWN_TRIGGER_FRAMES:
                    alarm.send_alarm()
                    unknown_frames = 0

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), 1)
            cv2.putText(frame, name, (x + 10, max(y - 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # 保存带标注的一帧（按原始分辨率，不放大）
        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            # 画面太小就放大后再显示，纯为看得清楚，不影响识别结果
            disp = frame
            if DISPLAY_MIN_WIDTH and frame.shape[1] < DISPLAY_MIN_WIDTH:
                s = DISPLAY_MIN_WIDTH / frame.shape[1]
                disp = cv2.resize(frame, None, fx=s, fy=s,
                                  interpolation=cv2.INTER_NEAREST)
            cv2.imshow('result', disp)
            # 用按源帧率算出的延时，而不是写死的 10ms
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print('带标注的结果已保存到 %s' % args.save)

    if not args.no_display:
        if args.hold:
            print('播放结束 —— 按任意键关闭窗口')
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    print_report(total, correct, wrong, reject, args.truth)


if __name__ == '__main__':
    main()
