"""
人脸识别与陌生人预警主程序。

流程：人脸检测 -> LBPH 识别 -> 已知人员放行 / 陌生人累计 -> 触发报警

判定逻辑（这是预警系统的核心）
------------------------------
    识别出【已知人员】  -> 绿框 + 姓名，并把陌生人计数清零
    识别出【陌生人】    -> 红框 + unknown，陌生人计数 +1
    计数累计到 UNKNOWN_TRIGGER_FRAMES -> 触发报警（发短信）

注意计的是次数而不是「连续帧」：每检测到一次陌生人就 +1，
只有识别出已知人员才清零。这样有人反复快速经过也无法把计数「冲掉」。

用法：
    python recognize.py                    # 已知人员演示视频
    python recognize.py samples/demo_stranger.avi
                                           # 陌生人演示视频（能看到累计与报警）
    python recognize.py 0                  # 本机摄像头
    python recognize.py --save outputs/out.avi   # 导出带标注的结果视频
    python recognize.py --no-display       # 不弹窗，只看统计

前置：
    先运行 train.py 训练。训练只使用「已知人员」，
    config.HOLDOUT_SUBJECTS 里的留出身份不参与训练，因此会被判为陌生人。

关于短信告警（可选）：
    账号密码走环境变量 SMS_USER / SMS_PASSWORD / SMS_PHONE，
    未配置时程序照常运行，只是不发短信。详见 src/alarm.py。
"""

import argparse
import os

import cv2

from src import alarm, recognizer
from src.config import UNKNOWN_THRESHOLD, UNKNOWN_TRIGGER_FRAMES
from src.detector import create_detector, crop_faces, detect_faces

# samples/、trainer/、outputs/ 这些路径都相对项目根目录。把工作目录切到脚本
# 所在目录，这样在 PyCharm 里运行时不依赖它的「工作目录」设置——PyCharm 默认
# 用的是外层工程目录，不切的话会去找 <工程目录>/samples，报找不到视频源。
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 颜色（注意 OpenCV 用的是 BGR 顺序，不是 RGB）
COLOR_KNOWN = (0, 200, 0)      # 绿：已知人员
COLOR_UNKNOWN = (0, 0, 230)    # 红：陌生人
COLOR_TEXT = (255, 255, 255)

# 画面最大宽度，超过则等比缩小（None 表示不缩放）
MAX_WIDTH = 1200
# 标注时的最小宽度：画面比这个小就先放大再画框写字。
#
# 识别本身仍在【原分辨率】上做（与训练时的检测口径保持一致），
# 只有「画框、写字」这一步放到放大后的画布上——
# 否则像 92x112 这种小画面，检测框和状态栏文字会小到根本看不清。
# 放大用 INTER_NEAREST，保证像素一一对应，不引入插值模糊。
ANNOTATE_MIN_WIDTH = 480
# 默认视频源
DEFAULT_SOURCE = 'samples/demo.avi'
# 触发报警后，画面上提示保留多少帧
ALERT_BANNER_FRAMES = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description='基于 OpenCV + LBPH 的人脸识别与陌生人预警')
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
        help='把带标注的结果视频保存下来（例如 outputs/out.avi），之后可反复观看')
    parser.add_argument(
        '--hold', action='store_true',
        help='播放结束后保持最后一帧，按任意键才关闭窗口')
    return parser.parse_args()


def draw_hud(frame, known, unknown, alerts):
    """在画面顶部画一条状态栏：已知识别数 / 陌生人累计 / 报警次数。

    状态栏宽度按文字实际宽度算，避免在小画面上被截断。
    """
    text = 'known:%d  strangers:%d/%d  alerts:%d' % (
        known, unknown, UNKNOWN_TRIGGER_FRAMES, alerts)
    font, fscale, fthick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, fscale, fthick)
    bar_w = min(tw + 14, frame.shape[1])
    bar_h = th + 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (bar_w, bar_h), (32, 32, 32), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, text, (6, th + 6), font, fscale, COLOR_TEXT, fthick)
    return bar_h


def print_report(stats, truth):
    total = stats['total']
    print('\n===== 识别统计 =====')
    if total == 0:
        print('未检测到任何人脸，无统计数据')
        return
    print('检测到人脸共 %d 次' % total)
    print('  已知人员：%d 次' % stats['known'])
    print('  陌生人：  %d 次' % stats['unknown'])
    print('触发报警：  %d 次' % stats['alerts'])
    if truth:
        print()
        print('（--truth %s 下的精度口径）' % truth)
        print('正确识别率：%.2f%%' % (stats['correct'] / total * 100))
        print('误识别率：  %.2f%%' % (stats['wrong'] / total * 100))
        print('拒识率：    %.2f%%' % (stats['reject'] / total * 100))
    else:
        print()
        print('（加 --truth <姓名> 可输出正确率 / 误识别率 / 拒识率）')


def main():
    args = parse_args()

    try:
        model, names = recognizer.load()
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    print('已加载模型，已知身份 %d 个' % len(names))
    if not alarm.is_configured():
        print('提示：未配置 SMS_* 环境变量，报警只打印不发短信')

    detector = create_detector()

    source = args.source
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit('无法打开视频源：%s' % source)

    # 按源帧率播放。只写死 waitKey(10) 的话，因为检测本身要 20~40ms，
    # 视频会被快放一倍以上，窗口一闪就过去了。
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0 or source_fps > 120:
        source_fps = 25.0
    frame_delay = max(1, int(round(1000.0 / source_fps)))

    # 标注缩放系数：画面太小就先放大再画框写字（识别本身仍在原分辨率上做）
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    annotate_scale = 1.0
    if ANNOTATE_MIN_WIDTH and src_w and src_w < ANNOTATE_MIN_WIDTH:
        annotate_scale = ANNOTATE_MIN_WIDTH / src_w
    out_size = (int(round(src_w * annotate_scale)),
                int(round(src_h * annotate_scale)))

    writer = None
    if args.save:
        parent = os.path.dirname(args.save)
        if parent:
            os.makedirs(parent, exist_ok=True)
        ext = os.path.splitext(args.save)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*(('mp4v') if ext == '.mp4' else ('MJPG')))
        writer = cv2.VideoWriter(args.save, fourcc, source_fps, out_size)
        if not writer.isOpened():
            raise SystemExit('无法创建输出视频：%s' % args.save)
        print('结果将保存到 %s（%dx%d）' % (args.save, out_size[0], out_size[1]))

    print("程序启动成功，按 'q' 退出")

    stats = {'total': 0, 'known': 0, 'unknown': 0, 'alerts': 0,
             'correct': 0, 'wrong': 0, 'reject': 0}
    unknown_count = 0          # 陌生人累计次数（每检测到一次 +1）
    alert_banner = 0           # 报警提示还要显示多少帧

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print('视频播放完毕 / 摄像头断开')
            break
        if frame.size == 0:
            continue

        if MAX_WIDTH and frame.shape[1] > MAX_WIDTH:
            scale = MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray)

        # 第一步：判定。全部在【原分辨率】上做，与训练时的检测口径一致。
        results = []          # [(x, y, w, h, color, caption), ...]
        for (x, y, w, h), face in zip(faces, crop_faces(gray, faces)):
            label, distance = recognizer.predict(model, face)
            stats['total'] += 1

            # distance 是「距离」：越小越像，超过阈值判为陌生人
            if distance <= UNKNOWN_THRESHOLD and label in names:
                name = names[label]
                color = COLOR_KNOWN
                unknown_count = 0                 # 看到已知人员，计数清零
                stats['known'] += 1
                if args.truth:
                    if name == args.truth:
                        stats['correct'] += 1
                    else:
                        stats['wrong'] += 1
                caption = '%s  d=%.0f' % (name, distance)
            else:
                color = COLOR_UNKNOWN
                unknown_count += 1                # 陌生人累计 +1
                stats['unknown'] += 1
                if args.truth:
                    stats['reject'] += 1
                caption = 'stranger  %d/%d' % (unknown_count,
                                               UNKNOWN_TRIGGER_FRAMES)

                if unknown_count >= UNKNOWN_TRIGGER_FRAMES:
                    stats['alerts'] += 1
                    alert_banner = ALERT_BANNER_FRAMES
                    print('[报警] 陌生人累计 %d 次，触发预警' % unknown_count)
                    alarm.send_alarm()
                    unknown_count = 0            # 报警后重新计数

            results.append((x, y, w, h, color, caption))

        # 第二步：画图。画面太小就先放大，否则框和状态栏文字会看不清。
        canvas = frame
        if annotate_scale != 1.0:
            canvas = cv2.resize(frame, None, fx=annotate_scale,
                                fy=annotate_scale,
                                interpolation=cv2.INTER_NEAREST)

        for (x, y, w, h, color, caption) in results:
            X, Y = int(x * annotate_scale), int(y * annotate_scale)
            W, H = int(w * annotate_scale), int(h * annotate_scale)
            cv2.rectangle(canvas, (X, Y), (X + W, Y + H), color, 2)
            cv2.putText(canvas, caption, (X + 6, max(Y - 8, 36)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        hud_h = draw_hud(canvas, stats['known'], unknown_count, stats['alerts'])
        if alert_banner > 0:
            alert_banner -= 1
            y0 = hud_h + 2
            cv2.rectangle(canvas, (0, y0), (canvas.shape[1], y0 + 24),
                          (0, 0, 200), -1)
            cv2.putText(canvas, '!! ALERT TRIGGERED !!', (8, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2)

        if writer is not None:
            writer.write(canvas)

        if not args.no_display:
            cv2.imshow('result', canvas)
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

    print_report(stats, args.truth)


if __name__ == '__main__':
    main()
