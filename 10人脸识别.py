"""
人脸识别主程序：人脸检测 -> LBPH 识别 -> 未知人员短信告警。

用法：
    python 10人脸识别.py                    # 默认处理 samples/demo.mp4
    python 10人脸识别.py 0                  # 使用本机摄像头
    python 10人脸识别.py D:\\my_video.mp4    # 处理指定视频
    python 10人脸识别.py samples/demo.mp4 --truth person01
                                           # 追加精度统计（需指定画面里的人是谁）

前置：
    先运行 09数据训练.py 生成 trainer/trainer.yml

关于短信告警（可选功能）：
    程序不会把账号密码写在代码里。要启用告警，先在系统里设置这三个环境变量：
        SMS_USER      短信平台账号
        SMS_PASSWORD  短信平台密码（明文即可，程序内部会做 md5）
        SMS_PHONE     接收告警的手机号
    没配置时程序照常运行，只是不会发短信。
"""
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import cv2

# ============================ 可调参数 ============================
# 置信度阈值：LBPH 的特性是「值越小越像」，超过该值判为未知人员。
# 这个值对结果影响很大，且随人脸尺寸、背景、编码方式变化，需要按自己的数据调。
# 参考：用本仓库的演示视频（原尺度、无损编码）时正确匹配的置信度接近 0；
#       换成有损压缩或改变了人脸在画面中的大小后，置信度会升到 60~120。
#       详细评测见 README。
UNKNOWN_THRESHOLD = 80
# 连续多少帧检测到未知人员后才触发告警（避免偶发误检就发短信）
UNKNOWN_TRIGGER_FRAMES = 70
# 两次短信之间的最短间隔（秒）
SMS_COOLDOWN = 60
# 画面最大宽度，超过则等比缩小（None 表示不缩放）
MAX_WIDTH = 1200
# 显示时的最小宽度：画面比这个小就放大后再显示，纯为看得清楚，不影响识别结果
DISPLAY_MIN_WIDTH = 480

# 人脸裁片统一缩放到这个尺寸，必须与 09数据训练.py 里的 FACE_SIZE 一致
FACE_SIZE = (100, 100)

# 短信平台（短信宝）接口与状态码
SMS_API = 'https://api.smsbao.com/'
SMS_STATUS = {
    '0': '短信发送成功',
    '-1': '参数不全',
    '-2': '服务器空间不支持，请确认支持 curl 或 fsocket',
    '30': '密码错误',
    '40': '账号不存在',
    '41': '余额不足',
    '42': '账号已过期',
    '43': 'IP 地址限制',
    '50': '内容含有敏感词',
}

TRAINER_PATH = os.path.join('trainer', 'trainer.yml')
LABELS_PATH = os.path.join('trainer', 'labels.json')


# ============================ 短信告警 ============================
def md5(text):
    """短信宝平台要求密码以 md5 形式提交"""
    return hashlib.md5(text.encode('utf8')).hexdigest()


last_sms_time = 0


def send_alarm():
    """检测到未知人员时发送短信告警。未配置环境变量则静默跳过。"""
    global last_sms_time

    user = os.environ.get('SMS_USER', '')
    password = os.environ.get('SMS_PASSWORD', '')
    phone = os.environ.get('SMS_PHONE', '')
    if not (user and password and phone):
        return  # 未配置，直接跳过

    now = time.time()
    if now - last_sms_time < SMS_COOLDOWN:
        print('距离上次发送不足 %d 秒，本次不发送' % SMS_COOLDOWN)
        return

    send_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    content = ('【人脸识别预警】\n'
               '原因：检测到未知人员\n'
               '地点：监控区域\n'
               '时间：%s' % send_time)

    data = urllib.parse.urlencode({
        'u': user, 'p': md5(password), 'm': phone, 'c': content,
    })
    try:
        response = urllib.request.urlopen(SMS_API + 'sms?' + data, timeout=15)
        code = response.read().decode('utf-8').strip()
        print(SMS_STATUS.get(code, '未知状态码: %s' % code))
        if code == '0':
            last_sms_time = now
            print('短信已发送（%s），进入 %d 秒冷却' % (send_time, SMS_COOLDOWN))
    except Exception as e:
        print('短信发送失败：%s' % e)


# ============================ 姓名映射 ============================
def load_names():
    """加载 09数据训练.py 生成的标签映射；没有则用 personXX 兜底"""
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, encoding='utf-8') as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


# ============================ 识别主逻辑 ============================
def main():
    global unknown_frames

    # 解析命令行：第 1 个位置参数是视频源，--truth 指定画面里真实的人
    args = [a for a in sys.argv[1:]]
    truth = None
    if '--truth' in args:
        i = args.index('--truth')
        if i + 1 < len(args):
            truth = args[i + 1]
            del args[i:i + 2]
    src = args[0] if args else 'samples/demo.avi'

    # 加载模型
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists(TRAINER_PATH):
        recognizer.read(TRAINER_PATH)
        print('已加载模型：%s' % TRAINER_PATH)
    else:
        print('未找到模型 %s，请先运行 09数据训练.py' % TRAINER_PATH)
        raise SystemExit(1)

    names = load_names()
    print('已知身份 %d 个' % len(names))
    if not (os.environ.get('SMS_USER') and os.environ.get('SMS_PASSWORD')
            and os.environ.get('SMS_PHONE')):
        print('提示：未配置 SMS_* 环境变量，未知人员告警只打印不出短信')

    # 与 09数据训练.py 使用同一个级联文件，保证检测口径一致
    face_detector = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt2.xml'))

    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        print('无法打开视频源：%s' % src)
        raise SystemExit(1)
    print("程序启动成功，按 'q' 退出")

    total = correct = wrong = reject = 0
    unknown_frames = 0

    while True:
        flag, frame = cap.read()
        if not flag or frame is None:
            print('视频播放完毕 / 摄像头断开')
            break
        if frame.size == 0:
            continue

        # 等比缩小到 MAX_WIDTH 以内（原来的固定 (1200,800) 会拉伸变形）
        if MAX_WIDTH and frame.shape[1] > MAX_WIDTH:
            scale = MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), 1)

            label, confidence = recognizer.predict(
                cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE,
                           interpolation=cv2.INTER_AREA))
            total += 1

            if confidence <= UNKNOWN_THRESHOLD and label in names:
                name = names[label]
                unknown_frames = 0
                if truth:
                    if name == truth:
                        correct += 1
                    else:
                        wrong += 1
            else:
                name = 'unknown'
                unknown_frames += 1
                if truth:
                    reject += 1
                if unknown_frames > UNKNOWN_TRIGGER_FRAMES:
                    send_alarm()
                    unknown_frames = 0

            cv2.putText(frame, name, (x + 10, max(y - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # 画面太小的话放大后再显示，纯为看得清楚，不影响识别结果
        disp = frame
        if DISPLAY_MIN_WIDTH and frame.shape[1] < DISPLAY_MIN_WIDTH:
            s = DISPLAY_MIN_WIDTH / frame.shape[1]
            disp = cv2.resize(frame, None, fx=s, fy=s,
                              interpolation=cv2.INTER_NEAREST)

        cv2.imshow('result', disp)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # ---------- 统计结果 ----------
    print('\n===== 识别统计 =====')
    if total == 0:
        print('未检测到任何人脸，无统计数据')
    elif not truth:
        print('总识别次数：%d' % total)
        print('（加 --truth <姓名> 可输出正确率/误识别率/拒识率）')
    else:
        print('总识别次数：%d' % total)
        print('真实身份：%s' % truth)
        print('正确识别率：%.2f%%' % (correct / total * 100))
        print('误识别率：  %.2f%%' % (wrong / total * 100))
        print('拒识率：    %.2f%%' % (reject / total * 100))


if __name__ == '__main__':
    main()
