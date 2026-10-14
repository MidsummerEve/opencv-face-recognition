"""
未知人员短信告警。

设计原则：**凭据绝不写进代码**。
账号、密码、手机号一律从环境变量读取：

    SMS_USER      短信平台账号
    SMS_PASSWORD  短信平台密码（明文即可，内部会做 md5）
    SMS_PHONE     接收告警的手机号

未配置时 `send_alarm()` 直接静默返回，程序照常运行，只是不发短信。
这样仓库里既不会有敏感信息，别人 clone 下来也能直接跑。
"""

import hashlib
import os
import time
import urllib.parse
import urllib.request

# 短信平台（短信宝）接口
SMS_API = 'https://api.smsbao.com/'

# 两次短信之间的最短间隔（秒），避免告警轰炸
SMS_COOLDOWN = 60

# 平台返回的状态码
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

# 上一次成功发送的时间戳（秒），用于冷却判断
_last_sms_time = 0.0


def is_configured():
    """是否已配置好 SMS_* 环境变量。"""
    return all(os.environ.get(k) for k in
               ('SMS_USER', 'SMS_PASSWORD', 'SMS_PHONE'))


def _md5(text):
    """短信宝平台要求密码以 md5 形式提交。"""
    return hashlib.md5(text.encode('utf8')).hexdigest()


def send_alarm(reason='检测到未知人员', location='监控区域'):
    """发送未知人员告警短信。

    返回 True 表示发送成功，False 表示未发送（未配置 / 冷却中 / 发送失败）。
    """
    global _last_sms_time

    if not is_configured():
        return False

    now = time.time()
    if now - _last_sms_time < SMS_COOLDOWN:
        print('距离上次告警不足 %d 秒，本次不发送' % SMS_COOLDOWN)
        return False

    send_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    content = ('【人脸识别预警】\n'
               '原因：%s\n'
               '地点：%s\n'
               '时间：%s' % (reason, location, send_time))

    params = urllib.parse.urlencode({
        'u': os.environ['SMS_USER'],
        'p': _md5(os.environ['SMS_PASSWORD']),
        'm': os.environ['SMS_PHONE'],
        'c': content,
    })

    try:
        response = urllib.request.urlopen(SMS_API + 'sms?' + params, timeout=15)
        code = response.read().decode('utf-8').strip()
        print(SMS_STATUS.get(code, '未知状态码: %s' % code))
        if code == '0':
            _last_sms_time = now
            print('告警短信已发送（%s），进入 %d 秒冷却' % (send_time, SMS_COOLDOWN))
            return True
    except Exception as e:
        print('短信发送失败：%s' % e)

    return False
