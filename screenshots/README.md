# screenshots/

README 里引用的图放在这里。

现在有两张动图，都是 `recognize.py --save` 导出视频之后转成的：

- `demo_recognition.gif` —— 认出登记过的人，绿框加姓名
- `demo_alert.gif` —— 陌生人累计到阈值，弹出报警条

想再补几张的话，可以跑：

```bash
python recognize.py --hold              # 播放完停住，方便截图
python tutorials/02_face_detection.py   # 单张 / 多人脸检测
python -m pytest tests/ -v              # 测试通过的样子
```

截完按同样方式写进 README 就行。截图时注意别把真实姓名、手机号、账号密码截进去。
