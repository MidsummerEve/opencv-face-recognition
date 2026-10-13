# data/ —— 数据集说明

本目录下的人脸图片来自公开数据集 **ORL / AT&T Database of Faces**（又称 Olivetti Faces）。

## 来源

- 原始出处：AT&T Laboratories Cambridge（原 Olivetti Research Laboratory）
- 官方页面：https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
- 官方下载：https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip

## 规格

| 项目 | 说明 |
|---|---|
| 人数 | 40 人 |
| 每人张数 | 10 张 |
| 合计 | 400 张 |
| 分辨率 | 92 × 112 像素 |
| 色彩 | 8 位灰度 |
| 采集时间 | 1992 年 4 月 – 1994 年 4 月 |

同一被试的 10 张照片是在不同时间拍的，光照、表情（睁眼/闭眼、笑/不笑）、
是否戴眼镜等条件都不一样。

## 文件命名规则

本仓库把原始 `.pgm` 转成了 `.jpg`，并重命名成：

```
<标签id>.s<被试编号两位>_<图片序号>.jpg
```

例如 `3.s03_7.jpg` 表示第 3 位被试的第 7 张照片。

**点号前面的数字就是训练时用的标签 id** —— 这是 `09数据训练.py` 里
`int(filename.split('.')[0])` 能正常工作的前提。如果你自己往这里放图片，
也必须遵守「数字.任意名称.扩展名」这个格式。

## 引用与致谢

使用时请注明来源：

> The ORL Database of Faces, AT&T Laboratories Cambridge.
> F. Samaria and A. Harter, "Parameterisation of a stochastic model for human face
> identification", 2nd IEEE Workshop on Applications of Computer Vision, December 1994.

数据集原始说明见同目录下的 `ORL_原始说明.txt`。
