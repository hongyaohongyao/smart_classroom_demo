# 智慧教室Demo

群体课堂专注度分析、考试作弊系统、动态点名功能的Qt Demo，使用多人姿态估计、情绪识别、人脸识别、静默活体检测等技术

[相关项目](https://github.com/hongyaohongyao/smart_classroom) 

## 项目环境

- Python 3.7
- PyQt5
- Pytorch1.8.1
- 更多可参考requirements.txt文件
- 人脸识别功能要使用gpu需要自己[编译gpu版的dlib](https://blog.csdn.net/qq_29168809/article/details/102655115) 
- 最好用有gpu的设备运行嗷，没有gpu可能需要自己在项目里改

## 使用步骤

### 1.下载 requirements.txt

```shell
pip install -r requirements.txt
```

### 2.下载权重文件

从[百度云（提取码：uk26）](https://pan.baidu.com/s/16av6CXWrCgGkniCwCc3qqQ) 下载smart_classroom_demo项目的权重文件放置到weights文件夹下。

### 3.运行smart_classroom_app.py

## 界面展示

### 作弊检测

视频是实时检测和播放的，可以选择视频文件或rtsp视频流作为视频源，视频通道下摄像头以外的选项在resource/video_sources.csv文件里设置。

![作弊检测界面](.img/README/作弊检测界面.jpg)

### 人脸注册

![人脸注册](.img/README/人脸注册.jpg)

静默活体检测，照片不能用来注册

![静默活体效果](.img/README/静默活体效果.jpg)

### 动态点名

学生面向摄像头完成签到，可以多人同时进行签到

![动态点名](.img/README/动态点名.jpg)