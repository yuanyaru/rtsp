# -*- coding: utf-8 -*-
"""
@Time ： 2022/9/8 9:44
@Auth ： yuanyr
@File ：yolo.py
@IDE ：PyCharm
python使用Yolov3实现电子围栏功能，检测目标是否进入指定区域
"""


import cv2 as cv
import numpy as np
from matplotlib.path import Path
import pyautogui


def yolo_detect(img):
    # 这里的m，n是将4个坐标点顺序连起来组成的四边形所围成的区域
    m = Path([(0, 338), (869, 333), (0, 443), (870, 425)])  # 警报区域
    n = Path([(968, 319), (1522, 341), (1521, 530), (958, 469)])
    confidence_thre = 0.5  # 置信度（概率/打分）阈值，即保留概率大于这个值的边界框，默认为0.5
    nms_thre = 0.3  # 非极大值抑制的阈值，默认为0.3
    LABELS = open('coco.names').read().strip().split("\n")  # 加载类别标签文件
    (H, W) = img.shape[:2]  # 获取图片维度
    net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')  # 加载模型配置和权重文件
    ln = net.getLayerNames()  # 获取YOLO输出层的名字
    # 如果这里报错的话，请把i[0]的[0]去掉，变成i
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True,
                                crop=False)  # 将图片构建成一个blob，设置图片尺寸，然后执行一次  YOLO前馈网络计算，最终获取边界框和相应概率
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []  # 初始化边界框，置信度（概率）以及类别
    confidences = []
    classIDs = []
    i = 0

    for output in layerOutputs:  # 迭代每个输出层，总共三个
        for detection in output:  # 迭代每个检测
            scores = detection[5:]  # 提取类别ID和置信度
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_thre:  # 只保留置信度大于某值的边界框
                box = detection[0:4] * np.array([W, H, W, H])  # 将边界框的坐标还原至与原图片相匹配，返回边界框的中心坐标以及边界框的宽度和高度
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])  # 更新边界框，置信度（概率）以及类别
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)  # 使用非极大值抑制方法抑制弱、重叠边界框
    if len(idxs) > 0:  # 确保至少一个边界框
        for i in idxs.flatten():  # 迭代每个边界框
            color = (255, 0, 0)
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # 报警条件
            if (m.contains_point((int(x + w / 2), int(y + h / 2))) or n.contains_point(
                    (int(x + w / 2), int(y + h / 2)))) and (LABELS[classIDs[i]] == 'person'):
                color = (0, 0, 255)
                # m.contain_point（x，y）可以判断点（x，y）是否在m区域内
                cv.putText(img, "Catch the thief!", (680, 425), cv.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 5)  # 警报信息
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)  # 绘制边界框以及添加类别标签和置信度
            text = '{}: {:.3f}'.format(LABELS[classIDs[i]], confidences[i])
            (text_w, text_h), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), color, -1)
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return img


def main():
    # 注意更改视频路径
    cap = cv.VideoCapture('sp1.mp4')
    i = 0
    while True:
        success, frame = cap.read()  # 读取视频流
        if success:
            if i % 1 == 0:  # 每隔固定帧处理一次
                frame = yolo_detect(frame)
                cv.namedWindow('asd', cv.WINDOW_NORMAL)
                cv.imshow('asd', frame)
            i += 1
            key = cv.waitKey(5) & 0xFF  # 手动停止方法

            if key == ord('q'):
                print('停止播放')
                break
        else:
            print('播放完成')
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
