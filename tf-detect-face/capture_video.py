# -*- coding: utf-8 -*-
"""
@Time ： 2022/8/29 11:25
@Auth ： yuanyr
@File ：capture_video.py
@IDE ：PyCharm
使用OpenCV的Python接口获取网络摄像头视频并保存
"""

import time
import cv2

# your camera's rtsp url
RTSP_URL = 'rtsp://admin:Admin123@192.168.2.214:554/Streaming/Channels/101'
DURATION = 30   # how many time in seconds you want to capture
OUTPUT_FILE = 'capture_video.mp4'

cap = cv2.VideoCapture(RTSP_URL)
# fourcc = cv2.VideoWriter_fourcc('h','2','6','4')
fourcc = 0x21
# fps = cap.get(cv2.CAP_PROP_FPS)
fps = 25
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
saver = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, size)
got_first_frame = False
frame_count = 0

while True:
    ret, frame = cap.read()
    if not(ret):
        continue
    frame_count += 1
    print("%s: frame %d received" % (time.time(), frame_count))
    if got_first_frame == False:
        start_time = time.time()
        got_first_frame = True

    saver.write(frame)
    now = time.time()
    if int(now - start_time) > DURATION:
        break