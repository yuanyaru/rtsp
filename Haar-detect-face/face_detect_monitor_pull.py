# 哈尔（Haar）级联法：专门解决人脸识别而推出的传统算法
import cv2


def videocapture():
	# Read the monitor
	# cap = cv2.VideoCapture(0)
	cap = cv2.VideoCapture('rtsp://admin:Admin123@192.168.2.148:554/Streaming/Channels/101')
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
	fourcc = 0x21  # 视频的编码
	fps = 25  # 视频的帧率
	output_file = 'test_video.mp4'
	size = (width, height)
	saver = cv2.VideoWriter(output_file, fourcc, fps, size)

	# Create the haar cascade
	cascPath = "haarcascade_frontalface_default.xml"
	# CascadeClassifier，是Opencv中做人脸检测的时候的一个级联分类器
	faceCascade = cv2.CascadeClassifier(cascPath)

	while cap.isOpened():
		ret, frame = cap.read()  # 读取摄像头画面
		if not ret:
			print("摄像头连接失败！")
			continue
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图
		# Detect faces in the monitor
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30),
		)

		print("Found {0} faces!".format(len(faces)))

		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		cv2.imshow('frame', frame)  # 显示画面
		saver.write(frame)  # 视频保存

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()  # 释放摄像头q


if __name__ == '__main__':
	videocapture()
