import cv2
import subprocess as sp


def videocapture():
    # filePath = 'E:\\git-workspace\\face_check\\Haar-detect-face\\'
    # cap = cv2.VideoCapture(filePath+"test_video.mp4")  # 从文件读取视频

    # 拉流 url 地址
    pull_url = 0
    # pull_url = "rtsp://admin:Admin123@192.168.2.148:554/Streaming/Channels/101"
    cap = cv2.VideoCapture(pull_url)  # 调用摄像头的rtsp协议流

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("width", width, "height", height, "fps：", fps)

    # 推流 url 地址
    push_url = 'rtsp://127.0.0.1:8554/stream'

    # ffmpeg推送rtsp,通过管道共享数据的方式
    command = ['D:\\install_package\\ffmpeg-5.1.2-essentials_build\\ffmpeg-5.1.2-essentials_build\\bin\\ffmpeg.exe',
               '-y', '-an',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',  # 像素格式
               '-s', "{}x{}".format(width, height),
               '-r', str(fps),
               '-i', '-',
               '-c:v', 'libx264',  # 视频编码方式
               '-pix_fmt', 'yuv420p',
               '-preset', 'ultrafast',
               '-f', 'rtsp',
               '-rtsp_transport', 'tcp',
               push_url]
    # '-i','E:\\git-workspace\\face_check\\Haar-detect-face\\test_video.mp4',

    pipe = sp.Popen(command, stdin=sp.PIPE, shell=False)  # 管道特性配置
    # Create the haar cascade
    cascPath = "haarcascade_frontalface_default.xml"
    # CascadeClassifier，是Opencv中做人脸检测的时候的一个级联分类器
    faceCascade = cv2.CascadeClassifier(cascPath)

    while True:
        ret, frame = cap.read()  # 逐帧采集视频流

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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('frame', frame)  # 显示画面
        pipe.stdin.write(frame.tobytes())  # 存入管道用于直播

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    cap.release()
    print("Over!")


if __name__ == '__main__':
    videocapture()
