# 哈尔（Haar）级联法：专门解决人脸识别而推出的传统算法
import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
# Read the image
image = cv2.imread(imagePath)

# Create the haar cascade
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

while True:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        # 输入图像，灰度图运算速度更快
        gray,
        # 每次循环增大的比例
        scaleFactor=1.1,
        # 表示构成检测目标的相邻矩形的最小个数(默认为3个)，数值越大人脸被筛选的难度越大，同样也越精确
        minNeighbors=5,
        # 小于这个范围的矩形，就不被认定为检测目标
        minSize=(30, 30),
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)
