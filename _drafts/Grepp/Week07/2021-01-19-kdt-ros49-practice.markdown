---
layout: post
title:  "[실습] 기본적인 OpenCV 기능 사용"
date:   2021-01-19 01:00:00 +0900
categories: "Grepp/KDT"
tags:
    - OpenCV
    - Python
---

## 예제 코드

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── ex_codes
        ├── opencv_ex
        └── rosbag_ex
{% endhighlight %}



## 사각형 그리기(rectangle.py)

{% highlight Python %}
import cv2

# cv2.imread(filename, flag): 이미지 파일을 읽기
#   flag: 이미지 파일을 읽을 때 옵션
img = cv2.imread("black.png", cv2.IMREAD_COLOR)

# cv2.rectangle(img, pt1, pt2, thickness): 사각형 그리기
#   pt1: 시작점
#   pt2: 끝점
#   thickness: 두께
img = cv2.rectangle(img, (100, 100), (300, 400), (0, 255, 0), 3)

# cv2.imshow(title, image): 화면에 이미지 파일 표시
#   title: 창의 제목
#   image: cv2.imread()의 return 값
cv2.imshow("black", img)
cv2.waitKey(10000)
{% endhighlight %}



## 이미지 파일을 읽고, 화면에 표시(girl.py)

{% highlight Python %}
import cv2

img = cv2.imread("girl.png", cv2.IMREAD_COLOR)

cv2.imshow("Girl", img)

cv2.waitKey(10000)
cv2.destoryAllWindows()
{% endhighlight %}



## 이미지에서 흰 점 한개 찾아내기(spot.py)

{% highlight Python %}
import cv2

img = cv2.imread("spot.png", cv2.IMREAD_GRAYSCALE)
h = img.shape[0]
w = img.shape[1]
print("The image dimension is %d x %d" % (w, h))

for i in range(h):
    for j in range(w):
        if img[i, j] == 255:
            print(i, j)

cv2.imshow("spot", img)
cv2.waitKey(10000)
{% endhighlight %}



## 관심 영역(ROI) 설정(roi.py)

{% highlight Python %}
import cv2

img = cv2.imread("cars.png")

cv2.imshow("car", img[120:270, 270:460])

cv2.waitKey(10000)
{% endhighlight %}



## 색상과 명도 범위로 차선 인식(hsv.py)

- 명도 범위를 조정해서 차선을 불리할 수 있다.
- 차선의 색상 또는 주변 밝기에 따라 범위 지정 값은 달라져야 한다.

{% highlight Python %}
import cv2
import numpy as np

img = cv2.imread("cars.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_white = np.array([0, 0, 150])
upper_white = np.array([179, 255, 255])

mask = cv2.inRange(hsv, lower_white, upper_white)

cv2.imshow("line", mask)
cv2.waitKey(10000)
{% endhighlight %}



## 카메라/동영상 파일 읽고 표시(video.py)

{% highlight Python %}
import cv2

# VideoCapture 객체 생성하여, 카메라 디바이스 또는 동영상 파일 열기
# 카메라 디바이스: cv2.VideoCapture(0)
# 동영상 파일: cv2.VideoCapture(filename)
vid = cv2.VideoCapture("small.avi")

while True:
    # 순환문을 반복하면서, frame을 읽고
    ret, frame = vid.read()
    if not ret:
        break
    
    # 읽은 프레임을 변환
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 프레임을 화면에 표시
    if ret:
        cv2.imshow("video", frame)
    if cv2.waitKey(1) > 0:
        break

# 영상 재생이 끝나면, VideoCapture 객체 해제
vid.release()
# 윈도우 닫기
cv2.distroyAllWindows()
{% endhighlight %}
