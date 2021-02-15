---
layout: post
title:  "[실습] 명도차 기반 차선 인식"
date:   2021-01-20 01:00:00 +0900
categories: "Grepp/KDT"
tag: ROS
---

## 차선을 따라 주행하기

- USB 카메라와 OpenCV를 이용하여 차선을 인식하고, 인식된 차선을 따라 스스로 주행할 수 있는 자율주행
- 직선/곡선구간 주행



## 차선을 인식하여 운전

- 차선 주종 주행: 좌우 차선을 찾아내어 차선을 벗어나지 않게끔 주행하면 된다.
- 카메라 입력으로 획득한 영상에서 적절한 영역을 잘라내어, 이진화한다.
    1. 우선 바깥에서 중앙으로 가면서 흰색 점을 찾는다.
    2. 그 점 주위에 사각형을 쳐서 사각형 안에 있는 흰색 점의 개수를 구한다. 기준 개수 이상이면, 바로 거기가 차선이다.



## OpenCV 기반의 영상 처리

차선을 찾기 위한 작업

1. Image Read: 카메라 영상 신호를 이미지로 읽기
2. Grayscale: 흑백 이미지로 변환
3. Gaussian Blur: 노이즈 제거
4. HSV Binary: HSV 기반으로 이진화 처리
5. ROI: 관심 영역 잘라내기



## 차선 검출을 위한 영상 처리

gray.py: 컬러(bgr8) 이미지를 흑백(grayscale) 이미지로 변환

{% highlight Python %}
import cv2

img = cv2.imread("sample.png")

gray = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)
cv2.waitKey(10000)
{% endhighlight %}


blur.py: 차선 인식에 방해가 되는 노이즈 제거

{% highlight Python %}
import cv2

img = cv2.imread("sample.png")
gray = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

cv2.imshow("gray", gray)
cv2.waitKey(10000)
{% endhighlight %}


line.py: HSV 기반 이진화 방법으로 차선을 추출

{% highlight Python %}
import cv2
import numpy as np

img = cv2.imread("sample.png")
hsv = cv.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_white = np.array([0, 0, 70])
upper_white = np.array([131, 255, 255])

mask = cv2.inRange(hsv, lower_white, upper_white)

cv2.imshow("line", mask)
cv2.waitKey(10000)
{% endhighlight %}


canny.py: Canny Edge Detector 이용하여, 최곽선을 추출해서 차선을 찾을 수도 있음

{% highlight Python %}
import cv2

img = cv2.imread("sample.png")
gray = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

edge = cv2.Canny(blur, 20, 190)

cv2.imshow("edge", edge)
cv2.waitKey(10000)
{% endhighlight %}


nonzero.py: 사각형 안에 있는 흰색 점의 갯수를 세서 일정 갯수 이상이면 녹색으로 표시

{% highlight Python %}
import cv2
import numpy as np

img = cv2.imread("sample.png")
hsv = cv.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_white = np.array([0, 0, 70])
upper_white = np.array([131, 255, 255])

img = cv2.inRange(hsv, lower_white, upper_white)

xx = 20
while True:
    area = img[430:450, xx:xx+15]
    if cv2.countNonZero(area) > 200:
        image = cv2.rectangle(image, (xx, 430), (xx+15, 450), (0, 255, 0), 3)
    else:
        image = cv2.rectangle(image, (xx, 430), (xx+15, 450), (255, 0, 0), 3)
    xx = xx + 20
    if xx > 640:
        break
        
cv2.imshow("countNonZero", image)
cv2.waitKey(10000)
{% endhighlight %}
