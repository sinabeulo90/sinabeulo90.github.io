---
layout: post
title:  "영상처리 기반 신호등 인식"
date:   2021-02-17 01:00:00 +0900
category: "Grepp/KDT"
tag: "Image Processing"
---

## Color Space 다루기

### Color Space
- RGB 색상 공간의 특성
    - 조명에 따른 색상 표현의 차이
    - RGB 색상은 Red, Green, Blue 값의 선형 조합에 의해 얻어짐
    - 문제점: 조명과 그림자에 따른 지각 불균일성
- HSV 색상 공간의 특성
    - 색상(Hue): 색 종류, 노란색, 빨간색, 파란색, ...
    - 채도(Saturation): 선명도, 원색에 가까울수록 채도가 높음
    - 명도(Value): 밝기, 명도가 높을수록 백색에 낮을 수록 흑색에 가까워 짐


### 작업 디렉토리
- ROS workspace의 src 아래에 traffic_sign 폴더 생성

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── traffic_sign
        └── src
{% endhighlight %}


### tf_hsv.py
- 신호등 이미지를 HSV로 변환 예제

{% highlight Python %}
import cv2

image = cv2.imread("green.jpg")

# BGR Color를 HSV Color로 변환하고
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 다시 이를 H, S, V 각각의 성분으로 나누고
h, s, v = cv2.split(hsv)

# 각각을 하나씩 윈도우에 표시함
cv2.imshow("h", h)
cv2.imshow("s", s)
cv2.imshow("v", v)

cv2.waitKey(0)
cv2.destroyAllWindows()
{% endhighlight %}

{% highlight bash %}
$ python tf_hsv.py
{% endhighlight %}


## 이미지에서 형태 찾기

### 원형 검출하기
- Hough Transform 알고리즘으로 원형 검출
- tf_circle.py

{% highlight Python %}
import cv2
import numpy as np

image = cv2.imread("green.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.medianBlur(img, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=50, param2=25, minRadius=0, maxRadius=0)

for i in circles[0, :]: # [ 중심의 x, y 좌표, 반지름 ]
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow("img", cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
{% endhighlight %}

{% highlight bash %}
$ python tf_circle.py
{% endhighlight %}

- 다른 이미지로 실행시켜보자.

{% highlight Python %}
image = cv2.imread("baduk1.jpg", cv2.IMREAD_GRAYSCALE)
{% endhighlight %}

{% highlight bash %}
$ python tf_circle.py
{% endhighlight %}


### Hough Circles 함수

{% highlight Python %}
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=50, param2=25, minRadius=0, maxRadius=0)
{% endhighlight %}

- cv2.HoughCircles(images, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]])
    - method: 검출 방법은 항상 2단계 허프 변환(21HT, Gradient)만 사용
    - dp: 해상도 비율은 원의 중심을 검출하는데 사용되는 누산 평면의 해상도를 의미
        - 인수를 1로 지정할 경우, 입력한 이미지와 동일한 해상도를 가짐. 즉, 입력 이미지 너비와 높이가 동일한 누산 평면이 생성됨.
        - 또한 인수를 2로 지정하면 누산 평면의 해상도가 절반으로 줄어, 입력 이미지의 크기와 반비례함
    - minDist: 최소 거리는 일차적으로 검출됨 원과 원 사이의 최소거리로, 이 값은 원이 여러 개 검출되는 것을 줄이는 역할을 함
    - param1: Canny Edge 임계값은 허프변환에서 자체적으로 Canny Edge를 적용하게 되는데, 이때 사용되는 상위 임계값을 의미함
        - 하위 임계값은 자동으로 할당되며, 상위 임계값의 절반에 해당하는 값을 사용함
    - param2: 중심 임계값은 Gradient 방법에 적용된 중심 히스토그램(누산 평면)에 대한 임계값임. 값이 낮을 경우 더 많은 원이 검출됨.
    - minRadius, maxRadius: 최소 반지름과 최대 반지름은 검출될 원의 반지름 범위임. 0을 입력할 경우 검출할 수 있는 반지름에 제한 조건을 두지 않음.
        - 최소 반지름과 최대 반지름에 각각 0을 입력할 경우, 반지름을 고려하지 않고 검출하며, 최대 반지름에 음수를 입력할 경우 검출된 원의 중심만 반환됨


### Blurring 노이즈 제거
- 너무 많은 원을 찾는 경우
{% highlight Python %}
img = cv2.medianBlur(img, 5)
{% endhighlight %}



## 신호등 인식하기

### 신호등 구의 위치 찾기
- 신호등 이미지에서 구의 위치를 찾기


### 신호 판별하기
- V 성분으로 점등된 구를 판별
    - 원 내의 사각 영역 픽셀값들의 범위를 기준으로 Light의 ON/OFF를 판정함
- tf_circle_mean.py

{% highlight Python %}
import cv2
import numpy as np

image = cv2.imread("green.jpg")
image = cv2.medianBlur(image, 5)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

circles = cv2.HoughCircles(v, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=50, param2=25, minRadius=0, maxRadius=30)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cr_img = v[i[1]-10:i[1]+10, i[0]-10:i[0]+10]
    img_str = "x: {}, y: {}, mean: {}".format(i[0], i[1], cr_img.mean())
    print(img_str)

cv2.imshow("img", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
{% endhighlight %}

{% highlight bash %}
$ python tf_circle_mean.py
{% endhighlight %}


### 우리 나라의 신호등 체계
- [우리나라의 신호등 체계는 어떻게 되어있을까? / YTN 사이언스](https://youtu.be/37UrbxBggjg)
