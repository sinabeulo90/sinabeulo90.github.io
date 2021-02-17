---
layout: post
title:  "영상처리 기반 정지선 인식"
date:   2021-02-17 02:00:00 +0900
category: "Grepp/KDT"
tag: "Image Processing"
---

## 정지선의 특징

### 정지선의 특징
- 너비가 높이보다 큰 직사각형이다.
- 색이 정해져 있다.
- 차선 위에 있으며, 종종 정지선 뒤에 횡단보도가 있다.

### 그렇다면
- 정지선을 감지하기 위해서는
    - 정지선 색을 찾아야 하고
    - 도로 위에 있는 너비가 큰 직사각형을 찾아야 하고
    - 횡단보도가 있다면, 횡단보도와 정지선이 구분이 되도록 해야한다.
- 위의 조건을 모두 만족한다면, 정확하게 정지선을 찾을 수 있을 것이다.


## 정지선의 색

### 아스팔트에 그려져 있는 기호의 색
- 일반적으로 아스팔트에 그려져 있는 정지선이나 차선들의 색을 생각해보자.
    - 노란색
    - 흰색
- 그렇다면 우리는 노란색과 흰색만 잘 찾으면 도로 위에 차선도 잘 찾을 수 있고, 정지선도 잘 찾을 수 있다.
- 그래서 노란색과 흰색을 잘 찾기 위해 주로 사용하는 이미지 포맷이 HSL과 LAB이다.


### HLS
- 색조, 밝기, 채도 모델을 의미한다.
- L 채널
    - 밝기를 표시, 흰색을 찾는데 유용하게 사용된다.
    - 흰색 차선, 정지선

### LAB
- 국제 조명 위원회가 1976년에 정의한 color space이다.
- 인간이 인지하는 밝기, 적록색, 황청색의 조합으로 표현한다.
- 일반적으로 피부색을 찾는데 많이 사용된다.
- 피부색 --> 노란색 계열, 노란 차선


### 흰색 감지
- white.py

{% highlight Python %}
#!/usr/bin/env python
import cv2

original = cv2.imread("white.png", cv2.IMREAD_UNCHANGED)
H, L, S = cv2.split(cv2.cvtColor(original, cv2.COLOR_BGR2HLS))

_, H = cv2.threshold(H, 125, 255, cv2.THRESH_BINARY)
_, L = cv2.threshold(L, 125, 255, cv2.THRESH_BINARY)
_, S = cv2.threshold(S, 125, 255, cv2.THRESH_BINARY)

cv2.imshow("original", original)
cv2.imshow("H", H)
cv2.imshow("L", L)
cv2.imshow("S", S)

cv2.waitKey(0)
cv2.destroyAllWindows()
{% endhighlight %}

- L 채널이 흰색의 선명도가 가장 좋은 것을 알 수 있음


### 잠깐
- Q: 왜 Threshold를 해주는 건가요?
- A: 단순화 시키기 위해서이다. 우리가 필요한 건 "이 부분이 흰식이다. 아니다."가 중요한 것이지, "여긴 60% 정도만 흰색이다."는 필요하지 않다. 판정 기준이 되는 Threshold 값을 지정해 줌으로써 작업량을 줄여준다.


### 황색 감지
- yellow.py

{% highlight Python %}
#!/usr/bin/env python
import cv2

original = cv2.imread("yellow.png", cv2.IMREAD_UNCHANGED)
L, A, B = cv2.split(cv2.cvtColor(original, cv2.COLOR_BGR2LAB))

_, L = cv2.threshold(L, 156, 255, cv2.THRESH_BINARY)
_, A = cv2.threshold(A, 156, 255, cv2.THRESH_BINARY)
_, B = cv2.threshold(B, 156, 255, cv2.THRESH_BINARY)

cv2.imshow("original", original)
cv2.imshow("L", L)
cv2.imshow("A", A)
cv2.imshow("B", B)

cv2.waitKey(0)
cv2.destroyAllWindows()
{% endhighlight %}

- B 채널이 황색의 선명도가 가장 좋은 것을 알 수 있음


## 카메라 캘리브레이션

### 카메라 캘리브레이션
- 이미지 처리 기능의 구현에는 일반화가 굉장히 중요하다.
- 같은 프로그램 소스를 여러 환경에서 쉽게 구동시키려면 반드시 필요하다.
- 카메라 캘리브레이션은 일반화 작업에 매우 중요한 요소이다.
- 특히 같은 화면에 더 많은 정보를 담기 위해 시야각을 넓힌 어안렌즈로 작업할 때 중요하다.

### ROS Camera Calibration
- 캘리브레이션 수치를 이용하여 OpenCV에서 캘리브레이션 처리된 영상 출력
- calibration.py

{% highlight Python %}
#!/usr/bin/env python
import cv2
import numpy as np

def calibrate_image(frame, mtx, dist, cal_mtx, cal_roi):
    tf_image = cv2.undistort(frame, mtx, dist, None, cal_mtx)
    x, y, w, h = cal_roi
    tf_image = tf_image[y:y+h, x:x+w]
    return cv2.resize(tf_image, (frame.shape[1], frame.shape[0]))

cap = cv2.VideoCapture("Taeback.avi")

mtx = np.array([
    [422.047858, 0.0, 245.895397],
    [0.0, 435.589734, 163.625535], 
    [0.0, 0.0, 1.0]])
dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])

Width, Height = 640, 480
cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width, Height), 1, (Width, Height))

while cap.isOpened():
    ret, original = cap.read()
    calibration = calibrate_image(original, mtx, dist, cal_mtx, cal_roi)
    hstack_cat = np.hstack((original, calibration))
    cv2.imshow("original", original)
    cv2.imshow("calibration", calibration)
    cv2.waitKey(1)
{% endhighlight %}

{% highlight bash %}
$ python calibration.py
{% endhighlight %}


## countNonZero를 이용한 정지선 검출

### countNonZero 활용하기
- 횡단보도와 정지선의 차이
    - 정지선은 연속적인 사각형이고, 횡단보도는 끊어진 사각형이다.
    - 만약 횡단보도와 정지선을 나란히 두고 그 위에 사각형을 그린다면, 사각형 안에 횡단보도 또는 정지선이 얼마나 들어갈까?
    - 당연하지만, 띄엄띄엄 있는 횡단보도보다 이어져 있는 정지선이 더 많이 들어갈 것이다.

### 활성화 픽셀 개수 세기
- Count None Zero 함수
    - cv2.countNonZero(image)
    - 이미지 내부에 0이 아닌 픽셀이 몇 개 있는지 찾아주는 함수이다.
    - 첫 번째 파라미터인 image는 바이너리 이미지를 넣는 것이 좋다.

### 정지선 찾기 - stopline_countNonZero.py
- countNonZero 함수를 이용한 정지선 검출 소스

{% highlight Python %}
#!/usr/bin/env python
import cv2, random, math, copy
import numpy as np

def detect_stopline(cal_image, low_threshold_value):
    stopline_roi, _, _ = set_roi(cal_image, 250, 350, 10)
    image = image_processing(stopline_roi, low_threshold_value)
    if cv2.countNonZero(image) > 1000:
        print("stopline")

def set_roi(frame, x_len, start_y, offset_y):
    _, width, _ = frame.shape
    start_x = int(width/2 - (x_len/2))
    end_x = int(width - start_x)
    return frame[start_y:start_y+offset_y, start_x:end_x], start_x, start_y

def image_processing(image, low_threshold_value):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, _, B = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2LAB))
    _, lane = cv2.threshold(B, low_threshold_value, 255, cv2.THRESH_BINARY)
    return lane

def calibrate_image(frame, mtx, dist, cal_mtx, cal_roi):
    height, width = frame.shape
    tf_image = cv2.undistort(frame, mtx, dist, None, cal_mtx)
    x, y, w, h = cal_roi
    tf_image = tf_image[y:y+h, x:x+w]
    return cv2.resize(tf_image, (width, height))

cap = cv2.VideoCapture("Taeback.avi")

Width, Height = 640, 480
mtx = np.array([
    [422.047858, 0.0, 245.895397],
    [0.0, 435.589734, 163.625535], 
    [0.0, 0.0, 1.0]])
dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])
cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width, Height), 1, (Width, Height))

while cap.isOpened():
    ret, original = cap.read()
    cal_image = calibrate_image(original, mtx, dist, cal_mtx, cal_roi)
    detect_stopline(cal_image, 150)
    cv2.imshow("simple detect", cal_image)
    cv2.waitKey(1)
{% endhighlight %}

{% highlight bash %}
$ python stopline_countNonZero.py
{% endhighlight %}


## FindContour를 이용한 정지선 검출

### Find Contours 활용하기
- 정지선의 외형적 특성을 이용하여 검출하는 방법
- 이때 사용되는 함수가 OpenCV API 중  findContours 함수이다.
    - cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
- findContours 함수는 3가지 인자로 구성되어 있다.
    - 작업 대상 이미지
    - 추출 모드
    - 근사 방법

### 윤곽선 찾기
- cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    - src
        - 윤곽선을 찾을 대상 이미지를 의미함
        - src에 들어가는 이미지는 바이너리 이미지여야 한다
        - HLS의 L채널과 LAB의 B채널
    - contour 추출 모드
        - cv2.RETR_EXTERNAL: 외곽선 중 가장 바깐의 외곽선만 추출하는 모드
        - findContours 함수의 2번째 반환값 hierarchy에는 바깐 외곽선만 나오므로, 자식/부모 인덱스 정보는 출력되지 않는다.
            - Hierarchy
                - Contour의 계층 정보가 들어있는 리스트형 반환값
                - 반환되는 리스트에는 총 4개의 값이 들어있음
                    - [다음 contour의 인덱스, 이전 contour의 인덱스, 첫 번째 자식 contour의 인덱스, 부모 contour의 인덱스]
                - 만약 존재하지 않다면 해당값은 -1이 들어간다.
                - 인덱스는 가장 이전 외곽선을 기준으로 번호를 매긴다.
                - 즉, 0번 contour에게 있어서 1 ~ 5번 까지의 자식 contour들이 있지만 이들을 다음 contour라고 말하지 않고, 자식 contour가 아닌 자신과 동등한 계층에 있는 contour를 다음 contour라고 한다.
                    - 자식 contour가 있는 리스트 내부에는 첫 번째 인자(다음 contour)가 전부 -1이다.
                - -1은 해당 인덱스에 contour가 없다는 뜻이다.
                - cv2.RETR_EXTERNAL에서는 모든 contour의 계층이 동일하지만, 자식 부모는 존재하지 않는다.
    - contour 근사 방법
        - cv2.CHAIN_APPROX_SIMPLE: contour의 점만 찾음
        - cv2.CHAIN_APPROX_NONE: 전부 찾음
        - None은 전부 다 찾는 방법이기 떄문에 시스템 리소스를 많이 쓸수 밖에 없다. 그래서 여기서는 SIMPLE을 사용한다.


### 윤곽선의 총 길이, 너비, 외접하는 사각형 찾기
- cv2.arcLength(contour, True)
    - contour를 첫 번째 인자로 입력하면 해당 contour의 총 길이를 반환한다.
    - 두 번째 인자는, 입력하는 contour의 폐곡선 여부를 입력하는 값이다. True면 완전히 닫힌 폐곡선을 의미하고, False면 연린 곡선을 의미한다.

- cv2.contourArea(contour)
    - contour를 첫 번째 인자로 입력하면 해당 contour 내부에 있는 픽셀 개수를 반환한다.

- cv2.boundingRect(contour)
    - 입력한 contour를 외접하는 사각형을 반환하는 함수이다.
    - 반환값은 튜플로 반환되며, 첫번째 인자부터 (사각형의 왼쪽 위 x 좌표, 사각형의 왼쪽 위 y 좌표, 사각형의 너비, 사각형의 높이) 순서대로 들어있다.
    - contour을 외접하는 사각형을 그린 후 해당 사각형의 중점을 구할 수 있다.


### 외곽선의 각이 몇개인지 찾기
- cv2.approxPolyDP(contour, epsilon, True)
    - 첫 번째 인자는 contour의 값을 의미하고, 세 번째 인자는 arcLength 함수의 두 번째 인자처럼 개폐의 여부를 의미한다.
    - 두 번째 인자는 꼭지점을 얼만큼 줄일지에 대한 계수인데, 일반적으로 `윤곽선 총 길이의 몇 %` 와 같이 값을 입력한다.

{% highlight Python %}
# 총 길이의 2%를 의미
cv.arcLength(contour, True) * 0.02
{% endhighlight %}

    - 출력값: 다각형의 꼭지점 x, y 좌표로 출력되며, 해당 리스트의 요소 갯수가 각의 갯수가 된다.

{% highlight Python %}
if len(cv2.approxPolyDP(contour, perimeter*0.02, True)) == 4:
    ...
{% endhighlight %}


### 윤곽선 찾기를 이용한 정지선 검출 - stopline_findContours.py
- findContours()를 이용한 정지선 검출 - 소스코드

{% highlight Python %}
#!/usr/bin/env python
import cv2
import numpy as np

def detect_stopline_contour(cal_image, low_threshold_value):
    blur = cv2.GaussianBlur(cal_image, (5, 5), 0)
    _, _, B = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2LAB))
    _, lane = cv2.threshold(B, low_threshold_value, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cont in contours:
        length = cv2.arcLength(cont, True)
        area = cv2.contourArea(cont)

        if not ((area > 3000) and (length > 500)):
            continue
        
        if len(cv2.approxPolyDP(cont, length*0.02, True)) != 4:
            continue
        
        (x, y, w, h) = cv2.boundingRect(cont)
        center = (x + int(w/2), y + int(h/2))
        _, width, _ = img.shape

        if 200 <= center[0] <= (width - 200):
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print("stopline")
            
def calibrate_image(frame, mtx, dist, cal_mtx, cal_roi):
    height, width = frame.shape
    tf_image = cv2.undistort(frame, mtx, dist, None, cal_mtx)
    x, y, w, h = cal_roi
    tf_image = tf_image[y:y+h, x:x+w]
    return cv2.resize(tf_image, (width, height))

cap = cv2.VideoCapture("Taeback.avi")

Width, Height = 640, 480
mtx = np.array([
    [422.047858, 0.0, 245.895397],
    [0.0, 435.589734, 163.625535], 
    [0.0, 0.0, 1.0]])
dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])
cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width, Height), 1, (Width, Height))

while cap.isOpened():
    ret, original = cap.read()
    cal_image = calibrate_image(original, mtx, dist, cal_mtx, cal_roi)
    detect_stopline_contour(cal_image, 150)
    cv2.imshow("Contours", cal_image)
    cv2.waitKey(1)
{% endhighlight %}

{% highlight bash %}
$ python stopline_countNonZero.py
{% endhighlight %}

{% highlight bash %}
$ python stopline_findContours.py
{% endhighlight %}