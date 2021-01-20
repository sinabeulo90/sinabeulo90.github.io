---
layout: post
title:  "[실습] 명도차 기반 차선 인식 + 동영상"
date:   2021-01-20 02:00:00 +0900
categories: "Grepp/KDT"
tag: ROS
---

## 관심 영역 ROI 설정

- 동영상 파일의 프레임 크기: 640x480
    - 세로 좌표: 430~450 영역을 ROI로 설정(차량 바로 앞의 차선)
    - 가로 좌표: 0~200, 440~640 영역을 각각 왼쪽과 오른쪽 차선을 발견하기 위한 구간으로 설정
    - 이 좌표는 차선을 잘 잡아낼 수 있는 영역을 정하는 것이므로, 상황에 따라 달라짐

- USB 카메라와 OpenCV를 이용하여 차선을 인식하고, 인식된 차선을 따라 스스로 주행할 수 있는 자율주행
- 직선/곡선구간 주행



## 영역 내 흰색 픽셀의 갯수를 기준으로 차선 인식

- 사각형 안의 흰색 픽셀 수를 기준으로 녹색 사각형은 검출된 차선의 위치를 표시
- 검출 영역의 가로x세로 크기: 20픽셀x10픽셀
- 위 200개(20x10) 픽셀 들 중 160개(80%) 이상이 흰색이면 차선으로 간주
- 검출을 위한 영역의 크기(20x10) 및 차선 인식을 위한 흰색 픽셀 비율의 하한(80%) 값은 모두 변경 가능하며, 시행 착오를 거쳐 가장 잘 찾는 값을 찾아낸다.



## 차선 인식

- 잘라낸 영역에서 한 픽셀씩 움직이면서 차선 찾기
    - 중앙에서 바깥으로?
    - 왼쪽과 오른쪽 끝에서 중앙으로?



## 패키지 생성

ROS workspace의 src폴더에서 line_drive 패키지 생성

{% highlight bash %}
$ catkin_create_pkg line_drive rospy tf geometry_msgs rviz xacro
{% endhighlight %}

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── line_drive
        └── src
            ├── line_fine.py
            ├── track1.avi
            └── track2.avi
{% endhighlight %}



## line_find.py

트랙 영상에서 차선을 추출하는 파이썬 프로그램

{% highlight Python %}
#!/usr/bin/env python
import cv2, time
import numpy as np

# 차선 추출에 이용할 동영상 파일 track1.avi 또는 track2.avi 선택
cap = cv2.VideoCapture("track1.avi")

threshold = 60      # 이미지 이진화에 이용할 명도 하한
width = 640
scan_width, scan_height = 200, 20   # 차선 검출을 하기 위해 검사할 영역
lmid, rmid = scan_width, width - scan_width     # 바깥쪽 부터 검사할 때,
                                                # 왼쪽/오른쪽 검사가 끝날 가로 좌표
area_width, area_height = 20, 10    # 흰 픽셀의 갯수를 검사할 영역의 크기
vertical = 430      # ROI 설정을 위한 세로 좌표(왼쪽 끝)
row_begin = (scan_height - area_height) // 2    # ROI 내에서 픽셀 검사 영역의
                                                # 세로 상대 좌표 시작
row_end = row_begin + area_height               # ROI 내에서 픽셀 검사 영역의
                                                # 세로 상대 좌표 끝
pixel_threshold = 0.8 * area_width * area_height    # 검사 영역을 차선으로 판단하는 흰색 픽셀 비율의 하한

while True:
    ret, frame = cap.read()
    # 동영상으로부터 프레임을 읽어들여서 마지막에 도달하거나,
    # ESC 키가 눌릴때까지 반복
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break

    roi = frame[vertical:vertical + scan_height, :]     # ROI 설정
    frame = cv2.rectangle(frame, (0, vertical), (width - 1, vertical + scan_height), (255, 0, 0), 3)    # 설정한 ROI의 둘레에 파란색 사각형을 그림
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lbound = np.array([0, 0, threshold], dtype=np.uint8)
    ubound = np.array([131, 255, 255], dtype=np.uint8)

    bin = cv2.inRange(hsv, lbound, ubound)
    view = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)

    left, riht = -1, -1

    # 왼쪽부터 안쪽으로 한 픽셀씩 당겨오면서 차례대로 검사
    for l in range(area_width, lmid):
        area = bin[row_begin:row_end, 1-area_width:1]
        if cv2.countNonZero(area) > pixel_threshold:
            left = l
            break
    
    # 오른쪽부터 안쪽으로 한 픽셀씩 당겨오면서 차례대로 검사
    for r in range(width - area_width, rmid, -1):
        area = bin[row_begin:row_end, r:r+area_width]
        if cv2.countNonZero(area) > pixel_threshold:
            right = r
            break
    
    # 왼쪽 차선이 검출되었으면, 잘라낸 ROI 이미지에 녹색 사각형을 그림
    if left != -1:
        lsquare = cv2.rectangle(view, (left-area_width, row_begin), (left, row_end), (0, 255, 0), 3)
    else:
        print("Lost left line")
        
    # 오른쪽 차선이 검출되었으면, 잘라낸 ROI 이미지에 녹색 사각형을 그림
    if right != -1:
        rsquare = cv2.rectangle(view, (right, row_begin), (right+area_width, row_end), (0, 255, 0), 3)
    else:
        print("Lost right line")
    
    # "origin" 타이틀의 윈도우에는 카메라를 이용하여 취득한 영상 표시 + ROI
    cv2.imshow("origin", frame)
    # "view" 타이틀의 윈도우에는 ROI를 잘라내어 이진화한 영상 표시
    cv2.imshow("view", view)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lbound = np.array([0, 0, threshold_60], dtype=np.uint8)
    ubound = np.array([131, 255, 255], dtype=np.uint8)
    hsv = cv2.inRange(hsv, lbound,  ubound)

    # "hsv" 타이틀의 윈도우에는 카메라 영상의 이진화한 영상 표시
    cv2.imshow("hsv", hsv)

    time.sleep(0.1)

cap.release()
cv2.destoryAllWindows()
{% endhighlight %}



## line_find.py 실행

- `~/xycar_ws/src/line_drive/src` 폴더 아래에 track1.avi 파일 준비

{% highlight bash %}
$ chmod + x line_find.py
$ python line_find.py
{% endhighlight %}
