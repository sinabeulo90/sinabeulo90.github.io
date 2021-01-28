---
layout: post
title:  "[과제] 허프 변환 기반 차선인식 주행"
date:   2021-01-26 03:00:00 +0900
categories: "Grepp/KDT"
tag: OpenCV
---

## 실제 트랙에서 차선을 벗어나지 않고 주행

카메라로 촬영한 차량 전방의 도로 영상을 OpenCV를 이용하여 허프 변환 기반으로 차선을 찾고, 양쪽 차선의 위치를 따져서 핸들을 얼마나 꺾을지 조향각을 결정



## 차선 인식 주행을 위해 필요한 것들

1. 카메라 입력 데이터에서 프레임 취득
    - 카메라 토픽 구독
2. 얻어낸 영상 데이터를 처리하여 차선 위치를 결정
    - 색 변환: BGR ---> Grayscale
    - 외곽선 추출: Canny 함수로 임계값 범위를 주고 외곽선 추출
    - 관심영역(ROI)을 잘라내기
3. 차선 검출: 허프 변환으로 직선 찾기
    - 양쪽 차선을 나타내는 평균 직선 구하기
    - 수평선 긎고 양쪽 직선과의 교점 좌표 구하기
4. 차선 위치를 기준으로 조향각 결정
    - 차선의 중앙을 차량이 달리도록 판단
5. 결정한 조향각에 따라 조향 모터 제어
    - 모터 제어 토픽 발행



## 허프변환을 이용한 차선 찾기

1. Image Read: 카메라 영상신호를 이미지로 읽기
2. Grayscale: 흑백 이미지로 변환
3. Gaussian Blur: 노이즈 제거
4. Canny: Edge 검출
5. ROI: 관심 영역 잘라내기
6. HoughLinesP: 선분 검출



## 작업 패키지

앞서 만들어 놓은 hough_drive 패키지에서 작업한다.

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── hough_drive
        ├── launch
        |   └── hough_drive.launch
        └── src
            ├── hough_drive.py
            └── steer_arrow.png
{% endhighlight %}



## hough_drive.launch

{% highlight XML %}
<launch>
    <!-- 노드 실행: Xycar 모터 제어기 구동 -->
    <!-- 노드 실행: Xycar 카메라 구동 -->
    <!-- 노드 실행: 허프 변환 기반 차선인식 주행 프로그램 실행(hough_drive.py) -->
</launch>
{% endhighlight %}



## 프로그램 흐름도(hough_drive.py)

1. 카메라 노드에서 토픽을 구족해서 영상 프레임 획득
2. 영상 프레임을 OpenCV 함수로 넘겨 처리
3. OpenCV 영상 처리
    - Grayscale: 흑백 이미지로 변환
    - Gaussian Blur: 노이즈 제거
    - Canny Edge: 외곽선 Edge 추출
    - ROI: 관심 영역 잘라내기
    - HoughLinesP: 선분 검출
4. 차선의 위치 찾고 화면 중앙에서 어느 쪽으로 치우쳤는지 파악
5. 핸들을 얼마나 꺽을 지 결정(조향각 설정 각도 계산)
6. 모터제어 토픽을 발행해서 차량의 움직임을 조종



## 노드 연결도

/usb_cam --- (/usb_cam/image_raw) ---> /hough_drive --- (/xycar_motor) ---> /xycar_motor
