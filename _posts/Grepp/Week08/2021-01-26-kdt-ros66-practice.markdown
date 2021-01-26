---
layout: post
title:  "[과제] 슬라이딩 윈도우 기반 차선인식 주행"
date:   2021-01-26 03:00:00 +0900
categories: "Grepp/KDT"
tag: OpenCV
---

## 실제 트랙에서 차선을 벗어나지 않고 주행

카메라로 촬영한 차량 전방의 도로 영상을 OpenCV를 이용하여 원근 변환과 슬라이딩 윈도우 기반으로 차선을 찾고, 양쪽 차선의 위치를 따져서 핸들을 얼마나 꺾을지 조향각을 결정



## 슬라이딩 윈도우 기반의 차선인식 주행

1. Image Read: 카메라 영상 신호를 이미지로 읽기
2. Warping: 원근 변환으로 이미지 변형
3. Gaussian Blur: 노이즈 제거
4. Threshold: 이진 이미지로 변환
5. Histogram: 히스토그램으로 차선 위치 추출
6. Sliding Window: 슬라이딩 윈도우 좌표에 9개씩 쌓기
7. Polyfit: 2차 함수 그래프로 차선 그리기
8. 차선 영역 표시: 원본 이미지에 차선 영역 오버레이


## 작업 패키지

앞서 만들어 놓은 sliding_drive 패키지에서 작업한다.

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── sliding_drive
        ├── launch
        |   └── sliding_drive.launch
        └── src
            ├── sliding_drive.py
            └── steer_arrow.png
{% endhighlight %}



## sliding_drive.launch

{% highlight XML %}
<launch>
    <!-- 노드 실행: Xycar 모터 제어기 구동 -->
    <!-- 노드 실행: Xycar 카메라 구동 -->
    <!-- 노드 실행: 슬라이딩 윈도우 기반 주행 프로그램 실행(sliding_drive.py) -->
</launch>
{% endhighlight %}



## 프로그램 흐름도(sliding_drive.py)

1. 카메라 노드에서 토픽을 구족해서 영상 프레임 획득
2. 카메라 Calibration 설정값으로 이미지 보정
3. 원근 변환으로 차선 이미지를 Bird's eye view로 변환
4. OpenCV 영상 처리
    - Gaussian Blur: 노이즈 제거
    - cvtColor: BGR을 HLS 포맷으로 변경
    - threshold: 이진화 처리
5. 히스토그램을 사용하여 좌우 차선의 시작 위치 파악
6. 슬라이딩 윈도우를 좌우 9개씩 쌓아 올리기
7. 왼쪽/오른쪽 차선의 위치 찾기
8. 적절한 조향값을 계산하고, 모터 제어 토픽 발행


## 노드 연결도

/usb_cam --- (/usb_cam/image_raw) ---> /sliding_drive --- (/xycar_motor) ---> /xycar_motor
