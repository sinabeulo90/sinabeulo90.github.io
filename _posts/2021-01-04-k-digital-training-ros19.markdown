---
layout: post
title:  "자율주행 모형차 Xycar 자이카 소개"
date:   2021-01-04 04:00:00 +0900
categories:
    - "K-Digital Training"
    - "자율주행 데브 코스"
---

## Xycar™ 자이카 소개

(주)자이트론에서 제작한 1/10 크기의 자율주행 모형차


## Xycar 제품 라인업

| Xycar-X 모델 | Xycar-A 모델 | Xycar-B 모델 | Xycar-C 모델 |
|:-|:-|:-|:-|
| NVIDIA AGX Xavier | NVIDIA Jetson TX2 프로세서 | NVIDIA Jetson TX2 프로세서 | 라즈베리파이 4 프로세서 |
| 카메라 + 라이다 + 관성센서 + LED Matrix + 수동조종 앱 | 카메라 + 라이다 + 관성센서 + LED Matrix + 수동조종 앱 | 카메라 + 초음파센서(전후좌우) + 관성센서 + RC카 조종기 | 카메라 + 초음파센서(전후좌우) + 관성센서 + RC카 조종기 |
| 기존 RC카 모터제어기를 VESC 모터제어기로 교체 | 기존 RC카 모터제어기를 VESC 모터제어기로 교체 | RC카 모터제어기 사용(아두이노를 통해 제어) | RC카 모터제어기 사용(아두이노를 통해 제어) |
| 24,000mAh Li-ion 배터리 + 3,000mAh Ni-MH 차량 배터리 | 24,000mAh Li-ion 배터리 + 3,000mAh Ni-MH 차량 배터리 | 20,000mAh Li-ion 배터리 + 3,000mAh Ni-MH 차량 배터리 | 20,000mAh Li-ion 배터리 + 1,700mAh Ni-MH 차량 배터리 |




## 다양한 자율주행 미션 수행

- 직선 주행
- 곡선 주행
- 정지선
- 돌발 장애물
- U-턴
- 로터리 진입과 진출
- 차선 변경
- 신호등 신호 구분
- 언덕
- 요철 구간
- 정차
- ...



## Xycar 소프트웨어 Stack

1. 운영체제: Ubuntu Linux OS 16.04 또는 18.04
- 프로세서 보드에 따라 버전에 차이 있음

2. 장치제어 미들웨어: ROS(Open Source Robot OS)
- 각종 센서와 모터의 통합 관리 및 모듈 간의 통신 지원 등

3. ROS Package: 장치별 제어를 위해 ROS package 구축
- ex. Speed/Steering Control, Camera, LIDAR, IMU, Arduino/LED, Ultrasonic

4. 자율주행 SW 프로그래밍: 오픈소스 라이브러리 활용
- ex. Python, TensorFlow/PyTorch, NVIDIA CUDA, OpenCV


