---
layout: post
title:  "[과제] OpenCV, rosbag을 이용한 차선 인식 주행"
date:   2021-01-20 03:00:00 +0900
categories: "Grepp/KDT"
tag: ROS
---

## OpenCV와 rosbag을 이용한 차선인식 주행

rosbag --- /usb_cam/image_raw ---> line_follow --- /xycar_motor ---> converter --- /joint_states ---> RVIZ 뷰어/rviz_odom



## 기존 패키지에서 작업

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── line_drive
        ├── launch
        │   ├── line_follow.launch
        │   └── line_follow_rosbag.launch
        └── src
            ├── cam_topic.bag
            └── line_follow.py
{% endhighlight %}



## line_follow.py

{% highlight Python %}
# /usb_cam/image_raw 토픽을 구독 - Subscribe
# 토픽 데이터를 OpenCV 이미지 데이터로 변환

"""
OpenCV 영상 처리
1. ROI : 관심 영역 잘라내기
2. Grayscale: 흑백 이미지로 변환
3. Gaussian blur: 노이즈 제거
4. HSV + Binary: HSV 기반으로 이진화 처리
"""

# 차선 위치 찾고 화면 중앙에서 어느 쪽으로 치우쳤는지 파악
# 핸들을 얼마나 꺾을 지 결정(조향각 설정 각도 계산)
# 모터 제어를 위한 토픽 발행 - Publish
{% endhighlight %}



## 조향각의 설정

- 인식된 양쪽 차선의 중점을 기준으로 화면 중앙의 가상의 지점의 거리에 따라 조향 정도를 설정
- 몇 픽셀에 1도씩 꺽어야 하는지 수행착오를 통해 파악



## Launch 파일 작성(line_follow.launch)

다른 터미널에서 별도로 rosbag 실행

{% highlight bash %}
$ rosbag play cam_topic.bag
{% endhighlight %}


{% highlight XML %}
<launch>
    <!-- 파라미터 설정: xycar_3d.urdf -->
    <!-- 노드 설정: rviz + rviz_odom.rviz -->
    <!-- 노드 설정: state_publisher -->
    <!-- 노드 설정: converter.py -->
    <!-- 노드 설정: rviz_odom.py -->
    <!-- 노드 설정: line_follow.py -->
</launch>
{% endhighlight %}



## Launch 파일 작성(line_follow_bag.launch)

아예 Launch 파일 안에 rosbag 명령 실행

{% highlight XML %}
<launch>
    <!-- 파라미터 설정: xycar_3d.urdf -->
    <!-- 노드 설정: rviz + rviz_odom.rviz -->
    <!-- 노드 설정: state_publisher -->
    <!-- 노드 설정: converter.py -->
    <!-- 노드 설정: rviz_odom.py -->
    <!-- 노드 설정: line_follow.py -->
    <!-- 노드 설정: rosbag + cam_topic.bag -->
</launch>
{% endhighlight %}



## 실행 결과

- 가상 공간에서 차량 주행

{% highlight bash %}
$ roslaunch line_drive line_follow.launch
$ roslaunch line_drive line_follow_rosbag.launch
{% endhighlight %}
