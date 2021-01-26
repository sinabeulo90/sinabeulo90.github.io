---
layout: post
title:  "[과제] 라이다 센서 활용: 데이터 시각화 응용"
date:   2021-01-12 06:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

## 데이터 전달 흐름

라이다 데이터를 Range 데이터로 바꿔서 RVIZ로 시각화

- `lidar_topic.bag` --- (메시지 타입: `LaserScan`, 토픽: `/scan`) ---> `lidar_urdf.py` --- (메시지 타입: `Range`, 토픽: `/scan1`, `/scan2`, `/scan3`, `/scan4`) ---> RVIZ 뷰어
- `lidar_topic.bag` 파일에 저장된 라이다 토픽을 `rosbag`으로 하나씩 발행
- `scan` 토픽에서 장애물까지의 거리 정보를 꺼내, `/scan1`, `/scan2`, `/scan3`, `/scan4` 토픽에 각각 담아 발행
- RVIZ에서는 Range 형식으로 거리 정보 시각화



## 파일 구성

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── rviz_lidar
        ├── launch
        │   └── lidar_urdf.launch
        ├── rviz
        │   └── lidar_urdf.rviz
        ├── src
        │   ├── lidar_topic.bag
        │   └── lidar_urdf.py
        └── urdf
            └── lidar_urdf.urdf
{% endhighlight %}



## lidar_urdf.py

- `LaserScan` 타입의 데이터를 `Range` 타입으로 변경
- `scan` 토픽을 받아 4개의 `scan#` 토픽을 발행

{% highlight Python %}
#!/usr/bin/env python

import serial, time, rospy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Range
from std_msgs.msg import Header

lidar_points = None

# 라이다 토픽이 도착하면 실행되는 callback 함수
def lidar_callback(data):
    # /scan 토픽에서 거리 정보를 추출해서 lidar_points에 담기

# 노드 생성
rospy.init_node("lidar")

# scan 토픽 구독 준비
rospy.Subscriber( ... )

# 4개의 토픽의 발행을 준비
pub1 = rospy.Publisher( ... )
pub2 = rospy.Publisher( ... )
pub3 = rospy.Publisher( ... )
pub4 = rospy.Publisher( ... )

# Range 데이터 타입의 메시지 만들 준비
msg = Range()
h = Header()
...


while not rospy.is_shutdown():
    # /scan 토픽 도착할 때까지 기다리기
    # Range 메시지에 헤더와 거리 정보 채우기
    # scan1, scan2, scan3, scan4 토픽 발행하기
{% endhighlight %}



## URDF 작성하기(lidar_urdf.urdf)

World의 중앙에 RED 박스를 만들고, 4방향에 센서 프레임을 연결한다. `base_link`에 가로, 세로 2cm RED 박스 baseplate를 만들어서 연결하고, 센서는 x, y축을 기준으로 중심에서 10cm씩 이동시켜서 박스의 끝 부분에 배치한다.

{% highlight XML %}
<?xml version="1.0" ?>
<robot name="xycar" xmlns:xacro="http://www.ros.org/wiki/xacro">
    <!-- link: base_link -->
    <link name="base_link" />
    <!-- link: baseplate -->
    <link name="baseplate">
        <visual>
            ...
        </visual>
    </link>
    <!-- joint: base_link_to_baseplate -->
    <joint name="base_link_to_baseplate" type="fixed">
        ...
    </joint>
    <!-- link: front -->
    <link name="front" />
    <!-- joint: baseplate_to_front -->
    <joint name="baseplate_to_front" type="fixed">
        ...
    </joint>
    <!-- link: back -->
    <link name="back" />
    <!-- joint: baseplate_to_back -->
    <joint name="baseplate_to_back" type="fixed">
        ...
    </joint>
    <!-- link: left -->
    <link name="left" />
    <!-- joint: baseplate_to_left -->
    <joint name="baseplate_to_left" type="fixed">
        ...
    </joint>
    <!-- link: right -->
    <link name="right" />
    <!-- joint: baseplate_to_back -->
    <joint name="baseplate_to_right" type="fixed">
        ...
    </joint>
    <material name="black">
        <color rgba="0.0 0.0 0.0 0.0 1.0" />
    </material>
    <material name="blue">
        <color rgba="0.0 0.0 0.0 0.8 1.0" />
    </material>
    <material name="green">
        <color rgba="0.0 0.8 0.0 0.0 1.0" />
    </material>
    <material name="red">
        <color rgba="0.8 0.0 0.0 0.0 1.0" />
    </material>
    <material name="white">
        <color rgba="1.0 1.0 1.0 1.0 1.0" />
    </material>
    <material name="orange">
        <color rgba="1.0 0.423529411765 0.0392156862745 1.0 1.0" />
    </material>
</robot>
{% endhighlight %}



## lidar_urdf.launch

{% highlight XML %}
<launch>
    <!-- 박스 형상 모델링: lidar_urdf.urdf -->
    <!-- Rviz 설정 파일: lidar_urdf.rviz -->
    <!-- 라이다 토픽 발행: lidar_topic.bag -->
    <!-- 토픽 변환(Subscriber & Publish): lidar_urdf.py -->
</launch>
{% endhighlight %}



## 실행 결과 확인

{% highlight XML %}
$ roslaunch rviz_lidar lidar_urdf.launch
$ rqt_graph
{% endhighlight %}
