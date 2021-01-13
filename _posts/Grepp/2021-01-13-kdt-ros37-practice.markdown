---
layout: post
title:  "[실습] 초음파 센서 활용: 패키지 만들기"
date:   2021-01-13 01:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

## 초음파 센서

장애물까지의 거리를 알려주는 센서



## 초음파센서 토픽

- `/xycar_ultrasonic` 토픽
- `std_msgs/Int32MultiArray`타입

{% highlight bash %}
std_msgs/MultiArrayLayout layout
  std_msgs/MultiArrayDimension[] dim
    string label
    uint32 size
    uint32 stride
  uint32 data_offset
int32[] data    # 초음파 센서 거리정보를 담고 있는 배열(int32 x 8개)
{% endhighlight %}



## 패키지 생성

ROS Workspace의 src 폴더에서 my_ultra 패키지를 만들고, `/launch` 서브 폴더를 만든다.

{% highlight bash %}
$ catkin_create_pkg my_ultra std_msgs rospy
{% endhighlight %}

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── my_ultra
        ├── launch
        │   └── ultra_scan.launch
        └── src
            └── ultra_scan.py
{% endhighlight %}



## 초음파 센서 값 출력 프로그램(ultra_scan.py)

초음파 센서로부터 주변 물체까지의 거리값을 받아 출력

{% highlight Python %}
#!/usr/bin/env python
import rospy
import time
# Int32MultiArray 메시지 사용 준비
from std_msgs.msg import Int32MultiArray

ultra_msg = None

# 초음파 센서 토픽이 들어오면 실행되는 callback 함수 정의
def ultra_callback(data):
    global ultra_msg
    ultra_msg = data.data

# ultra_node 이름의 노드 생성
rospy.init_node("ultra_node")

# 초음파 센서 토픽이 오면 callback 함수가 호출되도록 세팅
rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, ultra_callback)

while not rospy.is_shutdown():
    if ultra_msg == None:
        continue

    # 초음파 센서 데이터를 천천히 출력한다.
    print(ultra_msg)
    time.sleep(0.5)
{% endhighlight %}



## Launch 파일 작성(ultra_scan.launch)

{% highlight XML %}
<launch>
    <node pkg="xycar_ultrasonic" type="xycar_ultrasonic.py"
          name="xycar_ultrasonic" output="screen" />
    <node pkg="my_ultra" type="ultra_scan.py" name="my_ultra" output="screen" />
</launch>
{% endhighlight %}



## 결과 확인

{% highlight bash %}
$ roslaunch my_ultra ultra_scan.launch
$ rqt_graph
{% endhighlight %}
