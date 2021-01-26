---
layout: post
title:  "[과제] 초음파 센서 기반 장애물 회피 주행"
date:   2021-01-20 04:00:00 +0900
categories: "Grepp/KDT"
tag: ROS
---

## 과제 1: 초음파 센서를 사용하여 장애물이 있으면 정지했다가 다시 주행하는 기능 구현



### 패키지 생성

ROS Workspace의 src 폴더에서 ultra_drive 패키지를 만들고, `/launch` 서브 폴더를 만든다.

{% highlight bash %}
$ catkin_create_pkg ultra_drive std_msgs rospy
{% endhighlight %}


{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── ultra_drive
        ├── launch
        │   └── ultra_gostop.launch
        └── src
            └── ultra_gostop.py
{% endhighlight %}



### Launch 파일 생성(ultra_gostop.launch)

{% highlight XML %}
<launch>
    <!-- 노드 설정: Xycar 모터 제어기 구동(xycar_motor_b2.launch) -->
    <!-- 노드 설정: Xycar 초음파 센서 구동 -->
    <!-- 노드 설정: 초음파 센서를 이용한 Go-Stop 프로그램 실행(ultra_gostop.py) -->
</launch>
{% endhighlight %}



### ultra_gostop.py

초음파 센서로 장애물 감지하여 정차

{% highlight Python %}
#!/usr/bin/env python
import rospy, time
# Int32MultiArray, xycar_motor 메시지 사용 준비
from std_msgs.msg import Int32MultiArray
fom xycar_motor.msg import xycar_motor

ultra_msg = None    # 초음파 센서 거리 정보를 담을 저장공간 준비
motor_msg = xycar_motor()

# 초음파 센서 토픽이 들어오면, 실행되는 callback 함수 정의
def callback(data):
    pass

# 자동차 전진
def drive_go():
    pass

# 자동차 정지
def drive_stop():
    pass

# 노드 선언 및 구독/발행할 토픽 선언
rospy.init_node("ultra_driver")
...
...

# 초음파 센서가 기동할 때까지 잠시 대기

while not rospy.is_shutdown():
    # 전방에 장애물이 있으면
    drive_stop()

    # 전방에 장애물이 없으면
    drive_go()
{% endhighlight %}



### 실행 결과

자동차가 직진하다가 전방에 물체가 있으면 정지하고, 장애물이 없어지면 다시 출발한다.

{% highlight bash %}
$ roslaunch ultra_drive ultra_gostop.launch
$ rqt_graph
{% endhighlight %}



## 과제 2: 장애물이 있으면 일단 후진한 뒤에, 핸들을 꺽어 다른 방향으로 전진하는 방법 등으로 계속 주행하는 기능 구현



### 패키지 생성

- 과제 1에서 사용한 ultra_drive 패키지를 사용

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── ultra_drive
        ├── launch
        │   └── ultra_drive.launch
        └── src
            └── ultra_drive.py
{% endhighlight %}



### Launch 파일 생성(ultra_drive.launch)

{% highlight XML %}
<launch>
    <!-- 노드 설정: Xycar 모터 제어기 구동(xycar_motor_b2.launch) -->
    <!-- 노드 설정: Xycar 초음파 센서 구동 -->
    <!-- 노드 설정: 초음파 센서를 이용한 장애물 회피 주행 프로그램 실행(ultra_drive.py) -->
</launch>
{% endhighlight %}



### ultra_drive.py

초음파 센서로 장애물 감지하여 회피 주행

{% highlight Python %}
...

while not rospy.is_shutdown():
    # 전방에 장애물이 있으면, 후진 후 회피 주행

    # 전방에 장애물이 없으면, 직진 주행
{% endhighlight %}



### 실행 결과

자동차가 직진하다가 전방에 물체가 있으면 정차한 후에 후진하고, 핸들을 꺽어 장애물을 피해서 다시 주행한다.

{% highlight bash %}
$ roslaunch ultra_drive ultra_drive.launch
$ rqt_graph
{% endhighlight %}
