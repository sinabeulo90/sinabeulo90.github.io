---
layout: post
title:  "[과제] 라이다 기반 장애물 회피 주행"
date:   2021-01-19 02:00:00 +0900
categories: "Grepp/KDT"
tag: ROS
---

## 과제 1: 라이다를 사용하여 장애물이 있으면 정재했다가 다시 주행하는 기능 구현

### 패키지 생성

- ROS workspace의 src 폴더에서 lidar_drive 패키지를 만들고, /launch 서브 폴더를 만든다.


{% highlight bash %}
$ catkin_create_pkg lidar_drive std_msgs rospy
{% endhighlight %}


{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── lidar_drive
        ├── launch
        │   └── lidar_gostop.launch
        └── src
            └── lidar_gostop.py
{% endhighlight %}



### Launch 파일 생성(lidar_gostop.launch)

{% highlight XML %}
<launch>
    <!-- Xycar 모터 제어기 구동(xycar_motor_a2.launch) -->
    <!-- Xycar 라이다 구동 -->
    <!-- 라이다를 이용한 Go-Stop 프로그램 실행(lidar_gostop.py) -->
</launch>
{% endhighlight %}



### Python 파일 생성(lidar_gostop.py)

{% highlight python %}
#!/usr/bin/env python
import rospy, time
# LaserScan, xycar_motor 메시지 사용 준비
from sensor_msgs.msg import LaserScan
from xycar_motor.msg import xycar_motor

# 라이다 거리 정보를 담을 저장 공간 준비
motor_msg = xycar_motor()
distance = []

# 라이다 토픽이 들어오면 실행되는 callback 함수 정의
def callback(data):
    pass

# 자동차 전진
def drive_go():
    pass

# 자동차 정지
def drive_stop():
    pass

# 노드 선언, 구독과 발행할 토픽 선언
rospy.init_node("lidar_driver")
...
...

# Laser가 기동할 때까지 잠시 대기

while not rospy.is_shutdown():
    # 전방에 장애물이 있으면
    drive_stop()

    # 전방에 장애물이 없으면
    drive_go()
{% endhighlight %}



## 실행 결과

자동차가 직진하다가 전방에 물체가 있으면 정지하고, 장애물이 없어지면 다시 출발한다.

{% highlight bash %}
$ roslaunch lidar_drive lidar_gostop.launch
$ rqt_graph
{% endhighlight %}



## 과제 2: 장애물이 있으면 일단 후진한 뒤에, 핸들을 꺽어 다른 방향으로 전진하는 방법 등으로 계속 주행하는 기능 구현

### 패키지 생성

- 과제 1에서 사용한 lidar_drive 패키지를 사용

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── lidar_drive
        ├── launch
        │   └── lidar_drive.launch
        └── src
            └── lidar_drive.py
{% endhighlight %}



### Launch 파일 생성(lidar_drive.launch)

{% highlight XML %}
<launch>
    <!-- Xycar 모터 제어기 구동(xycar_motor_a2.launch) -->
    <!-- Xycar 라이다 구동 -->
    <!-- 라이다를 이용한 장애물 회피 주행 프로그램 실행(lidar_drive.py) -->
</launch>
{% endhighlight %}



### Python 파일 생성(lidar_drive.py)

{% highlight python %}
...

while not rospy.is_shutdown():
    # 전방에 장애물이 있으면, 후진 후 회피 주행

    # 전방에 장애물이 없으면, 직진 주행
{% endhighlight %}



## 실행 결과

자동차가 직진하다가 전방에 물체가 있으면 정차한 후 후진하여 핸들을 꺽어서 장애물을 피해서 주행한다.

{% highlight bash %}
$ roslaunch lidar_drive lidar_drive.launch
$ rqt_graph
{% endhighlight %}


