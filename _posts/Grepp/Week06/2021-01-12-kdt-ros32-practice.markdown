---
layout: post
title:  "[실습] 라이다 센서 활용: 거리 정보 출력"
date:   2021-01-12 02:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

## 패키지 생성

ROS Workspace의 src 폴더에 `my_lidar`패키지를 만들고, `/launch` 서브 폴더를 만든다.

{% highlight bash %}
$ catkin_create_pkg my_lidar std_msgs rospy
{% endhighlight %}

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── my_lidar
        ├── launch
        │   └── lidar_scan.launch
        └── src
            └── lidar_scan.py
{% endhighlight %}



## 라이다 값 출력 프로그램 예제(lidar_scan.py)

{% highlight python %}
#!/usr/bin/env python
import rospy
import time
# LaserScan 메시지 사용 준비
from sensor_msgs.msg import LaserScan

lidar_points = None

# 원할한 데이터 수신을 위해 최대한 간단하게 callback 함수를 구현한다.
def lidar_callback(data):
    global lidar_points
    lidat_points = data.ranges

# Lidar 이름의 노드 생성
rospy.init_node("my_lidar", anonymous=True)
# LaserScan 토픽이 오면 callback 함수가 호출되도록 세팅
rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)

while not rospy.is_shutdown():
    if lidat_points = None:
        continue
    
    rtn = ""

    # 30도씩 건너뛰면서 12개 거리값만 출력
    for i in range(12):
        rtn += str(format(lidat_points[i * 30], ".2f")) + ", "
    
    print(rtn[:-2])
    
    # 천천히 출력한다.
    time.sleep(1.0)
{% endhighlight %}



## Launch 파일 작성(lidar_scan.launch)

{% highlight XML %}
<launch>
    <node name="xycar_lidar" pkg="xycar_lidar" type="xycar_lidar" output="screen" >
        <param name="serial_port" type="string" value="/dev/ttyRPL" />
        <param name="serial_baudrate" type="int" value="115200" />
        <param name="frame_id" type="string" value="laser" />
        <param name="inverted" type="bool" value="false" />
        <param name="angle_compenstate" type="bool" value="true" />
        <param name="scan_mode" type="string" value="Express" />
    </node>
    <node name="my_lidar" pkg="my_lidar" type="lidar_scan.py" output="screen" />
</launch>
{% endhighlight %}



## 실행 결과

센서 거리 값이 `inf` 로 나올 경우: 매우 가까이 있거나(15~30cm), 매우 멀리 있을 경우

센서는 항상 어느 부분 오차와 불확실한 값이 혼재하기 때문에, 가장 최근에 받은 센서들의 평균값을 구하거나 하는 등 보정 작업이 필요하다.

{% highlight bash %}
$ roslaunch my_lidar lidar_scan.launch
$ rqt_graph
{% endhighlight %}