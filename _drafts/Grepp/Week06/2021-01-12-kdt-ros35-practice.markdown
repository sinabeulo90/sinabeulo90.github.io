---
layout: post
title:  "[실습] 라이다 센서 활용: range 메시지 시각화"
date:   2021-01-12 05:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

## 토픽 전달 흐름

- `lidar_range.py`: Range 데이터를 `/scan1`, `/scan2`, `/scan3`, `/scan4` 이름의 토픽으로 발행
- RVIZ 뷰어: 원뿔 그림으로 Range 거리 정보를 시각화하여 표시



## 파일 구성

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── rviz_lidar
        ├── launch
        │   └── lidar_range.launch
        ├── rviz
        │   └── lidar_range.rviz
        └── src
            └── lidar_range.py
{% endhighlight %}



## [Range 타입](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Range.html)

{% highlight bash %}
$ rosmsg show sensor_msgs/Range

uint8 ULTRASOUND=0
uint8 INFRARED=1
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
uint8 radiation_type
float32 field_of_view
float32 min_range
float32 max_range
float32 range
{% endhighlight %}



## lidar_range.py

{% highlight Python %}
#!/usr/bin/env python

import serial, time, rospy
from sensor_msgs.msg import Range
from std_msgs.msg import Header

rospy.init_node("lidar_range")

# 4개의 토픽의 발행을 준비
pub1 = rospy.Publisher("scan1", Range, queue_size=1)
pub2 = rospy.Publisher("scan2", Range, queue_size=1)
pub3 = rospy.Publisher("scan3", Range, queue_size=1)
pub4 = rospy.Publisher("scan4", Range, queue_size=1)

# Header와 원뿔 모양의 Range 표시에 필요한 정보 채우기
msg = Range()
h = Header()
h.frame_id = "sensorXY"
msg.header = h
msg.radiation_type = Range().ULTRASOUND
msg.min_range = 0.02
msg.max_range = 2.0
msg.field_of_view = (30.0 / 180.0) * 3.14

while not rospy.is_shutdown():
    msg.header.stamp = rospy.Time.now()

    msg.range = 0.4
    pub1.publish(msg)

    msg.range = 0.8
    pub2.publish(msg)

    msg.range = 1.2
    pub3.publish(msg)

    msg.range = 1.6
    pub4.publish(msg)

    time.sleep(0.2)
{% endhighlight %}



## lidar_range.launch

{% highlight XML %}
<launch>
    <!-- rviz display -->
    <node name="rviz_visualizer" pkg="rviz" type="rviz" required="true"
          args="-d $(find rviz_lidar)/rviz/lidar_range.rviz" />
    
    <node name="lidar_range" pkg="rviz_lidar" type="lidar_range.py" />
</launch>
{% endhighlight %}



## Launch 파일 실행

{% highlight bash %}
$ roslaunch rviz_lidar lidar_range.launch
{% endhighlight %}



## RVIZ 설정

- Global Options 항목에서 Fixed Frame 옵션 값을 `sensorXY`로 입력
- Display 탭 하단에 Add 버튼을 클릭한 후, By topic 탭을 선택하여 `/scan#` 아래의 `Range`를 선택하고 OK 버튼을 클릭
    - Topic 옵션: 수신하는 토픽 이름이 정확히 입력되어 있는지 확인
    - Color 옵션: 임의로 색상 지정

RVIZ 창을 닫으려고 하면, lidar_range.rviz 파일을 rviz 폴더에 저장하겠냐는 창이 열리는데, 이때 Save를 한다.
