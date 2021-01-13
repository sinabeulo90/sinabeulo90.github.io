---
layout: post
title:  "[실습] 초음파 센서 활용: ROS"
date:   2021-01-13 03:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

## 초음파 센서 패키지 생성

ROS Workspace의 src 폴더에서 ultrasonic 패키지를 만들고, `/launch` 서브 폴더를 만든다.

{% highlight bash %}
$ catkin_create_pkg ultrasonic std_msgs rospy
{% endhighlight %}

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── ultrasonic
        ├── launch
        │   └── ultra.launch
        └── src
            ├── ultrasonic_pub.py
            └── ultrasonic_sub.py
{% endhighlight %}



## ROS에서의 시리얼 통신

- 리눅스: 시리얼 디바이스에 접근 (/dev/ttyUSB0)
- 파이썬: Serial 모듈을 import해서 코딩 (import Serial)



## ROS 노드 프로그램 소스 코드(ultrasonic_pub.py)

초음파 센서가 보낸 거리 정보를 토픽에 담아 발행한다.

{% highlight Python %}
#!/usr/bin/env python
import serial, time, rospy
from std_msgs.msg import Int32

# 아두이노가 연결된 포트 지정
ser_front = serial.Serial(
    port="/dev/ttyUSB0",
    baudrate=9600)

def read_sensor():
    # 시리얼 포트로 들어온 데이터를 받아옴
    serial_data = ser_front.readline()
    ser_front.flushInput()
    ser_front.flushOutput()

    # str을 숫자로 변환
    ultrasonic_data = int(filter(str.isdigit, serial_data))
    msg.data = ultrasonic_data

if __name__ == "__main__":
    # 발행자 노드 설정, 토픽 초기화
    rospy.init_node("ultrasonic_pub", anonymous=False)  # initialize node
    pub = rospy.Publisher("ultrasonic", Int32, queue_size=1)

    msg = Int32()   # message type
    while not rospy.is_shutdown():
        # 시리얼 포트에서 센서가 보내준 문자열을 읽어서 거리 정보 추출
        read_sensor()

        # 토픽에 담아서 Publish
        pub.publish(msg)
        time.sleep(0.2)
    
    # 끝나면 시리얼 포트 닫기
    ser_front.close()
{% endhighlight %}



## 검증을 위한 Subscriber 노드(ultrasonic_sub.py)

Publisher의 메시지를 받아 출력한다.

{% highlight Python %}
#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32

# 토픽 내용을 화면에 출력하는 callback 함수
def callback(msg):
    print(msg.data)

# 발행자 노드 설정, 토픽 초기화
rospy.init_node("ultrasonic_sub")
sub = rospy.Subscriber("ultrasonic", Int32, callback)

rospy.spin()
{% endhighlight %}



## Launch 파일 만들기

{% highlight XML %}
<launch>
    <node pkg="ultrasonic" type="ultrasonic_pub.py" name="ultrasonic_pub" />
    <node pkg="ultrasonic" type="ultrasonic_sub.py" name="ultrasonic_sub" output="screen" />
</launch>
{% endhighlight %}



## 실행

초음파 센서 데이터를 담은 토픽의 내용이 화면에 출력

{% highlight bash %}
$ roslaunch ultrasonic ultra.launch
$ rqt_graph
$ rostopic echo ultrasonic
{% endhighlight %}
