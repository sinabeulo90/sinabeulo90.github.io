---
layout: post
title:  "[실습] Xycar IMU 센서 활용"
date:   2021-01-11 05:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

## Roll, Pitch, Yaw
- Roll
    - 각도 증가: 오른쪽 바퀴 상승
    - 각도 감소: 왼쪽 바퀴 상승
- Pitch
    - 각도 증가: 오르막길
    - 각도 감소: 내리막길
- Yaw
    - 각도 증가: 왼쪽 회전
    - 각소 감소: 오른쪽 회전



## IMU센서 관련 노드와 토픽

/xycar_imu 노드에서 /imu 토픽을 발행

{% highlight bash %}
std_msgs/Header header  # 헤더
  uint32 seq            # 시퀀스 번호
  time stamp            # 시간
  string frame_id       # 아이디 정보
geometry_msgs/Quaternion orientation    # 기울어짐 정보값
  float64 x
  float64 y
  float64 z
  float64 w
float64[9] orientation_covariance
geometry_msgs/Vector3 angular_velocity
  float64 x
  float64 y
  float64 z
float64[9] angular_velocity_covariance
geometry_msgs/Vector3 linear_acceleration
  float64 x
  float64 y
  float64 z
float64[9] linear_acceleration_covariance
{% endhighlight %}



## 패키지 생성

ROS workspace의 src폴더에서 my_imu 패키지와 /launch 서브 폴더 만들기

{% highlight bash %}
$ catkin_create_pkg my_imu std_msgs rospy
{% endhighlight %}

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── my_imu
        ├── launch
        │   └── roll_pitch_yaw.launch
        └── src
            └── roll_pitch_yaw.py
{% endhighlight %}



## 기울기 값 출력 프로그렘 예제(roll_pitch_yaw.py)

IMU로부터 센싱 값을 받아 roll, pitch, yaw값으로 변환하여 출력

{% highlight Python %}
#!/usr/bin/env python
import rospy
import time

# Imu 메시지와 euler_from_quaternion 함수 사용 준비
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

Imu_msg = None

def imu_callback(data):
    global Imu_msg
    Imu_msg = [
        data.orientation.x,
        data.orientation.y,
        data.orientation.z,
        data.orientation.w]

# "Imu_Print"이름의 노드 생성
rospy.init_node("Imu_Print")

# Imu 토픽이 오면 콜백함수가 호출되도록 세팅
rospy.Subscriber("imu", Imu, imu_callback)

while not rospy.is_shutdown():
    if Imu_msg == None:
        continue
    
    # 쿼터니언 값(x, y, z, w)을 Roll, Pitch, Yaw 값으로 변환
    (roll, pitch, yaw) = euler_from_quaternion(Imu_msg)

    # 화면에 Roll, Pitch, Yaw 값 출력
    print("Roll: %.4f, Pitch: %.4f, Yaw: %.4f" % (roll, pitch, yaw))

    time.sleep(1.0)
{% endhighlight %}



## Launch 파일 작성(roll_pitch_yaw.launch)

/launch 폴더 아래에 만든다.

{% highlight XML %}
<launch>
    <node pkg="xycar_imu" type="9dof_imu_node.py" name="xycar_imu" output="screen">
        <param name="rviz_mode" type="string" value="false" />
    </node>
    <node pkg="my_imu" type="roll_pitch_yaw.py" name="Imu_Print" output="screen" />
</launch>
{% endhighlight %}



##  결과 확인

{% highlight bash %}
$ roslaunch my_imu roll_pitch_yaw.launch
$ rqt_graph
{% endhighlight %}

