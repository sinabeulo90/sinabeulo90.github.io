---
layout: post
title:  "[과제] IMU 센서 활용: imu_data를 활용한 RVIZ 시각화"
date:   2021-01-12 01:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

## 과제 목표: IMU 데이터로 뷰어의 박스 움직이기

`imu_data.txt` ---> `imu_generator.py` ---> `RVIZ 뷰어`

1. 주어진 `imu_data.txt` 파일에서 한 줄씩 데이터를 읽어서 `Imu` 메시지 타입에 맞게 가공
2. 가공된 메시지를 `/imu` 토픽에 넣은 후 RVIZ 뷰어를 향해 발행



## 파일 구성

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── rviz_imu
        ├── launch
        │   └── imu_generator.launch
        ├── rviz
        │   └── imu_generator.rviz
        └── src
            ├── imu_data.txt
            └── imu_generator.py
{% endhighlight %}



## imu_data.txt 데이터 파일 포맷

- "roll: `value`, pitch: `value`, yaw: `value`"
- `value` 값은 모두 라디안으로 기록됨



## Imu 메시지 타입

- sensor_msgs/Imu
    - IMU 센서를 위한 메시지 타입
    - Roll, Pitch, Yaw가 아닌 쿼터니언 값(x, y, z, w: 4차원 벡터)을 담음
    - `std_msgs/Header header`, `geometry_msgs/Quaternion orientation` 수정
- 추천 라이브러리
    - 오일러 ---> 쿼터니언: `from tf.transformations import quaternion_from_euler`

{% highlight bash %}
$ rosmsg info sensor_msgs/Imu

std_msgs/Header header  # 데이터를 담을 부분
  uint32 seq
  time stamp
  string frame_id
geometry_msgs/Quaternion orientation    # 데이터를 담을 부분
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



## 참고 파일: imu_data_maker.py

IMU가 보내는 토픽을 받아서 `imu_data.txt` 파일을 만드는 파이썬 코드

우리가 만들 `imu_generator.py` 파일은 `imu_data_maker.py`와 정반대의 기능을 한다.

{% highlight python %}
#!/usr/bin/env python
import rospy, math, os
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

degrees2rad = float(math.pi) / float(180.0)
rad2degrees = float(180.0) / float(math.pi)

name = " >> ./imu_data.txt"

def listener():
    rospy.init_node("imu_data_maker", anonymous=False)
    rospy.Subscriber("imu", Imu, callback)

def call_back(data):
    global degrees2rad
    global rad2degrees

    euler = euler_from_quaternion((data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w))
    euler = [euler[0], euler[1], euler[2]]

    save_str = "roll: " + str(euler[0]) + ", " + "pitch: " + str(euler[1]) + ", " + "yaw: " + str(euler[2])
    command = 'echo "' + save_str + '" >> ./imu_data.txt'
    print(command)
    os.system(command)

if __name__ == "__main__":
    listener()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
{% endhighlight %}



## IMU 토픽을 발행하는 노드 제작

`imu_generator.py` 파이썬 코드 작성

{% highlight python %}
#!/usr/bin/env python
import rospy, math, os
from sensor_msgs.msg import Imu
from tf.transformations import quaternion_from_euler

# Publisher 노드 선언하고, Imu 타입의 토픽 정의
rospy.init_node( ... )
rospy.Publisher( ... )

# imu_data.txt 파일에서 한 줄식 읽어다가,
# 토픽에 담아 밖으로 Publish

# 모두 다 읽어서 보냈으면 종료
{% endhighlight %}



## 새로운 Launch 파일 작성

`launch` 디렉토리 아래에 `imu_generator.launch` 파일 만들기

{% highlight XML %}
<launch>
    <!-- rviz display -->
    <node name="rviz_visualizer" pkg="rviz" type="rviz" required="true"
          args="-d $(find rviz_imu)/rviz/imu_generator.rviz" />
    </node>
    <node name="imu_generator" pkg="rviz_imu" type="imu_generator.py" />
</launch>
{% endhighlight %}



## 실행 결과

{% highlight bash %}
# roslaunch 파일 실행
$ roslaunch rviz_imu imu_generator.launch

# 발행되는 토픽 확인
$ rostopic echo /imu
{% endhighlight %}