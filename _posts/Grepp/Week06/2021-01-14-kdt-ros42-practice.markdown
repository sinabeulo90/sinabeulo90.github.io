---
layout: post
title:  "[과제] RVIZ에서 모터와 센서 통합하기"
date:   2021-01-14 01:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

## 과제 설명

RVIZ 가상 공간에서 8자 주행하는 Xycar 3D 모델에 라이다 센서와 IMU 센서를 통합해 보자. 주변 장애물까지의 거리 값을 Range로 표시하고, IMU 센싱 값에 따라 차체가 기울어지게 만든다.

![rviz_all Diagram](/assets/grepp/rviz_all_diagram1.png)

![rviz_all Diagram](/assets/grepp/rviz_all_diagram2.png)

## 패키지 생성

{% highlight bash %}
$ catkin_create_pkg rviz_all rospy tf geometry_msgs urdf rviz xacro
{% endhighlight %}

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── rviz_all
        ├── launch
        │   └── rviz_all.launch
        ├── rvix
        │   └── rviz_all.rviz
        ├── src
        │   └── odom_imu.py
        └── urdf
            └── rviz_all.urdf
{% endhighlight %}



##  URDF 만들기(rviz_all.urdf)

xycar_3d.urdf + lidar_urdf.urdf



## RVIZ 만들기(rviz_all.rviz)

- rviz_odom.rviz 파일을 복사해서 사용
- 여기에는 라이다 센서의 시각화에 쓰이는 `Range`가 없으므로 설정 메뉴에서 추가한다. 
- 이동 궤적 표시를 위해 `Odometry` 추가
    - Topic: /odom
    - Keep: 100
    - Sharft Length: 0.05
    - Head Length: 0.1



## odom_imu.py 생성하기

- odometry 데이터를 생성하는 rviz_odom.py를 수정해 odom_imu.py 제작
- imu 토픽을 구독하여 획득한 쿼터니언 값을 odometry 데이터에 넣어준다.

{% highlight Python %}
odom_quat = Imudata

odom_broadcaster.sendTransform(
    (x_, y_, 0.),
    odom_quat,
    current_time
    "base_link",    # IMU 값에 따라 차체가 움직이도록 하기 위해
                    # Odometry 정보를 차체에 해당하는 base_link에 연결
    "odom"
)
{% endhighlight %}



## Laun 파일 생성하기(rviz_all.launch)

{% highlight XML %}
<launch>
    <param name="robot_description" textfile="$(find rviz_all)/urdf/rviz_all.urdf" />
    <param name="use_gui" value="true" />
    <node name="rviz_visualizer" pkg="rviz" type="rviz" required="true"
          args="-d $(find rviz_all)/rviz/rviz_all.rviz" />
    <node name="robot_state_publisher" pkg="robot_state_publisher" type="state_publisher" />
    
    <!-- 자동차 8자 주행: odom_8_drive.py, odom_imu.py, converter.py -->
    <!-- 라이다 토픽 발행: rosbag, lidar_urdf.py -->
    <!-- IMU 토픽 발행: imu_generator.py -->
{% endhighlight %}



## 결과 확인

{% highlight bash %}
$ roslaunch rviz_all rviz_all.launch
{% endhighlight %}
