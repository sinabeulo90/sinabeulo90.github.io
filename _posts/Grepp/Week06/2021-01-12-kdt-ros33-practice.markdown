---
layout: post
title:  "[실습] 라이다 센서 활용: RVIZ 시각화"
date:   2021-01-12 03:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

## 패키지 생성

ROS Workspace의 src 폴더에 `rviz_lidar`패키지를 만들고, 서브 폴더로 `/launch`, `/rviz`를 만든다.

{% highlight bash %}
$ catkin_create_pkg rviz_lidar rospy tf geometry_msgs urdf rviz xacro
{% endhighlight %}

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── rviz_lidar
        ├── launch
        │   └── lidar_3d.launch
        ├── rviz
        │   └── lidar_3d.rviz
        └── src
{% endhighlight %}



## Launch 파일 작성(lidar_3d.launch)

{% highlight XML %}
<launch>
    <!-- rviz display -->
    <node name="rviz_visualizer" pkg="rviz" type="rviz" required="true"
          args="-d $(find rviz_lidar)/rviz/lidar_3d.rviz" />

    <node name="xycar_lidar" pkg="xycar_lidar" type="xycar_lidar" output="screen" >
        <param name="serial_port" type="string" value="/dev/ttyRPL" />
        <param name="serial_baudrate" type="int" value="115200" />
        <param name="frame_id" type="string" value="laser" />
        <param name="inverted" type="bool" value="false" />
        <param name="angle_compenstate" type="bool" value="true" />
    </node>
</launch>
{% endhighlight %}




## 라이다 장치가 없을 경우(rosbag)

실제 라이다 장치를 대신하여 `/scan`토픽을 발행하는 프로그램을 이용할 수 있는데, ROS에서는 `rosbag` 프로그램을 이용할 수 있다. 이는 연구개발 또는 디버깅으로 사용할 수 있는 유용한 도구이다.

예를 들어, 라이다에서 발행하는 `\scan` 토픽을 `lidat_topic.bag` 파일이라는 이름으로 저장했다고 하자. 그럼 `rosbag` 프로그램으로 이 파일을 읽어들여 그 당시의 발행된 시간 간격에 맞춰 `\scan` 토픽을 똑같이 발행할 수 있다. 



## 라이다 대신 rosbag을 사용하는 경우



### lidar_3d_rosbag.launch 파일 작성

{% highlight XML %}
<launch>
    <!-- rviz display -->
    <node name="rviz_visualizer" pkg="rviz" type="rviz" required="true"
          args="-d $(find rviz_lidar)/rviz/lidar_3d.rviz" />

    <node name="rosbag_play" pkg="rosbag" type="play" output="screen" 
          required="true" args="$(find rviz_lidar)/src/lidar_topic.bag" />
</launch>
{% endhighlight %}



## RVIZ 실행(라이다 Viewer)

{% highlight bash %}
$ roslaunch rviz_lidar lidar_3d.launch
{% endhighlight %}

- Global Options 항목의 Fixed Frame 옵션 값을 `map` 대신 `laser`로 입력
- RVIZ Display 탭 하단에 Add 버튼을 클릭한 후 LaserScan을 선택하고 OK 버튼을 클릭
    1. Topic 옵션 값을 `/scan` 선택 또는 입력
    2. Size(m) 옵션 값을 `0.1`로 설정(단위: m)

RVIZ 창을 닫으려고 하면, lidar_3d.rviz 파일을 rviz 폴더에 저장하겠냐는 창이 열리는데, 이때 Save를 한다.
