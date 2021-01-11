---
layout: post
title:  "[실습] IMU 센싱 데이터 시각화"
date:   2021-01-11 06:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

## IMU 관련 RVIZ Plug-in 설치

- rviz_imu_plugin.tgz 파일을 `~/xycar_ws/src`폴더에 갖다 놓고 압축 풀기
- 빌드한 이후에는 RVIZ를 실행시키면, IMU Plug-in이 자동으로 적용된다.

{% highlight bash %}
$ tar xzvf rviz_imu_plugin.tgz
$ cm
{% endhighlight %}

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── rviz_imu_plugin
        └── src
            ├── *.cpp
            └── *.h
{% endhighlight %}



## 패키지 생성

- ROS workspace의 src 폴더에서 rviz_imu 패키지를 만들고, 서브폴더로 /launch, /rviz 폴더를 만든다.

{% highlight bash %}
$ catkin_create_pkg rviz_imu rospy tf geometry_msgs urdf rviz xacro
{% endhighlight %}

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── rviz_imu
        ├── launch
        │   └── imu_3d.launch
        ├── rviz
        │   └── imu_3d.rviz
        └── src
{% endhighlight %}



## Launch 파일 작성(imu_3d.launch)

{% highlight XML %}
<launch>
    <!-- rviz display -->
    <node name="rviz_visualizer" pkg="rviz" type="rviz" required="true"
          args="-d $(find rviz_imu)/rviz/imu_3d.rviz" />
    <node pkg="xycar_imu" type="9dof_imu_node.py" name="xycar_imu" output="screen">
        <param name="rviz_mode" type="string" value="false" />
    </node>
</launch>
{% endhighlight %}



## RVIZ 실행(IMU Viewer)

{% highlight XML %}
$ roslaunch rviz_imu imu_3d.launch
{% endhighlight %}

플러그인 추가: RVIZ Display 탭 하단에 Add 버튼을 클릭한 후, rviz_imu_plugin 아래의 Imu를 선택하고 OK 버튼을 클릭한다.

Display Tab의 추가한 Imu 설정
1. Topic 옵션 값을 `/imu` 선택 또는 입력
2. Box properties의 `Enable box` 값을 체크
    - IMU 데이터를 시각화하기 위해 육면체를 Grid 위에 출력
    - Scale: 육면체의 크기
    - Color: 육면체의 색
    - Alpha: 육면체의 투명도
3. Axes properties의 `Enable axes` 값을 체크
    - IMU 데이터를 시각화하기 위해 축을 Grid 위에 출력
    - Scale: 축의 크기
4. 세팅값을 바꾸어 자신만의 뷰어를 제작해 보자.

RVIZ 창을 닫으려고 하면, imu_3d.rviz 파일을 rviz 폴더에 저장하겠냐는 창이 열리는데, 이때 Save를 한다.



## 결과 확인

{% highlight XML %}
$ roslaunch rviz_imu imu_3d.launch
$ rqt_graph
{% endhighlight %}
