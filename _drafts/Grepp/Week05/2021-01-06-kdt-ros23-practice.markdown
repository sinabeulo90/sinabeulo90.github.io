---
layout: post
title:  "[실습] URDF 예제 패키지 제작 따라하기"
date:   2021-01-06 02:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

## RVIZ에서 회전 막대 모델 시각화 하기

- URDF 파일로 회전막대 모델링
- ROS 패키지 만들기
    - ex_urdf 패키지 생성
    - /urdf 폴더, .urdf 파일 만들기
    - /launch 폴더, .launch 파일 만들기
- RVIZ 실행해서 움직여 보기



## ex_urdf 패키지 생성

{% highlight bash %}
# ex_urdf 패키지 생성
$ catkin_create_pkg ex_urdf roscpp tf geometry_msgs urdf rviz xacro

# 서브 폴더 만들기
$ cd ex_urdf
$ mkdir urdf
$ touch urdf/pan_tilt.urdf

# 서브 폴더 만들기
$ mkdir launch
$ touch launch/view_pan_tilt_urdf.launch
{% endhighlight %}

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── ex_urdf
        ├── launch
        │   └── view_pan_tilt_urdf.launch
        └── urdf
            └── pan_tilt.urdf
{% endhighlight %}



## pan_tilt.urdf

{% highlight XML %}
<?xml version="1.0"?>
<robot name="ex_urdf_pan_tilt">
    <link name="base_link">
        <visual>
            <geometry>
                <cylinder length="0.01" radius="0.2"/>
            </geometry>
            <!-- 시작 위치: 0cm -->
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <material name="yellow">
                <color rgba="1 1 0 1"/>
            </material>
        </visual>

        <collision>
            <geometry>
                <cylinder length="0.03" radius="0.2"/>
            </geometry>
            <origin rpy="0 0 0" xyz="0 0 0"/>
        </collision>
        <inertial>
            <mass value="1"/>
            <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
        </inertial>
    </link>

    <joint name="pan_joint" type="revolute">
        <parent link="base_link"/>
        <child link="pan_link"/>
        <!-- Joint 위치: base_link 0 + 10cm = 10cm -->
        <origin xyz="0 0 0.1"/>
        <axis xyz="0 0 1"/>
        <limit effort="300" velocity="0.1" lower="-3.14" upper="3.14"/>
        <dynamics damping="50" friction="1"/>
    </joint>

    <link name="pan_link">
        <visual>
            <geometry>
                <cylinder length="0.4" radius="0.04"/>
            </geometry>
            <!-- 실린더 중앙점 위치: pan_joint 10cm + 9cm = 19cm -->
            <origin rpy="0 0 0" xyz="0 0 0.09"/>
            <material name="red">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>

        <collision>
            <geometry>
                <cylinder length="0.4" radius="0.06"/>
            </geometry>
            <origin rpy="0 0 0" xyz="0 0 0.09"/>
        </collision>

        <inertial>
            <mass value="1"/>
            <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
        </inertial>
    </link>

    <joint name="tilt_joint" type="revolute">
        <parent link="pan_link"/>
        <child link="tilt_link"/>
        <!-- Joint 위치: pan_link 10 + 20cm = 30cm -->
        <origin xyz="0 0 0.2"/>
        <axis xyz="0 1 0"/>
        <limit effort="300" velocity="0.1" lower="-4.71239" upper="-1.570796"/>
        <dynamics damping="50" friction="1"/>
    </joint>

    <link name="tilt_link">
        <visual>
            <geometry>
                <cylinder length="0.4" radius="0.04"/>
            </geometry>
            <origin rpy="0 1.570796 0" xyz="0 0 0"/>
            <material name="green">
                <color rgba="1 0 0 1"/>
            </material>
        </visual>

        <collision>
            <geometry>
                <cylinder length="0.4" radius="0.06"/>
            </geometry>
            <!-- 실린더 중앙점 위치: tilt_joint 30 + 0cm = 30cm -->
            <origin rpy="0 1.570796 0" xyz="0 0 0"/>
        </collision>

        <inertial>
            <mass value="1"/>
            <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
        </inertial>
    </link>
</robot>
{% endhighlight %}



## pan_tilt_urdf.launch

{% highlight XML %}
<launch>
    <arg name="model"/>
    <param name="robot_description" textfile="$(find ex_urdf)/urdf/pan_tilt.urdf"/>

    <!-- Setting gui parameter to true for display joint slider -->
    <param name="use_gui" value="true"/>
    <!-- Starting Joint state publisher node which will publish the joint values -->
    <node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher"/>
    <!-- Starting robot state publish which will publish tf -->
    <node name="robot_state_publisher" pkg="robot_state_publisher" type="state_publisher"/>
    <!-- Launch visualization in rviz -->
    <node name="rviz" pkg="rviz" type="rviz" args="-d $(find ex_urdf)/urdf.rviz" required="True"/>
</launch>
{% endhighlight %}



## 생성된 파일 확인

{% highlight bash %}
# 새로 생성된 패키지에 대해 빌드 진행
$ cm

# 작성된 urdf 파일에 문법적인 오류가 있는지 확인
$ cd src/ex_urdf_urdf
$ check_urdf pan_tilt.urdf

# Link와 joint 관계도를 PDF 파일로 만들어 준다.
$ urdf_to_graphiz pan_tilt.urdf
{% endhighlight %}



## joint-state-publisher-gui 패키지 설치 필요

joint-state-publisher-gui: URDF 파일로 모델링한 로봇의 Joint(관절) 부분을 움직이기 위한 윈도우 기반 GUI  ehrn

{% highlight bash %}
$ sudo apt update
$ sudo apt install ros-kinetic-joint-state-publisher-gui
{% endhighlight %}



## RVIZ 실행, 세팅 추가 및 수정

RVIZ 실행

{% highlight bash %}
# RVIZ 실행
$ roslaunch ex_urdf view_pan_tilt_urdf.launch
{% endhighlight %}

세팅 추가 및 수정

1. Add 버튼을 클릭하여, `By display type`탭에서 `RobotModel`과 `TF`를 추가한다.
2. 메인 창에서, `Display` 항목 아래에 `Global Options`의 `Fixed Frame` 값을 map 대신 `base_link`를 넣는다.

![URDF RVIZ](/assets/grepp/urdf_rviz.png)



## RVIZ 설정 내용 저장

이것 저것을 추가하고 수정한 내용을 저장하면, `ex_urdf` 폴더에 `urdf.rviz` 파일이 생성되고 다시 RVIZ를 실행시키면 저장했던 내용을 읽는다.
