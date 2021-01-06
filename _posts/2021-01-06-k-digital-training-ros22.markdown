---
layout: post
title:  "로봇 모델링 언어 URDF 소개"
date:   2021-01-06 01:00:00 +0900
categories:
    - "K-Digital Training"
    - "자율주행 데브 코스"
---

## 로봇 모델링 언어 URDF 소개

ROS는 로봇을 3D로 모델링하는 URDF 언어를 지원한다. 로봇의 3D 형상 및 외관, 관성 등 물리적 특성 등을 XML 언어로 정의하여, RVIZ에서 3차원으로 시각화 또는 Gazebo에서 물리 시뮬레이션을 가능하게 한다.

URDF: Unified Robot Description Format



## URDF 좌표계와 단위

- 좌표계
    - 위치나 크기를 표현하기 위해 데카르트 좌표계 x, y, z를 사용
    - 회전을 표현하기 위해 오일러 각도 roll(x축 회전), pitch(y축 회전), yaw(z축 회전)를 사용

- 단위
    - 길이: m, 미터
    - 각도: radian, 라디안
    - 질량: kg, 킬로그램
    - 속도: m/s, 병진/직진운동
    - 각속도: radian/s, 회전운동



## Radian, 라디안

- 라디안, 호도법: ($\pi$ = 3.14)

| 60분법 | $0^\circ$ | $30^\circ$ | $45^\circ$ | $60^\circ$ | $90^\circ$ | $120^\circ$ | $135^\circ$ | $150^\circ$ | $180^\circ$ |
| 호도법 | $0$ | $\frac{\pi}{6}$ | $\frac{\pi}{4}$ | $\frac{\pi}{3}$ | $\frac{\pi}{2}$ | $\frac{2}{3}\pi$ | $\frac{3}{4}\pi$ | $\frac{5}{6}\pi$ | $\pi$ |

- 1라디안 = $180^\circ / \pi$ = $180^\circ / 3.14$ = $57.3^\circ$
- $\pi$ 라디안 = 1라디안 x $\pi$ = $180^\circ / \pi$ x $\pi$ = $180^\circ$
- $\pi$ 라디안 = 1라디안 x $\pi$ = $57.3^\circ$ x 3.14 = $180^\circ$



## URDF 형상과 색상 표현

- 형상 표현
    - cylinder, 원통
    - box, 상자
    - sphere, 공

- 색상 표현: RGB 3원색과 투과율을 0 ~ 1 사이의 숫자로 정의



## URDF 기구 표현

- 기구의 표현
    - Base: 고정 부분(grounded)
    - Link: 관절에 연결되는 로봇 팔의 부분
    - Joint: 링크를 연결하는 부위로 보통 모터의 회전을 통해 움직임을 만듦

- Joint의 동작 정의
    - fixed: 고정
    - revolute: 작동 범위 제한
    - continuous: 연속 회전


연결되는 2개의 Link는 각각 Parent link, Child link로 표현될 수 있으며, Base부터 만들어지는 Link를 따라 Parent-Child 쌍을 가지며, 새로 만드는 Link는 Child, Child link를 만들기 위해 기준이 되는 Link는 다시 Parent link로 표현된다.



## URDF - XML

[ROS URDF의 기초예제 Pan/Tilt 시스템](https://pinkwink.kr/1007)

Link 태그의 구성

{% highlight XML %}
    <link name="base_link">
        <!-- <visual>: 시각화를 위해 형상과 위치를 정의 -->
        <visual>
            <!-- <geometry>: 형상 정의(원통, 상자, 공) -->
            <geometry>
                <cylinder length="0.01" radius="0.2"/>
            </geometry>
            <!-- <origin>: 고정축을 기준으로 link 형상의 roll, pitch, yaw 위치를 라디안으로 나타내고, x, y, z 좌표 위치를 미터 단위로 지정 -->
            <origin rpy="0 0 0" xyz="0 0 0">
            <!-- <material>: 형상의 컬러값을 지정 -->
            <material name="yellow">
                <color rgba="1 1 0 1"/>
            </material>
        </visual>
        ...
    </link>
{% endhighlight %}

Joint 태그의 구성: 형상으로 표현되지 않음

{% highlight XML %}
    <joint name="pan_joint" type="revolute">
        <!-- <parent>: parent frame의 이름을 지정해서 child frame과 연결 -->
        <parent link="base_link"/>
        <!-- <child>: child frame의 이름을 지정해서 parent frame과 연결 -->
        <child link="pan_link"/>
        <!-- <origin>
            - 고정축을 기준으로 형상의 roll, pitch, yaw 위치를 라디안으로 나타내고, x, y, z 좌표 위치를 미터 단위로 지정
            - joint는 parent의 origin을, child는 joint의 origin을 고정축으로 정함 -->
        <origin xyz="0 0 0.1"/>
        <axis xyz="0 0 1"/>
        <!-- <limit>
            - Joint의 운동 범위 제한값을 지정
            - lower: revolute type의 joint에서 최저각(radian)지정
            - upper: revolute type의 joint에서 최대각(radian)지정
            - effort: N힘의 최대값 지정
            - velocity: radian/s 최대속도 지정 -->
        <limit effort="300" velocity="0.1" lower="-3.14" upper="3.14"/>
        <dynamics damping="50" friction="1"/>
    </joint>
{% endhighlight %}
