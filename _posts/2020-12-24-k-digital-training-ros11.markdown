---
layout: post
title:  "[Day 4.1] ROS 노드간 원격 통신"
date:   2020-12-24 01:00:00 +0900
categories:
    - "K-Digital Training"
    - "ROS 노드 통신 프로그래밍"
---

## 1. ROS 노드간 통신

ROS에서 노드와 노드가 서로 통신하면서 협업

- 통신 프로토콜(TCP/IP 계열)
    - 무선 Wifi, 유선 이더넷, 광 통신 등 네트워크 접속에 사용하는 통신 프로토콜
    - XMLRPC 프로토콜: 노드들이 마스터와 통신할 때 이용, http방식으로 통신하는 기법
    - TCPROS 프로토콜: 각 노드간 통신할 때 이용
- LOCAL에서 동작한다고 하더라도, ROS 에서는 인터넷 통신 방식을 사용해서 동작한다.

- 마스터

> XMLRPC: 서버
>
> http://ROS_MASTER_URI:1131

- 노드1

> TCPROS: 서버
>
> ROS_HOSTNAME:3456
>
> 정보 발행

- 노드2

> TCPROS: 클라이언트
>
> ROS_HOSTNAME:7890
>
> 정보 구독


## 2. 여러가지 통신 방식

하나의 장치 안에서 여러 노드가 서로 통신

- 노드는 OS 도움을 받아 각 하드웨어 장비마다 서로 통신한다.
- 노드들은 마스터의 도움을 받아서 서로 메시지를 주고 받는다.
- 노드와 마스터가 LOCAL에 있어도, 인터넷 통신 방식으로 동작한다.


## 3. 노드와 노드간 통신

ROS에서 노드와 노드가 서로 네트워크 통신하면서 협업

- 서로 다른 하드웨어 사이에서 네트워크로 통신
- 단일 하드웨어 안에 있는 노드끼리도 네트워크로 통신
- ex. ROS1(OS1: Linux) <--- TCP/UDP ---> ROS2(OS2: Linux)
- 노드-마스터 통신 : 노드의 요청이 OS에게 인터넷 통신을 부탁하고, 네트워크를 통해 마스터가 있는 PC로 전달되고 OS에서 그 요청을 마스터에게 전달한다.



## 4. 원격 통신 예제 구동

### 4.1 ROS 원격 통신 예제 구동 환경

전체 구성

- PC 쪽 키보드로 Xycar에 있는 거북이 조종
- PC --- 토픽 ---> Xycar
- 노드1에서 노드2로 /turtle1/cmd_vel 토픽 전송


- PC: Publisher 노드의 정보
> (turtle_teleop_key)
>
> /teleop_turtle
>
> /turtle1/cmd_vel
>
> geometry_msgs/Twist
>
> http://192.168.1.25


- Xycar: Subscriber 노드의 정보 (turtlesim_node)
> /turtlesim
>
> /turtle1/cmd_vel
>
> geometry_msgs/Twist
>
> http://192.168.1.12


- 마스터 core: 노드 정보 관리
> http://192.168.1.25:11311 (만천하에 공개된 주소)


### 4.2 N:1 원격 통신 구동

학생 1, 학생 2, 학생 3 ---> Xycar (마스터 Core)

- 학생: 발행자 노드 (turtle_teleop_key)
- Xycar: 마스터 Core, 구독자 노드 (turtlesim_node)



### 4.3 ROS 원격통신을 위한 환경 설정

IP 주소 설정

~/.bashrc 내용 수정

{% highlight bash %}
$ vi ~/.bashrc  # 아래 내용 수정
{% endhighlight %}

{% highlight vim %}
...
# roscore가 구동되는 장치(Xycar)의 IP 주소
export ROS_MASTER_URI=http://192.168.0.5:11311
# 내 PC/노트북의 IP 주소
# $ ifconfig 명령으로 확인 가능
export ROS_HOSTNAME=192.168.0.11
...
{% endhighlight %}

수정한 내용을 시스템에 반영

{% highlight bash %}
$ source ~/.bashrc
{% endhighlight %}


### 4.4 ROS 원격 통신 실습

돌려보기

{% highlight bash %}
$ rosrun turtlesim turtle_teleop_key
{% endhighlight %}

노드간 메시지 흐름 확인: 새로운 터미널 창을 열어서 작업

{% highlight bash %}
$ rostopic list
$ rosnode info /turtlesim       # 원격에서 작동하고 있는 /turtlesim 노드 정보 출력
$ rostopic echo /turtle1/cmd_vel
$ rqt_graph
{% endhighlight %}
