---
layout: post
title:  "ROS 기초 실습"
date:   2020-12-21 03:00:00 +0900
categories:
    - "K-Digital Training"
    - "자율주행 데브코스"
---

## 1. ROS 설치

 - 실습 기자재인 모형차에는 TX2 보드가 탑재되어 있고, 이 보드에는 이미 Ubuntu가 설치되어 있으며 그 위에는 ROS kinetic이 설치되어 있음
 - 준비물: Linux Ubuntu 16.04


### ROS 설치 과정

1. ROS를 제공하는 Software Repository 등록
{% highlight bash %}
$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
$ cat /etc/apt/sources.list.d/ros-latest.list

deb http://packages.ros.org/ros/ubuntu xenial main
{% endhighlight %}

2. apt key 셋업: apt key 값은 바뀔수도 있으므로, [공식 홈페이지](http://wiki.ros.org/kinetic/Installation/Ubuntu)에서 확인 가능
{% highlight bash %}
$ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
{% endhighlight %}

3. 패키지 설치
{% highlight bash %}
$ sudo apt-get update
# Desktop용 ROS 패키지 설치
# (ROS, RQT, RVIZ, 로봇 관련 각종 라이브러리 포함)
$ sudo apt-get install ros-kinetic-desktop-full
{% endhighlight %}

4. rosdep 초기화
{% highlight bash %}
# ROS의 핵심 컴포넌트를 사용하거나 컴파일 할 때,
# 의존성 패키지를 쉽게 설치하여 사용자 편의성을 높여주는 기능
$ sudo rosdep init
$ rosdep update
{% endhighlight %}

5. 쉘 환경 설정
{% highlight bash %}
$ echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc$ source ~/.bashrc
{% endhighlight %}

6. 추가로 필요한 도구 등 설치
{% highlight bash %}
$ sudo apt install python-rosinstall python-rosinstall-generator python-wstool build-essential
{% endhighlight %}

7. ROS 설치 확인
{% highlight bash %}
$ roscore
...
ROS_MASTER_URL=http://ubuntu:xxxx/  # ROS_MASTER_URL이 표시되면 정상 작동 의미
...

# 다른 터미널에서 실행
$ rosnode list
/rosout     # 기본적은 standard out, 디버그 노드를 의미
{% endhighlight %}



## 2. ROS 환경 설정


### 2.1 ROS Workspace

- ROS에서 프로그래밍을 하기 위해서는 Workspace라는 작업 공간 필요
- 실습 또는 실습 차량의 작업공간을 통일하기 위해 Workspace를 `xycar_ws`로 명명

{% highlight bash %}
# "xycar_ws" Workspace 생성
$ cd    # Home 폴더로 이동
$ mkdir -p ~/xycar_ws/src   # 서브 폴더 생성
$ cd xycar_ws   # xycar_ws 폴더로 이동
{% endhighlight %}


### 2.2 빌드 명령: `catkin_make`
- Workspace에 새로운 소스코드 파일이나 패키지가 만들어지면, `catkin_make` 명령으로 빌드(build) 작업 진행
- ROS 프로그래밍 작업과 관련 있는 모든 것들을 깔끔하게 정리해서 최신 상태로 만드는 작업이라고 생각하면 됨.

{% highlight bash %}
$ catkin_make   # ROS 빌드(코딩 환경 셋업 및 정리)
{% endhighlight %}

- 앞서 작업이 끝나면 아래와 같은 구조의 Workspace가 만들어짐
{% highlight bash %}
~/xycar_ws/     # xycar_ws: ROS 프로그래밍 작업을 하는 Workspace
├── build
├── devel
└── src         # 소스코드를 넣는 위치
    └── CMakeLists.txt
{% endhighlight %}


### 2.3 ROS 작업환경 설정
- ~/.bashrc에 설정 추가

{% highlight bash %}
...

alias h='history'
alias cw='cd ~/xycar_ws'
alias cs='cd ~/xycar_ws/src'
alias cm='cd ~/xycar_ws && catkin_make'
source /opt/ros/kinetic/setup.bash
source ~/xycar_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
{% endhighlight %}

- ROS 환경 변수 설정을 확인
{% highlight bash %}
$ source ~/.bashrc
$ printenv | grep ROS

ROS_ROOT=/opt/ros/kinetic/share/ros
ROS_PACKAGE_PATH=/home/seongmok/xycar_ws/src:/opt/ros/kinetic/share
ROS_MASTER_URI=http://localhost:11311
ROS_PYTHON_VERSION=2
ROS_VERSION=1
ROS_HOSTNAME=localhost
ROSLISP_PACKAGE_DIRECTORIES=/home/seongmok/xycar_ws/devel/share/common-lisp
ROS_DISTRO=kinetic
ROS_ETC_DIR=/opt/ros/kinetic/etc/ros
{% endhighlight %}



## 3. ROS 예제 프로그램 구동 실습


## 3.1 실습 내용
- 전체 구성
    - Publisher: `turtle_teleop_key` 노드 (`P 노드`)
        > /teleop_turtle (자신의 노드 이름)  
        > /turtle1/cmd_val (토픽 이름)  
        > geometry_msgs/Twist (메시지 타입)
    - Subscriber: `turtlesim_node` 노드 (`S 노드`)
        > /turtlesim  
        > /turtle1/cmd_vel  
        > geometry_msgs/Twist
    - Topic: /turtle1/cmd_vel
    - Master: ROS Core
        1. `S 노드` 노드가 자신의 정보를 `Master`에게 알려준다.
        2. `P 노드` 노드 또한 자신의 정보를 `Master`에게 알려준다.
        3. `Master`는 `S 노드` 노드가 원하는 정보를 `P 노드` 노드가 제공해 준다는 것을 감지하고, `S 노드` 노드에게 `P 노드` 노드의 정보를 전달
        4. `S 노드` 노드가 `P 노드` 노드에게 접속 요청
        5. `P 노드` 노드은 `S 노드` 노드의 접속에 응답하고, `S 노드`에게 토픽을 전송


## 3.2 우분투 터미널 사용법
- 터미널: 명령 실행을 위한 윈도우 창, 터미널 창 각각이 별개의 컴퓨터라고 생각해도 됨
    - `터미널 #1` : `roscore` 실행
    - `터미널 #2` : `turtlesim_node` 노드 실행
    - `터미널 #3` : `teleop_turtle_key` 노드 실행
    - `터미널 #4` : `rosnode` 실행, `rostopic` 실행

1. ROS Core 실행
        
{% highlight bash %}
# 터미널 #1: 마스터(roscore) 실행
$ roscore

# 터미널 #4: 현재 동작중인 ROS Node 확인
$ rosnode list

/rosout
{% endhighlight %}

{:start="2"}
2. ROS Node 실행: 토픽을 받아서 turtle을 이동시킴

{% highlight bash %}
# 터미널 #2: turtlesim_node 노드 실행
# 새로운 창이 만들어지고, 화면에 turtle 모양이 나타난다.
$ rosrun turtlesim turtlesim_node

# 터미널 #4: 현재 동작중인 ROS Node 확인
$ rosnode list

/rosout
/turtlesim
{% endhighlight %}

{:start="3"}
3. ROS 노드 실행: 사용자 입력에 맞춰 토픽을 발행함

{% highlight bash %}
# 터미널 #3: turtlesim_node 노드 실행
# 터미널 #3에서 키보드 방향키를 누르면, 키보드 입력값이 토픽에 담겨서 보내는 작업을 한다.
# 화면에 turtle이 입력한 키보드 방향키에 따라 움직인다.
$ rosrun turtlesim turtle_teleop_key
{% endhighlight %}

{:start="4"}
4. 노드 사이에서 토픽(topic) 주고 받기
- 노드 `/telop_turtle`는 사용자의 키보드 입력에 따라 토픽을 발행 —--> 노드 `/turtlesim`은 해당 토픽을 구독하여 키보드 입력값을 알아내고, 이에 따라 turtle을 이동시킴

{% highlight bash %}
# 터미널 #4: 현재 동작중인 ROS Node 확인
$ rosnode list

/rosout
/teleop_turtle
/turtlesim
{% endhighlight %}

{:start="5"}
5. 참고 사항

{% highlight bash %}
# 패키지 이름: turtlesim, 노드의 이름: turtlesim_node
$ rosrun turtlesim turtlesim_node
# 패키지 이름: turtlesim, 노드의 이름: turtle_teleop_key
$ rosrun turtlesim turtle_teleop_key

# turtle를 조종하는데 사용되는 실제 토픽 출력
$ rqt_graph
{% endhighlight %}


### 3.3 토픽을 조사해 보자

1. 토픽 확인

{% highlight bash %}
# 터미널 #4: 현재 어떤 토픽들이 날라다니는 지 확인 가능
$ rostopic list

/rosout         # ROS Core가 주고 받는 토픽
/rosout_agg     # ROS Core가 주고 받는 토픽
/turtle1/cmd_vel        # turtle과 관련된 토픽
/turtle1/color_sensor   # turtle과 관련된 토픽
/turtle1/pose           # turtle과 관련된 토픽
{% endhighlight %}

{:start="2"}
2. 노드와 토픽 관계를 시각화

{% highlight bash %}
# 터미널 #4
$ rqt_graph
{% endhighlight %}

{:start="3"}
3. 토픽을 좀 더 자세히 살펴보면

{% highlight bash %}
$ rostopic list -v

Published topics:
 * /turtle1/color_sensor [turtlesim/Color] 1 publisher
 # 토픽 이름: /turtle1/cmd_vel
 # 데이터 타입: geometry_msgs/Twist
 # 제공하는 Publisher 수: 1개
 * /turtle1/cmd_vel [geometry_msgs/Twist] 1 publisher
 * /rosout [rosgraph_msgs/Log] 2 publishers
 * /rosout_agg [rosgraph_msgs/Log] 1 publisher
 * /turtle1/pose [turtlesim/Pose] 1 publisher

Subscribed topics:
 # 토픽 이름: /turtle1/cmd_vel
 # 데이터 타입: geometry_msgs/Twist
 # 제공하는 Subscriber 수: 1개
 * /turtle1/cmd_vel [geometry_msgs/Twist] 1 subscriber
 * /rosout [rosgraph_msgs/Log] 1 subscriber


# 메시지의 타입과 구성을 살펴보자
$ rostopic type /turtle1/cmd_vel

geometry_msgs/Twist     # 이 토픽에 발행되는 메시지의 타입(type) 출력
                        # 메시지의 타입은 개발자가 정의할 수 있음


$ rosmsg show geometry_msgs/Twist   # 해당 토픽의 메시지가 어떻게 구성되어 있는지 출력

geometry_msgs/Vector3 linear
  float64 x
  float64 y
  float64 z
geometry_msgs/Vector3 angular
  float64 x
  float64 y
  float64 z
{% endhighlight %}

{:start="4"}
4. ROS 토픽의 메시지가 궁금할 때...

[ROS.org](http://wiki.ros.org) 에서 geometry_msgs/Twist 검색


### 3.4 토픽을 직접 발행해 보자

1. 토픽 직접 발행
- ROS 해킹 가능
- Subscriber는 자신이 원하는 토픽을 특정된 Publisher에 따라 구분지어 받지 않는다. 이는 데이터 통신을 간편하게 한 수도 있지만, 한편으로는 보안에 위협이 된다.

{% highlight bash %}
# rostopic pub [토픽] [메시지 타입] -- [메시지 내용 ...]
# -1: 한 번만 발행
$ rostopic pub -1 /turtle/cmd_vel geometry_msgs/Twist -- "[2.0, 0.0, 0.0]""[0.0, 0.0, 1.8]"
{% endhighlight %}

2. 주기적으로 반복 발행되는 메시지
{% highlight bash %}
# -r 1: 발행 주기, (1Hz, 1초에 한번씩 발행)
$ rostopic pub /turtle1/cmd_vel geometry_msgs/Twist -r 1 -- "[2.0, 0.0, 0.0]" "[0.0, 0.0, 1.8]"
{% endhighlight %}
