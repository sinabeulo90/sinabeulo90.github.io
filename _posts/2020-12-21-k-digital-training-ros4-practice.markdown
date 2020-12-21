---
layout: post
title:  "ROS 프로그래밍 기초"
date:   2020-12-21 04:00:00 +0900
categories:
    - "K-Digital Training"
    - "자율주행 데브코스"
---

## 1. ROS Package 기초



### 1.1 ROS 패키지(package)

- 패키지: ROS에서 개발되는 소프트웨어를 논리적 묶음으로 만든 것
- 하나의 프로젝트가 하나의 패키지 (ex. 보행자 추적 프로그램 ⇒ 보행자 추적 패키지)

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    ├── CMakeLists.txt
    └── pkg_name        # 새로운 패키지가 만들어 지는 곳
            ├── CMakeLists.txt
            ├── package.xml
            ├── launch      # 파일을 만들어 넣는 부분
            │   └── C.launch
            └── src         # 파일을 만들어 넣는 부분
                ├── A.py
                └── B.py
{% endhighlight %}



### 1.2 ROS가 제공하는 편리한 명령어들

| 명령어 | 설명 |
| :- | :- |
| $ rospack list | 어떤 패키지가 있는지 나열 |
| $ rospack find [package_name] | 이름을 이용해서 패키지 검색 |
| $ roscd [location_name[/subdir]] | ROS 패키지 디렉토리로 이동 |
| $ rosls [location_name[/subdir]] | Linux ls와 유사 (경로를 몰라도 이름 적용 가능) |
| $ rosed [file_name] | (환경 설정에 따른) 에디터로 파일 편집 |



## 2. ROS 패키지 만들기

{% highlight bash %}
$ cd ~/xycar_ws/src # 패키지를 담을 디렉토리로 이동

# catkin_create_pkg name [dependencies [dependencies ...]]
# 패키지 이름: my_pkg1
# 의존할 다른 패키지들: std_msgs, rospy
$ catkin_create_pkg my_pkg1 std_msgs rospy
{% endhighlight %}



### 2.1 ROS 패키지 빌드

- 시스템에 패키지가 만들어졌다는 것을 알려주는 작업

{% highlight bash %}
# 방법 1
$ cd ~/xycar_ws
$ catkin_make

# 방법 2
$ cm    # ~/.bashrc 파일에서 alias 선언을 했기 때문에 한번에 실행할 수 있다.
{% endhighlight %}


### 2.2 만들어진 패키지 확인
{% highlight bash %}
# 패키지 my_pkg1의 위치 출력
$ rospack find my_pkg1      # rospack list | grep my_pkg1 명령어와 동일

/home/seongmok/xycar_ws/src/my_pkg1

# my_pkg1가 의존하고 있는 다른 패키지 출력
$ rospack depends1 my_pkg1

rospy
std_msgs


# 패키지 my_pkg1으로 이동
$ roscd my_pkg1
{% endhighlight %}



## 3. Publisher 코드 작성 - 프로그래밍



### 3.1 Publisher 구현
~/xycar_ws/src/my_pkg1/src 위치에 pub.py 파일 작성

{% highlight Python %}
#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist

# 노드 'my_node' 생성
# anonymous=True
#   - 각 노드는 유일한 이름을 가져야만 하므로,
#     노드 이름 뒤에 임의의 시리얼 번호를 붙여준다.
rospy.init_node('my_node', anonymous=True)  

# Publisher 객체 생성
# 토픽 이름: /turtle1/cmd_vel
# 메시지 타입: Twist
pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)

# 메시지 Twist의 데이터 값 설정
msg = Twist()
msg.linear.x = 2.0
msg.linear.y = 0.0
msg.linear.z = 0.0
msg.angular.x = 0.0
msg.angular.y = 0.0
msg.angular.z = 1.8

rate = rospy.Rate(1)    # 초 당 수행할 반복 횟수 설정
                        # ex. rospy.Rate(1/10) : 10초당 1회일 경우, 

# 1초에 한 번씩 발행. (1Hz)
while not rospy.is_shutdown():
    pub.publish(msg)    # 메시지 발행
    rate.sleep()        # 메시지 발행하는데 0.3초가 걸리면, 0.7초 뒤에 재개
{% endhighlight %}



### 3.2 프로그램 실행 권한 부여

작성한 파이썬 코드를 실행시키려면 실행권한이 있어야 하므로, 아래와 같이 실행권한을 부여해야 한다.

{% highlight bash %}
$ chmod +x pub.py
$ ls -l     # 실행권한 여부 확인
{% endhighlight %}



### 3.3 프로그램 실행과 확인

- `터미널 #1` : `roscore` 실행
- `터미널 #2` : `turtlesim_node` 노드 실행
- `터미널 #3` : `pub.py` 실행
- `터미널 #4` : `rqt_graph` 실행

{% highlight bash %}
# 터미널 #1: roscore 실행
$ roscoure

# 터미널 #2: turtlesim_node 노드 실행
$ rosrun turtlesim turtlesim_node

# 터미널 #3: pub.py 실행
$ chmod +x pub.py
$ rosrun my_pkg1 pub.py

# 터미널 #4: pub.py가 잘 동작하는지 확인
$ rqt_graph
$ rosnode list

/my_node_13983_1608564106076
/rosout
/teleop_turtle
/turtlesim
{% endhighlight %}



## 4. Subscriber 코드 작성 - 프로그래밍


### 4.1 어떤 토픽을 사용할 것인지 선택

- 우선, turtle이 어떤 토픽에 어떤 메시지를 발행하고 있는지 알아보자

{% highlight bash %}
# 토픽 리스트 출력
$ rostopic list

/rosout
/rosout_agg
/turtle1/cmd_vel
/turtle1/color_sensor
/turtle1/pose   # 구독할 토픽 선택


# 토픽의 메시지 타입 확인
$ rostopic type /turtle1/pose

turtlesim/Pose


# 메시지의 구성 확인
rosmsg show turtlesim/Pose

float32 x
float32 y
float32 theta
float32 linear_velocity
float32 angular_velocity


# 실제 토픽에서 어떤 메시지를 담고 있는지 출력
$ rostopic echo /turtle1/pose

...
angular_velocity: 1.79999995232
---
x: 5.02011299133
y: 5.66742134094
theta: 5.79362869263
linear_velocity: 2.0
angular_velocity: 1.79999995232
---
{% endhighlight %}



### 4.2 구독자(Subscriber) 구현

- ~/xycar_ws/src/my_pkg1/src 위치에, sub.py 파일 작성

{% highlight Python %}
#!/usr/bin/env python
import rospy
from turtlesim.msg import Pos

edef callback(data):
    s = "Location: %.2f, %.2f" % (data.x, data.y)
    rospy.loginfo(rospy.get_caller_id() + s)

# 노드 'my_listener' 생성
# anonymous=True
#   - 각 노드는 유일한 이름을 가져야만 하므로,
#     노드 이름 뒤에 임의의 시리얼 번호를 붙여준다.
rospy.init_node("my_listener", anonymous=True)

# Subscriber 객체 생성
# 구독하는 토픽 이름: /turtle1/pose
# 구독하는 토픽의 메시지 타입: Pose
# 메시지를 수신할 때, 호출할 함수: callback
#   - ROS 시스템에서 토픽이 도착했는지를 감지하고, 등록한 함수를 호출해준다.
rospy.Subscriber("/turtle1/pose", Pose, callback)

rospy.spin()    # 프로그램이 종료될때까지 무한루프
{% endhighlight %}



### 4.3 프로그램 실행 권한 부여

작성한 파이썬 코드를 실행시키려면 실행권한이 있어야 하므로, 아래와 같이 실행권한을 부여해야 한다.

{% highlight bash %}
$ chmod +x sub.py
$ ls -l     # 실행권한 여부 확인
{% endhighlight %}


### 4.3 프로그램 실행과 확인


- `터미널 #1` : `roscore` 실행
- `터미널 #2` : `turtlesim_node` 노드 실행
- `터미널 #3` : `pub.py` 실행
- `터미널 #4` : `sub.py` 실행

{% highlight bash %}
# 터미널 #1: roscore 실행
$ roscoure

# 터미널 #2: turtlesim_node 노드 실행
$ rosrun turtlesim turtlesim_node

# 터미널 #3: pub.py 실행
$ rosrun my_pkg1 pub.py

# 터미널 #4: sub.py 실행
$ chmod +x sub.py
$ rosrun my_pkg1 sub.py

# 새로운 터미널에서 sub.py가 잘 동작하는지 확인
$ rqt_graph
{% endhighlight %}



## 5. 정리하면

패키지 my_pkg1을 만들고, `pub.py`, `sub.py` 2개 파일을 작성

{% highlight bash %}
# 패키지 my_pkg1 생성 및 빌드
$ cd ~/xycar_ws/src
$ catkin_create_pkg my_pkg1 std_msgs rospy
$ cm

# pub.py, sub.py 작성 및 실행권한 부여
$ cd ~/sycar_ws/src/my_pkg1/src
$ vim sub.py
$ vim pub.py
$ chmod +x sub.py pub.py

# Publisher, Subscriber 노드 실행
$ rosrun my_pkg1 pub.py
$ rosrun my_pkg1 sub.py
{% endhighlight %}
