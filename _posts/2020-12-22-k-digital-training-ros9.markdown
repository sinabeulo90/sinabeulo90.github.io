---
layout: post
title:  "[Day 2.5] 나만의 메시지 만들기"
date:   2020-12-22 05:00:00 +0900
categories:
    - "K-Digital Training"
    - "ROS 노드 통신 프로그래밍"
---

# 나만의 메시지 만들기



## 1. 디렉토리 구조

나만의 메시지(특정 자료구조체)를 만들 수 있다.

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── msg_send
        ├── CMakeList.txt   # 임의의 메시지를 추가할 경우, 수정 필요
        ├── package.xml     # 임의의 메시지를 추가할 경우, 수정 필요
        ├── msg             # 메시지 폴더
        │   └── my_msg.msg  
        └── src
{% endhighlight %}



## 2. 커스텀 메시지 구현

- 메시지 파일 생성 및 작성

{% highlight bash %}
$ cd ~/xycar_ws/src
$ roscd msg_send
$ mkdir msg
$ cd msg
$ vi my_msg.msg
{% endhighlight %}

{% highlight vim %}
string first_name
string last_name
int32 age
int32 score
string phone_number
int32 id_number
{% endhighlight %}



## 3. 커스텀 메시지 선언

package.xml 파일 아래 쪽에 내용 추가

{% highlight bash %}
$ vi package.xml
{% endhighlight %}

{% highlight XML %}
...
<build_depend>message_generation</build_depend>
<exec_depend>message_runtime</exec_depend>
...
{% endhighlight %}

CMakeLists.txt 파일 수정 (# 코멘트 삭제하고, 추가 삽입하고)

{% highlight CMake %}
find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
    message_generation      # 추가
)

add_message_files(      # 코멘트를 제거
    FILES
    my_msg.msg              # 추가
#   Message1.msg
#   Message2.msg
)

generate_messages(      # 코멘트 제거
    DEPENDENCIES
    std_msgs
)

catkin_package(
    CATKIN_DEPENDS message_runtime      # 추가
    # INCLUDE_DIRS include
    # LIBRARIES my_package
    # CATKIN_DEPENDS rospy std_msgs
    # DEPENDS system_lib
)
{% endhighlight %}



## 4. 빌드 및 커스텀 메시지 타입 확인

{% highlight bash %}
$ cm
$ rosmsg show msg_send/my_msg

string first_name
string last_name
int32 age
int32 score
string phone_number
int32 id_number

# 메시지 이름만으로도 타입 확인 가능
$ rosmsg show my_msg

[msg_send/my_msg]:
string first_name
string last_name
int32 age
int32 score
string phone_number
int32 id_number
{% endhighlight %}

- 아래의 디렉토리에서 생성된 msg 확인 가능
    - ~/xycar_ws/devel/lib/python2.7/dist-packages/msg_send/msg



## 5. 코드안에서 커스텀 메시지 사용법

- 소스 코드에서 import(Python) 하는 방법

{% highlight Python %}
from msg_send.msg import my_msg
{% endhighlight %}

- 다른 패키지에서도 custom msg 사용 가능
- 참고 링크: [ROS/Tutorials/CreatingMsgAndSrv - ROS Wiki](http://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv)



## 6. 커스텀 메시지 사용하기

메시지 발행 Publisher 노드: `msg_sender.py`

{% highlight bash %}
$ vi msg_sender.py
{% endhighlight %}

{% highlight Python %}
#!/usr/bin/env python
import rospy
from msg_send.msg import my_msg     # my_msg 사용

rospy.init_node('msg_sender', anonymous=True)
pub = rospy.Publisher('msg_to_xycar', my_msg)

# my_msg 안에 데이터 채워 넣기
msg = my_msg()
msg.first_name = "seongmok"
msg.last_name = "Byeon"
msg.id_number = 20121222
msg.phone_number = "010-2012-1222"

rate = rospy.Rate(1)

while not rospy.is_shutdown():
    pub.publish(msg)            # my_msg 토픽 발행하기
    print("sending message")
    rate.sleep()
{% endhighlight %}

메시지 구독 Subscriber 노드: `msg_receiver.py`

{% highlight bash %}
$ vi msg_receiver.py
{% endhighlight %}

{% highlight Python %}
#!/usr/bin/env python
import rospy
from msg_send.msg import my_msg     # my_msg 사용

def callback(msg):  # my_msg 데이터 꺼내기
    print("1. Name :", msg.last_name + msg.first_name)
    print("2. ID:", msg.id_number)
    print("3. Phone Number:", msg.phone_number)

rospy.init_node('msg_receiver', anonymous=True)
sub = rospy.Subscriber('msg_to_xycar', my_msg, callback)    # my_msg 토픽 구독하기

rospy.spin()
{% endhighlight %}

실행 권한 부여

{% highlight bash %}
$ chmod +x msg_sender.py smg_receiver.py
{% endhighlight %}

빌드 & 실행

{% highlight bash %}
# 터미널 #1
$ cm
$ roscore

# 터미널 #2
$ rosrun msg_send msg_receier.py

# 터미널 #3
$ rosrun msg_send msg_sender.py

# 터미널 #4 : 내가 만든 노드가 잘 동작하는지 확인
$ rqt_graph
{% endhighlight %}



### 7. Launch 파일로 실행하기

`m_sender_sr.launch` 파일 생성

{% highlight XML %}
<launch>
    <node pkg="msg_send" type="msg_sender.py" name="sender1"/>
    <node pkg="msg_send" type="msg_sender.py" name="sender2"/>
    <node pkg="msg_send" type="msg_receiver.py" name="receiver" output="screen"/>
</launch>
{% endhighlight %}

{% highlight bash %}
$ roslaunch msg_send m_send_sr.launch
{% endhighlight %}

내가 만든 노드가 잘 동작하는지 확인

{% highlight bash %}
$ rqt_graph
{% endhighlight %}
