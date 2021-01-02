---
layout: post
title:  "[Day 2.3] 노드 통신을 위한 패키지 만들기"
date:   2020-12-22 03:00:00 +0900
categories:
    - "K-Digital Training"
    - "ROS 노드 통신 프로그래밍"
---

# 노드 통신을 위한 패키지 만들기



## 1. 전체 구성
- teacher 노드(Publisher) --- my topic(Topic) 전송 ---> student 노드(Subscriber)
- 토픽: 편지봉투 ≈ 메시지: 편지 내용 ≈ 데이터



## 2. 디렉토리 구조

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── msg_send
        ├── launch
        │   └── m_send.launch
        └── src
            ├── student.py
            └── teacher.py
{% endhighlight %}



## 3. 패키지 만들기

{% highlight bash %}
# 패키지를 담을 디렉토리로 이동
$ cd ~/xycar_ws/src

# 패키지 새로 만들기
# 패키지 이름: msg_send
# 이 패키지가 의존하고 있는 다른 패키지들: std_msgs rospy
$ catkin_create_pkg msg_send std_msgs rospy

# msg_send 폴더 아래에 Launch 디렉토리 만들기
$ mkdir launch

# 새로 만든 패키지를 빌드
$ cm
{% endhighlight %}



## 4. 파이썬 코드 구현하기

토픽을 발행하고 구독하는 예제 Python 코드

- 토픽 이름: `my_topic`
- Publisher: `teacher.py`, 토픽에 "call me please" 담아서 전송
- Subscriber: `student.py`, 토픽을 받아서 내용을 꺼내서 화면에 출력



### 4.1 teacher.py

{% highlight Python %}
#!/usr/bin/env python
import rospy
from std_msgs.msg import String

# 'teacher' 노드를 새로 만든다.
rospy.init_node('teacher')

# 'my_topic' 이라는 이름으로 String 메시지를 발행하는 Publisher 생성
pub = rospy.Publisher('my_topic', String)

rate = rospy.Rate(2)    # 초당 2번 실행되도록 time slot을 생성
                        # 0.5초마다 실행되는 time slot

while not rospy.is_shutdown():
    pub.publish('call me please')   # 0.2초 사용한다면
    rate.sleep()                    # 0.3초 대기한 뒤에 재개
{% endhighlight %}



### 4.2 student.py

{% highlight Python %}
#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def callback(msg):
    print msg.data

# 'student' 노드를 새로 만든다.
rospy.init_node('student')

# String 메시지를 담고 있는 'my_topic' 이라는 토픽을 구독하고,
# 해당 토픽을 도착할 때마다 callback 함수를 호출하는 Subscriber 생성
sub = rospy.Subscriber('my_topic', String, callback)

rospy.spin()    # 무한루프
{% endhighlight %}



## 5. Launch 파일 작성하고 실행하기

Launch 파일 작성

{% highlight bash %}
$ vi m_send.launch
{% endhighlight %}

{% highlight XML %}
<launch>
    <node pkg="msg_send" type="teacher.py" name="teacher"/>
    <node pkg="msg_send" type="student.py" name="student" output="screen"/>
</launch>
{% endhighlight %}

{% highlight bash %}
$ cm
{% endhighlight %}

Python 파일에는 실행 권한을 부여해야 한다.

{% highlight bash %}
$ chmod +x teacher.py student.py
{% endhighlight %}

Launch 파일 실행 방법

{% highlight bash %}
$ roslaunch msg_send m_send.launch
{% endhighlight %}

내가 만든 노드가 잘 동작하는지 확인

{% highlight bash %}
$ rqt_graph
{% endhighlight %}



### 정리하면

전체 작업 과정

{% highlight bash %}
$ cd ~/xycar_ws/src
$ catkin_create_pkg msg_send std_msgs rospy
$ mkdir ~/xycar_ws/src/msg_send/launch
$ cd ~/xycar_ws/src/msg_send/src
$ vi student.py
$ vi teacher.py
$ chmod +x student.py teacher.py
$ cd ~/xycar_ws/src/msg_send/launch
$ vi m_send.launch
$ cm
$ roslaunch msg_send m_send.launch
{% endhighlight %}
