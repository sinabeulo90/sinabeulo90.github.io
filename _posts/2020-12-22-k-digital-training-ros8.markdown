---
layout: post
title:  "[Day 2.4] 1:N, N:1, N:N 통신"
date:   2020-12-22 04:00:00 +0900
categories:
    - "K-Digital Training"
    - "ROS 노드 통신 프로그래밍"
---

# 1:N, N:1, N:N 통신



## 1. 다양한 통신 구성

- 1:N 통신
    - ex. teacher ---> [ student ] x N
    - ex. 카메라 데이터 ---> [ 인공지능 SW, 영상처리 SW, 영상 출력, ...]
- N:1 통신
    - ex. [ teacher ] x n ---> student
    - ex. [ 인공지능 SW, 알고리즘 SW, 장애물이 있는지 없는지 감지하는 SW, ... ] ---> 동작 제어 SW
- N:N 통신
    - ex. [ teacher ] x n ---> [ student ] x N



## 2. 토픽의 메시지를 변경해보자

메시지 타입을 String 대신에 Int32을 사용하여, Counter 값을 발행해보자. 

- teacher 노드 --- my_topic 전송 ---> student 노드
- "my topic": 1, 2, 3, 4



## 3 메시지 타입 변경에 따라 파이썬 코드 수정

{% highlight bash %}
$ vi teacher_int.py
$ vi student_int.py
{% endhighlight %}

teacher_int.py

{% highlight Python %}
#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32

# 'teacher' 노드를 새로 만든다.
rospy.init_node('teacher')

# 'my_topic' 이라는 이름으로 String 메시지를 발행하는 Publisher 생성
pub = rospy.Publisher('my_topic', Int32)

rate = rospy.Rate(2)    # 초당 2번 실행되도록 time slot을 생성
                        # 0.5초마다 실행되는 time slot
count = 1

while not rospy.is_shutdown():
    pub.publish(count)      # 0.2초 사용한다면
    rate.sleep()            # 0.3초 대기한 뒤에 재개
    count = count + 1
{% endhighlight %}

student.py

{% highlight Python %}
#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32

def callback(msg):
    print(msg.data)

# 'student' 노드를 새로 만든다.
rospy.init_node('student')

# String 메시지를 담고 있는 'my_topic' 이라는 토픽을 구독하고,
# 해당 토픽을 도착할 떄마다 callback 함수를 호출하는 Subscriber 생성
sub = rospy.Subscriber('my_topic', Int32, callback)

rospy.spin()    # 무한루프
{% endhighlight %}



## 4 파이썬 코드에서 여러개의 노드를 연결 할 경우

ROS 시스템에서 노드들은 각각 고유한 이름을 가져야만 하므로, 하나의 Python 코드에서 여러 개의 노드를 생성할 때 각 노드의 이름을 달리해야한다.
이때 단순하게 필요한 노드의 갯수만큼 이름만 다른 동일한 Python 파일을 만들어서 실행 할 수도 있지만, `init_node`함수의 `anonymous` 인자를 `True`로 설정하면 노드 이름 뒤에 임의의 숫자들이 삽입되므로, 하나의 Python 파일에서 중복되지 않는 이름을 가지는 노드들을 여러개 생성할 수 있다.

{% highlight Python %}
rospy.init_node('student', anonymouse=True)
{% endhighlight %}

{% highlight bash %}
# Publisher (anonymouse 파라미터를 사용하지 않을 경우)
$ rosrun msg_send teacher_int-1.py
$ rosrun msg_send teacher_int-2.py
$ rosrun msg_send teacher_int-3.py

# Subscriber (anonymouse 파라미터를 사용하는 경우)
$ rosrun msg_send student_init.py
$ rosrun msg_send student_init.py
$ rosrun msg_send student_init.py
{% endhighlight %}



## 5. Launch 파일에서 여러개의 노드를 연결할 경우

Launch 파일을 이용해서 여러개의 노드를 생성할 수 있는데, 이 때는 단순히 node 태그의 name 속성 값만 다르게 설정하면 된다. `roslaunch`명령어를 실행하게 되면, 소스코드에서 지정된 노드 이름 대신 node 태그의 name 값이 노드 이름이 된다.

{% highlight XML %}
<node pkg="msg_send" type="teacher_int.py" name="teacher1"/>
{% endhighlight %}

{% highlight XML %}
<launch>
    <node pkg="msg_send" type="teacher_int.py" name="teacher1"/>
    <node pkg="msg_send" type="teacher_int.py" name="teacher2"/>
    <node pkg="msg_send" type="teacher_int.py" name="teacher3"/>
    <node pkg="msg_send" type="student_int.py" name="student1" output="screen"/>
    <node pkg="msg_send" type="student_int.py" name="student2" output="screen"/>
    <node pkg="msg_send" type="student_int.py" name="student3" output="screen"/>
</launch>
{% endhighlight %}



### 5.1 1:N 통신 실행

Launch 파일만 바꿔서 실행: m_send_1n.launch

{% highlight XML %}
<launch>
    <node pkg="msg_send" type="teacher_int.py" name="teacher"/>
    <node pkg="msg_send" type="student_int.py" name="student1" output="screen"/>
    <node pkg="msg_send" type="student_int.py" name="student2" output="screen"/>
    <node pkg="msg_send" type="student_int.py" name="student3" output="screen"/>
</launch>
{% endhighlight %}

내가 만든 노드가 잘 동작하는지 확인

{% highlight bash %}
# 새로 생성되는 창에서 Node Graph의 "Nodes only" 대신
# "Nodes/Topics (all)"을 선택해서 다른 종류의 그래프를 볼수 있다.
$ rqt_graph
{% endhighlight %}



### 5.2 N:1 통신 실행

Launch 파일만 바꿔서 실행: m_send_n1.launch

{% highlight XML %}
<launch>
    <node pkg="msg_send" type="teacher_int.py" name="teacher1"/>
    <node pkg="msg_send" type="teacher_int.py" name="teacher2"/>
    <node pkg="msg_send" type="teacher_int.py" name="teacher3"/>
    <node pkg="msg_send" type="student_int.py" name="student" output="screen"/>
</launch>
{% endhighlight %}

내가 만든 노드가 잘 동작하는지 확인

{% highlight bash %}
# 새로 생성되는 창에서 Node Graph의 "Nodes only" 대신
# "Nodes/Topics (all)"을 선택해서 다른 종류의 그래프를 볼수 있다.
$ rqt_graph
{% endhighlight %}


### 5.3 N:N 통신 실행

Launch 파일만 바꿔서 실행: m_send_nn.launch

{% highlight XML %}
<launch>
    <node pkg="msg_send" type="teacher_int.py" name="teacher1"/>
    <node pkg="msg_send" type="teacher_int.py" name="teacher2"/>
    <node pkg="msg_send" type="teacher_int.py" name="teacher3"/>
    <node pkg="msg_send" type="student_int.py" name="student1" output="screen"/>
    <node pkg="msg_send" type="student_int.py" name="student2" output="screen"/>
    <node pkg="msg_send" type="student_int.py" name="student3" output="screen"/>
</launch>
{% endhighlight %}

내가 만든 노드가 잘 동작하는지 확인

{% highlight bash %}
# 새로 생성되는 창에서 Node Graph의 "Nodes only" 대신
# "Nodes/Topics (all)"을 선택해서 다른 종류의 그래프를 볼수 있다.
$ rqt_graph
{% endhighlight %}
