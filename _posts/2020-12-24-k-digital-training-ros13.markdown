---
layout: post
title:  "[Day 4.3] 원격 통신 프로그래밍"
date:   2020-12-24 03:00:00 +0900
categories:
    - "K-Digital Training"
    - "ROS 노드 통신 프로그래밍"
---

## 전체 구성

- 노드1: remote_student
- 노드2: remote_teacher


- 노드1 --- 토픽 전송 msg_to_xycar(my_msg) ---> 노드2
- 노드1 <--- 토픽 전송 msg_from_xycar(my_msg) --- 노드2
- Publisher ---> Subscriber
- Subscriber ---> Publisher


- IP 주소: 192.168.0.AAA --- IP 주소: 192.168.0.BBB


## 기존에 만든 msg_send 패키지에서 작업

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── msg_send
        ├── launch
        │   └── m_send.launch
        ├── msg
        │   └── my_msg.msg
        └── src
            ├── teacher.py
            └── student.py
{% endhighlight %}


## 커스텀 메시지 전송

my_msg 메시지 사용

{% highlight Python %}
from msg_send.msg import my_msg
{% endhighlight %}


## 시나리오

학생 1, 학생 2, 학생 3: 발행자 & 구독자 노드 remote_student.py --> 노드의 명칭이 모두 달라져야 한다.

Xycar: 마스터 core, 발행자 & 구독자 노드, remote_teacher.py


### 원격 통신 실습 #1

학생들이 my_msg 타입의 데이터를 가진 msg_to_xycar 토픽을 1번만 보내고, 그걸 강사가 받아서 출력

{% highlight Python %}
msg.first_name = "sungmin"
msg.last_name = "Her"
msg.id_number = 20209876
msg.phone_number = "010-8950-1010"
{% endhighlight %}

강사 출력 코드
{% highlight Python %}
print("1. Name:", ***)
print("2. ID:", ***)
print("3. Phone Number :", ***)
{% endhighlight %}

- 앞서 제작한 msg_send 패키지를 사용

강사
{% highlight bash %}
$ roscore
$ rosrun msg_send msg_receiver.py
{% endhighlight %}

학생들
{% highlight bash %}
$ rosrun msg_send msg_sender.py
{% endhighlight %}

결과: 학생들이 보낸 my_msg 타입의 메시지 내용을 화면에 출력함


### 원격 통신 실습 #2

학생들이 my_msg 타입의 데이터를 보내고 그걸 강사가 받으면 이름을 String 문자열에 담아 회신함. 학생들은 String 내용을 확인하고 출력함.

학생1 --- msg_to_xycar ---> 강사

{% highlight Python %}
msg.first_name = "sungmin"
msg.last_name = "Her"
msg.id_number = 20209876
msg.phone_number = "010-8950-1010"
{% endhighlight %}

강사 --- msg_from_xycar ---> 학생1

- "Good afternoon, Sungmin"
- 앞서 제작한 msg_send 패키지를 사용

강사
{% highlight bash %}
$ roscore
$ rosrun msg_send remote_teacher.py
{% endhighlight %}

학생들
{% highlight bash %}
$ rosrun msg_send remote_student.py
{% endhighlight %}

결과: 학생들이 보낸 토픽이 도착하면 강사는 이름을 String 문자열에 담아 학생에게 회신함.
