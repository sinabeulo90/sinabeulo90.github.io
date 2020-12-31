---
layout: post
title:  "[Day 4.4] 받은 토픽을 가공해서 보내기 실습"
date:   2020-12-24 04:00:00 +0900
categories:
    - "K-Digital Training"
    - "ROS 노드 통신 프로그래밍"
---

## 노드간 통신 개요

전체 구성

- 노드1: teacher <--- msg_to_xycar --- 노드2: student
- Subscriber <--- Publisher

- 노드1: teacher --- msg_from_xycar ---> 노드2: student
- Publisher ---> Subscriber


## 패키지 생성

디렉토리 구조

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── msg_send
        ├── launch
        │   └── remote.launch
        └── src
            ├── remote_teacher.py
            └── remote_student.py
{% endhighlight %}



## 파이썬 프로그램 코딩

토픽을 발행하고 구독하는 예제 파이썬 코드

- 토픽 이름: msg_to_xycar, msg_from_xycar
- msg_to_xycar: my_msg 타입 사용
- msg_from_xycar: String 타입 사용
- remote_student.py: 자신의 인적사항을 전송한다. msg_to_xycar 라는 토픽으로 전송하고, teacher에서 온 msg_from_xycar를 화면에 출력
- remote_teacher.py: msg_to_xycar 토픽에서 학생의 이름만 뽑아내서 "good morning 학생이름 현재시간" 이라는 string으로 msg_from_xycar 토픽을 생성해서 전송



## 원격통식 Overview

Student가 my_msg 타입의 데이터를 보내고, 그걸 teacher가 받으면 [인사말, 이름, 현재시간]을 String 문자열에 담아 회신함. 학생들은 String 내용을 확인하고 출력함

msg_to_xycar:

{% highlight Python %}
msg.first_name = "Gil-Dong"
msg.last_name = "Hong"
msg.age = 15
msg.score = 100
msg.id_number = 091234
msg.phone_number = "010-7896-1234"
{% endhighlight %}

msg_from_xycar

{% highlight bash %}
"Good morning, Hong Gil-Dong 2020-12-04 16:30:14"
{% endhighlight %}


## Launch 파일 만들고 실행하기

Launch 파일 작성

{% highlight bash %}
$ vi remote.launch
{% endhighlight %}

{% highlight XML %}
<launch>
    <node pkg="msg_send" type="remote_teacher.py" name="teacher" output="screen"/>
    <node pkg="msg_send" type="remote_student.py" name="student" output="screen"/>
</launch>
{% endhighlight %}

{% highlight bash %}
# 빌드한다.
$ cm

# 파이썬 파일에는 실행권한을 부여해야 한다.
$ chmod +x remote*.py

# Launch 파일 실행 방법
$ roslaunch msg_send remote.launch
{% endhighlight %}


## 실행 결과

- 앞서 제작한 msg_send 패키지를 사용
- 결과: 학생들이 보낸 토픽이 도착하면 강사는 이름을 String 문자열에 담아 학생에게 회신함.

{% highlight bash %}
$ rqt_graph
$ rostopic echo /msg_to_teacher
$ rostopic echo /smg_from_teacher
{% endhighlight %}
