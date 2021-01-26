---
layout: post
title:  "[과제] 초음파 센서 활용: 응용"
date:   2021-01-13 04:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

## 초음파 센서 4개를 지원하는 ROS 패키지 제작

초음파 센서 4개 --- (4개의 선 X 4개 점퍼 연결) ---> Arduino Nano --- (USB 케이블) ---> PC(Linux & ROS)



## 노드와 토픽, 메시지 타입

- 노드: ultrasonic4
- 토픽: /ultra4
- 메시지 타입: Int32MultiArray (4개의 Int32 숫자 저장)



## 초음파 센서 4개를 아두이노 보드와 연결

- 각 센서의 Vcc, Trig, Echo, Gnd를 각 핀에 연결
- Vcc, Gnd를 묶어서 한꺼번에 아두이노 핀에 연결 가능



## 초음파 센서 4개를 관리하도록 펌웨어 코드 수정

- ultrasonic_4_fw.ino 파일 작성
- 초음파 센서 갯수 1개에서 4개로 수정
- 초음파 센서 4개의 거리 정보를 "300mm 121mm 186mm 67mm"와 같은 포맷으로 문자열 출력



## 디렉토리 구조와 파일 이름

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── ultrasonic
        ├── launch
        │   └── ultra4.launch
        └── src
            ├── ultra4_pub.py
            └── ultra4_sub.py
{% endhighlight %}



## ROS 노드 프로그램 소스 코드(ultra4_pub.py)

초음파 센서가 보낸 거리 정보를 토픽에 담아 발행한다.

{% highlight Python %}
#!/usr/bin/env python
# 필요한 모듈 import

# 아두이노가 연결된 포트 지정

def read_sensor():
    # 시리얼 데이터를 한번에 문자열로 받아옴
    # 문자열에서 숫자 4개를 추출하여, 리스트로 담음

if __name__ == "__main__":
from std_msgs.msg import Int32
    # ultra4_pub 노드 생성
    # ultra4 토픽 발행 준비

    while not rospy.is_shutdown():
        # 아두이노에서 정보를 받아와서
        # 토픽 안에 잘 채워 넣고 토픽 발행
    
    # 시리얼 포트를 닫고 정리
{% endhighlight %}



## 검증을 위한 Subscriber 노드 제작(ultra4_sub.py)

Publisher의 메시지를 받아 출력한다.

{% highlight Python %}
#!/usr/bin/env python
# 필요한 모듈 import

def callback(msg):
    # 토픽에서 데이터 꺼내서 화면에 출력

# ultra4_sub 노드 생성
# ultra4 토픽 구독 준비
# ultra4 토픽을 받으로 callback 함수가 호출되도록 함

# 무한루프
{% endhighlight %}



## Launch 파일(ultra4.launch)

{% highlight XML %}
<launch>
    <!-- 토픽의 발행자인 ultra4_pub.py 파일 실행 -->
    <!-- 토픽의 구독자인 ultra4_sub.py 파일 실행 -->
</launch>
{% endhighlight %}



## 실행과 결과 확인

{% highlight bash %}
$ roslaunch ultrasonic ultra4.launch
$ rqt_graph
$ rostopic echo ultra4
{% endhighlight %}
