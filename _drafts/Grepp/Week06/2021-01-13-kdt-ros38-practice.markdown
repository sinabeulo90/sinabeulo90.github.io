---
layout: post
title:  "[실습] 초음파 센서 활용: 아두이노"
date:   2021-01-13 02:00:00 +0900
categories: "Grepp/KDT"
tags:
    - ROS
    - Arduino
---

## 초음파 센서 ROS 패키지

초음파 센서를 제어하여 물체까지의 거리를 알아내고, 그 정보를 ROS 토픽으로 만들어 노드들에게 보내준다.

초음파 센서 ---> 아두이노 ---> 리눅스 ROS

1. 초음파 센서
    - 물체로 [초음파(Ultrasonic Wave)]({% post_url Grepp/Week06/2021-01-11-kdt-ros28 %})를 쏘고, 반사된 초음파 신호를 감지
    - 처음 초음파를 쏜 시점과 반사파를 수신한 시점을 표시한 펄스(Pulse) 신호를 아두이노에게 보냄
2. 아두이노
    - 아두이노 펌웨어 제작 필요
    - 초음파 센서가 보내준 펄스 신호를 받아 분석
    - 초음파를 쏜 시점과 반사파를 받은 시점의 시간 차이를 이용하여, 물체까지의 거리를 계산하고 이를 리눅스(ROS)에 알려줌
3. 리눅스 ROS
    - ROS 노드 SW 제작 필요
    - 아두이노가 보내준 물체까지의 거리 정보를 사용하기 좋은 형태로 적절히 가공한 후에 ROS 토픽에 담아 그게 필요한 노드들에게 Publish 함



## 초음파 센서와 아두이노의 연결

| 초음파 센서 | 아두이노 |
|:-:|:-:|
| Vcc | 5V |
| Trig | D2 |
| Echo | D3 |
| Gnd | GND |



## 아두이노 핀 배치도

- 초음파 센서가 아두이노의 어떤 핀에 연결되었는지 확인
- 펌웨어에서 세팅 작업에 필요

![Arduino NANO Pinout](/assets/grepp/pinout-nano_latest.png)



## [Arduino IDE](https://www.arduino.cc/en/software) 설치

- 아두이노 펌웨어 프로그래밍 도구
- 아두이노 코드 작성하고, 제작된 펌웨어를 아두이노에 적어서 넣을 때 사용
- 다운로드한 파일의 압축을 풀고 안에 있는 `install.sh`을 실행시키면 설치 완료

{% highlight bash %}
$ sudo ./install.sh
{% endhighlight %}



## 펌웨어 프로그래밍

- Arduino IDE 실행하고, `ultrasonic_1_fw.ino` 파일 작성

{% highlight bash %}
$ sudo arduino
{% endhighlight %}


- `ultrasonic_1_fw.ino`: 초음파 센서가 보내는 신호로부터 거리 정보 추출 코드

{% highlight ino %}
/*
HC-SR04 초음파 센서 아두이노 펌웨어
*/

#define trig 2  // 트리거 핀 선언
#define echo 3  // 에코 핀 선언

void setup()
{
    Serial.begin(9600);     // 9600bps 통신 속도로 시리얼 통신 시작
    pinMode(trig, OUTPUT);  // Trig 핀을 출력으로 선언
    pinMode(echo, INPUT);   // Echo 핀을 입력으로 선언
}

void loop()
{
    long duration, distance;    // 거리 측정을 위한 변수 선언
    
    // Trig 핀으로 10us 동안 펄스 출력
    digitalWrite(trig, LOW);    // Trig 핀 Low
    delayMicroseconds(2);       // 2us 딜레이
    digitalWrite(trig, HIGH);   // Trig 핀 High
    delayMicroseconds(10);      // 10us 딜레이
    digitalWrite(trig, LOW);    // Trig 핀 Low

    // pulseIn(): 핀에서 펄스 신호를 읽어서 마이크로초 단위로 반환
    duration = pulseIn(echo, HIGH);
    distance = duration * 170 / 1000;   // 왕복 시간이므로 340/2=170 곱하는 것으로 계산
    Serial.print("Distance(mm): ");
    Serial.println(distance);           // 거리 정보를 시리얼 모니터에 출력
    delay(100);
}
{% endhighlight %}



## PC에서 아두이노 연결 확인

{% highlight bash %}
$ lsusb

...
Bus XXX Device XXX: ... HL-340 USB-Serial ...
...
{% endhighlight %}



## USB 케이블 연결

- 아두이노와 PC 또는 Nvidia 보드는 물리적으로 USB 케이블로 연결되어 있지만, 내부적으로는 Serial 통신으로 이루어져 있다. (Serial over USB)
- SW 입장에서는 Serial 통신을 맞춰야 한다.



## 리눅스에서 아두이노 연결 확인

Tools 메뉴에서 Board, Processor, Port 확인

- Board: Arduino Nano
- Processor: ATmega328P
- Port: /dev/ttyUSB0 또는 /dev/ttyACM0



## 컴파일 및 업로드

- 컴파일: 소스코드에 에러가 있는지 확인하고 컴파일
- 업로드: 만들어진 펌웨어를 아두이노 안에 업로드



### 결과 확인

Tools 메뉴에서 Serial Monitor를 통해 아두이노의 출력값 확인
