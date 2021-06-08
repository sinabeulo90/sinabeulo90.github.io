---
layout: post
title:  "[과제] AR Tag를 이용하여 주차하기"
date:   2021-02-16 02:00:00 +0900
category: "Grepp/KDT"
tag: "AR Tag"
---

## 과제: AR Tag를 보고 주차 영역에 똑바로 주차하기

### Xycar Parking Simulator 환경에서 수행
- 자동차는 가상 AR Tag를 보고 주행하여 주차구역(사각형) 안에 주차해야 한다.
- 차가 주차구역 안에 완전히 들어갔을 때, 주차구역 선이 녹색으로 바뀐다.

### Xycar Parking Simulator로부터 AR Tag 정보 입수
- 차와 움직임을 같이하는 좌우 빨간선 한 쌍은 시뮬레이터 속 가상의 카메라 범위를 의미
- 가상의 카메라 시야각 안에 AR Tag(녹색 표시)가 존재하면 /ar_pose_marker 토픽에 AR Tag의 위치와 자세 정보가 담겨 publish 된다.


### AR Tag의 자세 정보를 이용
- AR Tag가 카메라 시야각 안에 들어오면 (영상처리를 통해 추출된) 정보가 발행된다.
- 추출된 자세 정보는 /ar_pose_marker 토픽에 담겨 발행된다.
- 이 정보를 이용해서 주차 영역으로 진입하면 된다.


### 작업 디렉토리
- 기존에 만든 ar_viewer 패키지에서 작업
- 런치파일, 파이썬 코드 작성하기

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── ar_viewer
        ├── launch
        │   └── ar_parking.launch
        └── src
            └── ar_parking.py
{% endhighlight %}


### 시뮬레이터에서 ROS 노드와 토픽 구성도
- xycar_sim_parking 노드 --- /ar_pose_marker 토픽 ---> ar_parking.py
- xycar_sim_parking 노드 <--- /xycar_motor_msg 토픽 --- ar_parking.py


### 런치 파일 만들기 - ar_parking.launch
- 시뮬레이터와 주차주행 관련 내용을 담는다.
- 시뮬레이터
    - 패키지 이름: xycar_sim_parking
    - 파이썬 파일: main.py
- 주차 주행 코드
    - 패키지 이름: ar_viewer
    - 파이썬 파일: ar_parking.py

{% highlight XML %}
<launch>
    <node name="xycar_sim_parking" pkg="xycar_sim_parking" type="main.py" output="screen" />
    <node name="ar_parking" pkg="ar_viewer" type="ar_parking.py" output="screen" />
</launch>
{% endhighlight %}


### 시뮬레이터가 보내는 토픽 - /ar_pose_marker
- 현실에서 ar_track_alvar 패키지가 보내는 토픽과 동일
    - 토픽 이름: /ar_pose_marker
    - 카메라 화면에서 AR Tag가 어떤 위치에 있는지, 어떤 자세를 하고 있는지에 대한 정보를 /ar_pose_marker 토픽을 담아 전송한다.
    - 실제와 달리 시뮬레이터가 보내는 정보의 거리 단위는 m(미터)가 아닌 pixel(픽셀)이다.
- 3차원 공간에서의 좌표값: x, y, z
    - pose.pose.position (x, y, z)
    - 픽셀 단위의 거리 정보
- 자세 정보값: roll, pitch, yaw
    - pose.pose.orientation (x, y, z, w)
    - 하늘에서 보이는 자동차의 진행 각도


### 자동차 자세 잡기
- AR Tag 자세를 보고 자동차의 자세를 잡아야 한다.
- AR Tag까지의 거리를 따져서 핸들링을 해야 한다.


### 주차를 위한 자동차 제어
- 차량의 왼쪽과 오른쪽 구분
    - 왼쪽에 주차영역이 있는 경우
    - 오른쪽에 주차영역이 있는 경우
- x 좌표 값으로 구분

{% highlight Python %}
if (arData["DX"] >= 0):     # AR Tag가 오른쪽에 있을 때
    ...
    angle = ...             # 조향각 angle 값을 결정
elif (arData["DX"] < 0):    # AR Tag가 왼쪽에 있을 때
    ...
    angle = ...             # 조향각 angle 값을 결정

speed = ...                 # AR Tag까지의 거리가 가까워지면, 속도 늦추기

# 자동차 구동 토픽 발행
xycar_msg.data = [angle, speed]
motor_pub.publish(xycar_msg)
{% endhighlight %}

- 똑같은 거리, 하지만 다른 자세(각도, yaw)

{% highlight Python %}
# AR Tag가 오른쪽에 있을 때
if (yaw 값):
    ...         # x, y 값에 따른 angle 설정
elif (yaw 값):
    ...         # x, y 값에 따른 angle 설정
elif (yaw 값):
    ...         # x, y 값에 따른 angle 설정
{% endhighlight %}

- (이번엔 AR Tag가 왼쪽에 있을 때) 똑같은 거리, 하지만 다른 자세(각도, yaw)

{% highlight Python %}
# AR Tag가 왼쪽에 있을 때
if (yaw 값):
    ...         # x, y 값에 따른 angle 설정
elif (yaw 값):
    ...         # x, y 값에 따른 angle 설정
elif (yaw 값):
    ...         # x, y 값에 따른 angle 설정
{% endhighlight %}

- 비스듬히 주차한 경우, 후진해서 다시 주차 시도를 해야 한다.
- 자동차를 옆으로 옮기는 방법


### 파이썬 파일 - ar_parking.py

{% highlight Python %}
#!/usr/bin/env python
import rospy, math
import cv2, time
import numpy as np
from ar_track_alvar_msgs.msg import AlvarMarkers    # AR Tag 거리/자세 정보 토픽의 메시지
from tf.transformations import euler_from_quaternion    # 쿼터니언 값을 오일러 값으로 변환
from std_msgs.msg import Int32MultiArray            # 자동차 구동 제어 토픽의 메시지

xycar_msg = Int32MultiArray()

# AR Tag의 거리정보와 자세 정보를 저장할 공간 마련
arData = {
    "DX": 0.0, "DY": 0.0, "DZ": 0.0,
    "AX": 0.0, "AY": 0.0, "AZ": 0.0, "AW": 0.0}
roll, pitch, yaw = 0, 0, 0

# /ar_pose_marker 토픽을 받을때 마다 호출되는 콜백함수 정의
def callback(msg):
    global arData
    for i in msg.markers:
        # AR Tag의 위치 정보(3차원 벡터값, x, y, z)
        arData["DX"] = i.pose.pose.position.x
        arData["DY"] = i.pose.pose.position.y
        arData["DZ"] = i.pose.pose.position.z
        # AR Tag의 자세 정보(쿼터니언 값, roll, pitch, yaw)
        arData["AX"] = i.pose.pose.orientation.x
        arData["AY"] = i.pose.pose.orientation.y
        arData["AZ"] = i.pose.pose.orientation.z
        arData["AW"] = i.pose.pose.orientation.w

rospy.init_node("ar_parking")
rospy.Subscriber("ar_pose_marker", AlvarMarkers, callback)  # 토픽의 구독 준비
motor_pub = rospy.Publisher("xycar_motor_msg", Int32MultiArray, queue_size=1)   # xycar_motor_msg 토픽의 발행 준비

while not rospy.is_shutdown():
    # AR Tag의 자세 정보를 담은 쿼터니언 값을 오일러 값으로 변환
    (roll, pitch, yaw) = euler_from_quaternion(
        (arData["AX"], arData["AY"], arData["AZ"], arData["AW"]))
    
    # 각도 표현법을 라디안에서 60분법(도)으로 변경
    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)

    print("==========================")
    # Roll/Pitch/Yaw 값 출력
    print(" roll : ", round(roll, 1))
    print("pitch : ", round(pitch, 1))
    print("  yaw : ", round(yaw, 1))
    # 3차원 좌표계 x, y, z 값으로 출력
    print("x : ", arData["DX"])
    print("y : ", arData["DY"])
    print("z : ", arData["DZ"])

    # 높이x폭 = 100x500 크기의 이미지 준비
    img = np.zeros((100, 500, 3))

    # 빨간색으로 선 긋기
    img = cv2.line(img, (25, 65), (475, 65), (0, 0, 255), 2)
    img = cv2.line(img, (25, 40), (25, 90), (0, 0, 255), 3)
    img = cv2.line(img, (250, 40), (250, 90), (0, 0, 255), 3)
    img = cv2.line(img, (475, 40), (475, 90), (0, 0, 255), 3)

    # 녹색 원을 그릴 위치 계산하기
    # DX 값이 "0"일 때, 화면의 중앙 위치에 오도록 세팅하기 위해
    # DX + 250 = point 값으로 설정
    point = int(arData["DX"]) + 250

    # 제일 작은 값을 25, 제일 큰 값을 475로 제한
    if point > 475:
        point = 475
    elif point < 475:
        point = 25

    # point 위치에 녹색 원 그리기
    img = cv2.circle(img, (point, 65), 15, (0, 255, 0), -1)

    # x, y 축 방향의 좌표값을 가지고, AR Tag까지의 거리를 계산
    # DX, DY 값ㅇ르 가지고 거리 계산(직각삼각형)
    distance = math.sqrt(pow(arData["DX"], 2) + pow(arData["DY"], 2))
    
    # 거리값을 우측 상단에 표시
    cv2.putText(img, str(int(distance)), (350, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    # DX, DY, Yaw 값을 문자열로 만들고, 좌측 상단에 표시
    dx_dy_yaw = "DX: {} DY: {} Yaw: {}".format(arData["DX"], arData["DY"], round(yaw, 1))
    cv2.putText(img, dx_dy_yaw, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))

    # 윈도우에 타이틀을 붙이고 그림 그리기
    cv2.imshow("AR Tag Position", img)
    cv2.waitKey(1)

    # 주차영역과의 거리와 각도를 잘 따져서 핸들 조향각과 속도를 조정해야 함
    # 필요에 따라 후진도 가능
    angle = ...
    speed = ...

    # 모터 제어 토픽을 발행: 시뮬레이터의 차량을 이동시킴
    xycar_msg.data = [angle, speed]
    motor_pub.publish(xycar_msg)

cv2.destroyAllWindows()
{% endhighlight %}


### 실행

{% highlight bash %}
$ roslaunch ar_viewer ar_parking.launch
{% endhighlight %}
