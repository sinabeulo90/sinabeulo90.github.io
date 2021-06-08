---
layout: post
title:  "[과제] AR Tag 위치 파악과 표시"
date:   2021-02-16 01:00:00 +0900
category: "Grepp/KDT"
tag: "AR Tag"
---

## 과제: AR Tag 시뮬레이터에서 태그위 위치를 파악하고 이를 화면에 표시

### AR Tag의 위치를 파악해서 시각화 하는 뷰어 제작
- AR Tag가 차량의 좌우 어디쯤에 있는지, 얼마만큼 떨어져 있는지를 파악하기 위한 목적
- OpenCV를 이용하여 시뮬레이터에 있는 자동차의 관점에서 AR 위치와 거리를 표시


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
        │   └── ar_drive.launch
        └── src
            └── ar_drive.py
{% endhighlight %}


### 시뮬레이터에서 ROS 노드와 토픽 구성도
- xycar_sim_parking 노드 --- /ar_pose_marker 토픽 ---> ar_dirve.py
- xycar_sim_parking 노드 <--- /xycar_motor_msg 토픽 --- ar_dirve.py


### 런치 파일 만들기 - ar_drive.launch
- 시뮬레이터와 AR Tag 위치 정보 시각화 관련 내용을 담는다.
- 시뮬레이터
    - 패키지 이름: xycar_sim_parking
    - 파이썬 파일: main.py
- AR Tag 위치 정보 시각화 코드
    - 패키지 이름: ar_viewer
    - 파이썬 파일: ar_drive.py

{% highlight XML %}
<launch>
    <node name="xycar_sim_parking" pkg="xycar_sim_parking" type="main.py" output="screen" />
    <node name="ar_drive" pkg="ar_viewer" type="ar_drive.py" output="screen" />
</launch>
{% endhighlight %}


### 파이썬 파일 - ar_drive.py

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

rospy.init_node("ar_drive")
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
    cv2.line(img, ...)
    cv2.line(img, ...)
    cv2.line(img, ...)
    cv2.line(img, ...)

    # 녹색 원을 그릴 위치 계산하기
    point = ...

    # point 위치에 녹색 원 그리기
    img = cv2.circle(img, (point, 65), ...)

    # x, y 축 방향의 좌표값을 가지고, AR Tag까지의 거리를 계산
    distance = ...
    
    # 거리값을 우측 상단에 표시
    cv2.putText(img, str(int(distance)), ...)

    # DX, DY, Yaw 값을 문자열로 만들고, 좌측 상단에 표시
    dx_dy_yaw = ...
    cv2.putText(img, dx_dy_yaw, ...)

    cv2.imshow("AR Tag Position", img)
    cv2.waitKey(10)

    # 시계 방향으로 천천히 회전 주행
    angle = 50
    speed = 5

    # 모터 제어 토픽을 발행: 시뮬레이터의 차량을 이동시킴
    xycar_msg.data = [angle, speed]
    motor_pub.publish(xycar_msg)

cv2.destroyAllWindows()
{% endhighlight %}


### 실행

{% highlight bash %}
$ roslaunch ar_viewer ar_drive.launch
{% endhighlight %}
