---
layout: post
title:  "[실습] 카메라 기본 활용"
date:   2021-01-19 02:00:00 +0900
categories: "Grepp/KDT"
tag: ROS
---

## 카메라 활용 사례

- 카메라로 차선 등을 찾아 자율주행 구현
    - 차선을 찾아서 벗어나지 않고 주행하기
    - 사람을 찾고, 쫒아가며 주행하기
    - 앞 차의 뒤꽁무니를 찾고, 따라가도록 주행(군집 주행)
- 카메라를 이용한 주변상황 인지
    - 전방 이동 물체 인식: 차량, 사람, 자전거 등 추돌 방지에서 사용
    - 전방 고정 물체 인식: 교통 표지판, 신호등, 정지선, 횡단보도, 언덕 등
- 카메라 영상으로 자신의 위치 파악(Localization)
    - 앞에 펼쳐진 전경, 지형 지물 등으로 보고 파악
    - 지도 데이터와 비교하여 현재 차량의 위치를 유추



## 패키지 생성

- ROS workspace의 src 폴더에서 my_cam 패키지를 만들고, /launch 서브 폴더를 만든다.


{% highlight bash %}
$ catkin_create_pkg my_cam std_msgs rospy
{% endhighlight %}


{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── my_cam
        ├── launch
        │   └── edge_cam.launch
        └── src
            └── edge_cam.py
{% endhighlight %}


## Launch 파일 작성(edge_cam.launch)

{% highlight XML %}
<launch>
	<node name="usb_cam" pkg="usb_cam" type="usb_cam_node" output="screen">
		<param name="video_device" value="/dev/v4l/by-id/..." />
		<!-- 노출도(Exposure) -->
		<param name="autoexposure" value="false" />
		<param name="exposure" value="150" />
		<!-- 해상도(Resolution): 640x480 -->
		<param name="image_width" value="640" />
		<param name="image_height" value="480" />

		<param name="pixel_format" value="yuyv" />
		<param name="camera_frame_id" value="usb_cam" />
		<param name="io_method" value="mmap" />
	</node>
    <node name="my_cam" pkg="my_cam" type="edge_cam.py" output="screen" />
</launch>
{% endhighlight %}



## 카메라 영상 출력 프로그램 예지(edge_cam.py)

카메라 영상을 OpenCV로 처리하여 그 결과를 화면에 출력

{% highlight Python %}
#!/usr/bin/env python
#-*- coding: utf-8 -*-
import cv2          # OpenCV 사용 준비
import rospy
import numpy as np  # numpy 사용 준비
from sensor_msgs.msg import Image
from cv_bridge import CvBridge      # ROS에서 OpenCV를 편하게 사용하기 위한 CvBridge 사용 준비

bridge = CvBridge()
cv_image = np.empty(shape=[0])

# Image 토픽을 처리하는 콜백함수 정의
def image_callback(data):
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

rospy.init_node("cam_tune", anonymous=True)

# Image 토픽이 도착하면, 콜백함수가 호출되도록 세팅
rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)

while not rospy.is_shutdown():
    # 640x480 이미지가 한 장 모일 때까지 대기
    if cv_image.size != (840 * 480 * 3):
        continue

    # 원본 이미지를 Grayscale로 변경
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    # 부드럽게 변경
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # 이미지의 외곽선만 표시하게 변경
    edge_img = cv2.Canny(np.uint8(blur_gray), 60, 70)

    # 원본 포함 4개 이미지를 표시
    cv2.imshow("original", cv_image)
    cv2.imshow("gray", gray)
    cv2.imshow("gaussian blur", blur_gray)
    cv2.imshow("edge", edge_img)
    cv2.waitKey(1)
{% endhighlight %}



## 결과 확인

{% highlight bash %}
$ roslaunch my_cam edge_cam.launch
$ rqt_graph
{% endhighlight %}
