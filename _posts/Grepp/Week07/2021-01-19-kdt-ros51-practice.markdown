---
layout: post
title:  "[실습] Rosbag을 활용한 동영상 토픽 처리"
date:   2021-01-19 02:00:00 +0900
categories: "Grepp/KDT"
tag: ROS
---

## ROS 토픽의 저장

카메라의 ROS 토픽을 저장했다가 나중에 사용할 수 있다.

{% highlight bash %}
# 날아다니는 모든 토픽을 저장, Ctrl + C로 멈춤
$ rosbag record -a
# rosout, xycar_imu 2개 토픽을 저장
$ rosbag record rosout xycar_imu
# xycar_ultrasonic 토픽을 subset.bag 파일로 저장
$ rosbag record -O subset xycar_ultrasonic
# 저장된 subset.bag 파일의 각종 정보를 보여줌
$ rosbag info subset.bag
{% endhighlight %}



## ROS 토픽의 재생

저장한 ROS 토픽을 재생할 수 있다.

{% highlight bash %}
# 저장했던 토픽을 재생
$ rosbag play subset.bag
# 저장했던 토픽을 2배속으로 재생
$ rosbag play -r 2 subset.bag
{% endhighlight %}



## 저장된 ROS bag 파일에서 카메라 토픽만 꺼내기

{% highlight bash %}
# 1. 모든 토픽이 담겨있는 full_topic.bag 파일을 재생
$ rosbag play full_topic.bag
# 2. 카메라 토픽만 골라서 cam_topic.bag 파일로 저장
$ rosbag record -O cam_topic /usb_cam/image_raw/
# 3. 생성된 파일 확인
$ rosbag info cam_topic.bag
{% endhighlight %}



## 카메라 토픽을 모아서 동영상 파일 만들기

{% highlight bash %}
# 1. /usb_cam/image_raw 토픽을 모야서 track2.avi 동영상 파일을 만들 준비
$ rosrun image_view video_recorder image:="/usb_cam/image_raw" _filename:="track2.avi" _fps:=30
# 2. 모든 토픽이 담겨있는 full_topic.bag 파일을 재생
$ rosbag play full_topic.bag
{% endhighlight %}
