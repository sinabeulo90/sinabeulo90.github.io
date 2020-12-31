---
layout: post
title:  "[Day 3] 터틀심 8자 주행하기"
date:   2020-12-23 01:00:00 +0900
categories:
    - "K-Digital Training"
    - "ROS 노드 통신 프로그래밍"
---

## 실행 결과


![turtlesim 8](/assets/k-digital-training/ros_turtlesim_1.png)

![turtlesim 8](/assets/k-digital-training/ros_turtlesim_2.png)



## Publisher Python 코드


{% highlight Python %}
#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist

def pub_right(radius, pub, rate):
    for _ in range(4):
        msg = Twist()
        msg.linear.x = radius
        msg.angular.z = 1.5708
        pub.publish(msg)

        rate.sleep()

def pub_left(radius, pub, rate):
    for _ in range(4):
        msg = Twist()
        msg.linear.x = radius
        msg.angular.z = -1.5708
        pub.publish(msg)

        rate.sleep()

def main():
    rospy.init_node('turtle_8', anonymous=True)

    pub = rospy.Publisher('/turtle1/cmd_vel', Twist)

    while pub.get_num_connections() == 0:
        rospy.sleep(1)

    rate = rospy.Rate(0.5)
    radius = 0.5
    while not rospy.is_shutdown():
        pub_right(radius, pub, rate)
        pub_left(radius, pub, rate)
        radius += 0.5


if __name__ == "__main__":
    main()
{% endhighlight %}
