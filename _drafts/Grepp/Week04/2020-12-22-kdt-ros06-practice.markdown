---
layout: post
title:  "[실습] Launch에서 Tag 활용"
date:   2020-12-22 02:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

# Launch에서 Tag 활용



## 1. *.launch 파일 사례

ex. USB 카메라 구동과 파라미터 세팅을 위한 launch 파일

- 패키지 이름: `usb_cam`
- 파라미터 : `autoexposure`, `exposure`, `image_width`, `image_height`, `camera_frame_id`

{% highlight XML %}
<launch>
    <!-- type의 값에 확장자가 없는 것으로 봐서 C/C++로 작성된 프로그램이라고 유추 가능 -->
    <node name="usb_cam" pkg="usb_cam" type="cam_node" output="screen">
        <param name="autoexposure" value="false"/>
        <param name="exposure" value="150"/>
        <param name="image_width" value="640"/>
        <param name="image_height" value="480"/>
        <param name="camera_frame_id" value="usb_cam"/>
    </node>
</launch>
{% endhighlight %}



## 2. *.launch에서 사용하는 Tag: param

param: ROS 파라미터 서버에 변수를 등록하고, 그 변수에 값을 설정하기 위한 태그

{% highlight XML %}
<param name="변수의 이름" type="변수의 타입" value="변수 값"/>
{% endhighlight %}

| 속성 | 설명 |
| :-: | :- |
| name | 등록할 변수의 이름 |
| type | 등록할 변수의 타입 <br/> 사용 할 수 있는 타입의 종류는 str, int, double, bool, yaml |
| ~~type~~ | type을 생략할 경우, value의 값에 따라 자동으로 지정 |
| value | 등록할 변수의 값 |

ROS 파라미터 서버에 등록된 변수들은 노드의 소스 코드에서 불러와 사용할 수 있다.

{% highlight XML %}
<node pkg="패키지 명" type="노드가 포함된 소스파일 명" name="노드" output="screen">
    <param name="age" type="int" value="11"/>
</node>
{% endhighlight %}

{% highlight Python %}
import rospy
rospy.init_node('노드')
print(rospy.get_param('~age'))  # private parameter는 앞에 '~'를 붙인다.
{% endhighlight %}



## 3. Launch 파일에서 파라미터 전달 실습

Launch 파일을 새로 만들고, `.launch` 파일의 파라미터 값을 읽어서 거북이의 회전 반경이 변경을 동작하게끔 만들어보자.

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── my_pkg1
        ├── launch                      # launch 디렉토리 생성
        │   └── pub-sub-param.launch    # launch 파일 생성
        ├── package.xml
        └── src
{% endhighlight %}

{% highlight bash %}
$ vi pub-sub-param.launch
{% endhighlight %}

{% highlight XML %}
<launch>
    <node pkg="turtlesim" type="turtlesim_node" name="turtlesim_node"/>
    <!--
        패키지 이름: my_pkg1
        타입(소스코드 파일): pub_param.py
        노드 이름: node_param
        파라미터 이름: circle_size
        파라미터 값: 2
    -->
    <node pkg="my_pkg1" type="pub_param.py" name="node_param">
        <param name="circle_size" value="2"/>
    </node>
    <node pkg="my_pkg1" type="sub.py" name="sub_node" output="screen"/>
</launch>
{% endhighlight %}

`pub.py` 파일을 복사+수정해서 `pub_param.py` 파일을 새로 만든다.

{% highlight bash %}
$ cp pub.py pub_param.py
$ vi pub_param.py
{% endhighlight %}

{% highlight Python %}
def move():
    rospy.init_node('my_node', anonymous=True)
    pub = rospy.Publisher('/tutle1/cmd_vel1', Twist, queue_size=10)

    msg = Twist()

    # msg.linear.x = 2.0
    # .launch 파일에서 "circle_size" param 값을 읽어들여 사용한다.
    linear_X = rospy.get_param('~circle_size')
    msg.linear.x = linear_X

    msg.linear.y = 0.0
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 1.0
{% endhighlight %}



## 4. Launch 파일 실행

`pub-sub-param.launch` 파일에서 values 값을 2 또는 4로 바꿔보자

{% highlight XML %}
        <param name="circle_size" value="2"/>
{% endhighlight %}

실행하기

{% highlight bash %}
$ roslaunch my_pkg1 pub-sub-param.launch
{% endhighlight %}

노드 동작 확인

{% highlight bash %}
# "/node_param" 이름의 노드에서 토픽이 발행됨을 볼 수 있다.
$ rqt_graph
{% endhighlight %}
