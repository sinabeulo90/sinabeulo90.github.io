---
layout: post
title:  "[실습] Launch 파일 사용하기"
date:   2020-12-22 01:00:00 +0900
categories: "Grepp/KDT"
tags: ROS
---

# Launch 파일 사용하기



## 1. ROS의 명령어 - roslaunch

- 이전에 사용한 `rosrun` 명령어를 사용하면 노드들을 일일히 실행시켜야만 한다.
- `roslaunch` 명령어 : `*.launch` 파일 내용에 따라 여러 노드들을 한꺼번에 실행시킬 수 있다.

{% highlight bash %}
# roslaunch [패키지 이름] [실행시킬 launch 파일 이름]
# 이때 [실행 시킬 launch 파일 이름]은 반드시 패키지에 포함된 launch 파일이어야만 한다.
$ roslaunch my_pkg1 aaa.launch
{% endhighlight %}



## 2. *.launch 파일

- roslaunch 명령어를 이용하여 많은 노드를 동시에 실행시키기 위한 파일
- 실행시킬 노드들의 정보가 XML 형식으로 기록되어 있음
- XML: Tag 방식의 텍스트 파일

{% highlight XML%}
<launch>
    <!-- roslaunch 실행시 실행될 노드들 -->
    <node> ~~ </node>
    <node> ~~ </node>
    <node> ~~ </node>
    <node> ~~ </node>
</launch>
{% endhighlight %}



## 3. *.launch에서 사용하는 Tag: node

### 3.1 node 태그: 실행할 노드 정보를 입력할 때 사용되는 태그

{% highlight XML %}
<node pkg="패키지 명" type="노드가 포함된 소스파일 명" name="노드 이름"/>
<!-- 예시 -->
<node pkg="my_pkg1" type="pub.py" name="pub_node"/>
{% endhighlight %}

| 속성 | 설명 |
| :-: | :- |
| pkg | 실행시킬 노드의 패키지 이름을 입력하는 속성 <br/> 반드시 빌드된 패키지의 이름을 입력해야 함 |
| type | 노드의 소스코드가 담긴 Python 파일의 이름을 입력하는 속성 |
| name | 노드의 이름을 입력하는 속성 <br/> 소스코드에서 지정된 이름은 대신, launch 파일의 name 속성을 따름 |

Python 파일은 반드시 실행권한이 있어야 하며, 실행권한이 없을 시 다음 에러가 발생한다.

{% highlight bash %}
ERROR: cannot launch node of type [ 패키지 명 / 소스파일명 ]:
    can't locate node [ 소스파일 명 ] in package [ 패키지명 ]
{% endhighlight %}



{:start="2"}
### 3.2 include 태그: 다른 launch 파일을 불러오고 싶을 때 사용하는 태그

{% highlight XML %}
<include file="같이 실행할 *.launch 파일 경로"/>
<!-- 예시 -->
<include file="../cam/cam_test.launch" />
<include file="$(find usb_cam)/src/launch/aaa.launch"/>
{% endhighlight %}


| 속성 | 설명 |
| :-: | :- |
| file | 함께 실행시킬 `*.launch` 파일의 경로를 입력하는 속성 |



### 3.3 *.launch 파일의 위치

`catkin_create_pkg`로는 src만 생성되기 때문에, launch 폴더와 파일을 직접 만들어야 한다.

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── Package
        ├── launch
        │   └── *.launch
        └── src
            └── *.py
{% endhighlight %}



## 4.Launch 파일 생성

패키지 디렉토리 아래에 'launch'라는 디렉토리를 만들고, 그 안에 `.launch` 확장자의 파일을 만들어야 함

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── my_pkg1
        ├── CMakeLists.txt
        ├── launch              # launch 디렉토리 생성
        │   └── pub-sub.launch  # launch 파일 생성
        ├── package.xml
        └── src
            ├── pub.py
            └── sub.py
{% endhighlight %}



## 5. Launch 파일 작성

다수의 노드를 한꺼번에 실행시킬 수 있으며, 파라미터 값을 노드에 전달할 수 있다. (설정 값을 소스파일로 전달 가능)

{% highlight bash %}
$ roslaunch [package_name] [file.launch]`
{% endhighlight %}

Launch 파일 작성

{% highlight XML %}
<launch>
    <!-- roslaunch 명령어를 실행하면서 roscore도 자동으로 실행되므로, roscore를 따로 지정할 필요는 없다. -->
    <node pkg="turtlesim" type="turtlesim_node" name="turtlesim_node" />
    <node pkg="my_pkg1" type="pub.py" name="pub_node"/>
    <!-- 노드 sub_node는 결과를 화면으로 출력한다. -->
    <node pkg="my_pkg1" type="sub.py" name="sub_node" output="screen"/>
</launch>
{% endhighlight %}



## 6. Launch 파일 실행

Launch 파일 실행 방법

{% highlight bash %}
$ roslaunch my_pkg1 pub-sub.launch
{% endhighlight %}

`roslaunch` 명령을 실행하면서 내부적으로 `roscore`가 자동으로 실행되므로, 별도로 `roscore` 명령을 실행할 필요가 없다.  
---> 터미널 4개를 열어서 작업할 필요가 없으므로 매우 편리



## 7. 노드 동작 확인

노드가 잘 동작하고 있는지 확인

{% highlight bash %}
$ rqt_graph
{% endhighlight %}



## 8. 정리하면

{% highlight bash %}
$ cd ~/xycar_ws/src/my_pkg1
$ mkdir launch
$ cd ~/sycar_ws/src/my_pkg1/launch
$ vi pub-sub.launch
$ cm
$ roslaunch my_pkg1 pub-sub.launch
{% endhighlight %}
