---
layout: post
title:  "ROS 명령어"
date:   2020-12-21 02:00:00 +0900
categories:
    - "K-Digital Training"
    - "자율주행 데브코스"
---

## 2. ROS 명령어

### 2.1 ROS 기본 용어

### 2.2 ROS 기본 명령어
- ROS 셸 명령어

| 명령어 | 설명 |
| :-: | :- |
| roscd | 지정한 ros 패키지 폴더로 이동 |
| rosls | ros 패키지 파일 목록 확인 |
| rosed | ros 패키지 파일 편집 |
| roscp | ros 패키지 파일 복사 |

- ROS 실행 명령어

| 명령어 | 설명 |
| :-: | :- |
| **roscore** | master + rosout + parameter server 실행, 마스터 실행 |
| **rosrun** | 패키지 노드 실행 |
| **roslaunch** | 패키지 노드를 여러 개 실행 |
| rosclean | ros 로그파일 검사 및 삭제 |

- ROS 정보 명령어

| 명령어 | 설명 |
| :-: | :- |
| **rostopic** | 토픽 정보 확인 |
| **rosnode** | 노드 정보 확인 |
| rosparam | 파라미터 정보 확인, 수정 |
| **rosbag** | 메시지 기록, 재생 |
| rosmsg | 메시지 정보 확인 |
| rosversion | 패키지 및 배포 버전정보 확인 |
| roswtf | ros 시스템 검사 |

- ROS catkin 명령어: 새로운 패키지를 만들때 쓰임

| 명령어 | 설명 |
| :-: | :- |
| **catkin_create_pkg** | catkin 빌드 시스템으로 패키지 생성 |
| **catkin_make** | catkin 빌드 시스템으로 빌드 |
| catkin_eclipse | 패키지를 eclipse에서 사용할 수 있게 변경 |
| catkin_prepare_release | changelog 정리 및 버전 태깅 |
| catkin_init_workspace | catkin 빌드 시스템의 작업 폴더 초기화 |
| catkin_find | 검색 |

- ROS package 명령어

| 명령어 | 설명 |
| :-: | :- |
| rospack | 패키지와 관련된 정보 보기 |
| rosinstall | 추가 패키지 설치 |
| rosdep | 해당 패키지의 의존성 파일 설치 |
| roslocate | 패키지 정보 관련 명령어 |
| rosmake | 패키지 빌드(구 시스템에서 사용) |
| roscreate-pkg | 패키지 자동 생성(구 시스템에서 사용) |

### 2.3 ROS 주요 명령어
**roscore**: ROS 기본 시스템이 구동되기 위해 필요한 프로그램들을 실행(마스터 실행)
{% highlight bash %}
$ roscore
{% endhighlight %}

**rosrun** [package name] [node_name] : 패키지에 있는 노드를 선택 실행
{% highlight bash %}
$ rosrun turtlesim turtlesim_node
{% endhighlight %}

**rosnode** [info...] : 노드의 정보를 표시 (발행, 구독 정보)
{% highlight bash %}
$ rosnode info node_name
{% endhighlight %}

**rostopic** [option]: 토픽의 정보를 표시
{% highlight bash %}
$ rostopic info /imu    # 토픽의 정보를 출력(메시지 타입, 노드의 이름 등)
{% endhighlight %}

**roslaunch** [package_name] [file.launch] : 파라미터 값과 함께 패키지에 있는 여러개의 노드를 실행
{% highlight bash %}
$ roslaunch usb_cam sub_cam-test.launch
{% endhighlight %}


### 2.4 ROS에서 제공하는 쓸만한 도구
**rqt_graph**:  노드와 토픽의 관계 정보를 그래프로 출력
{% highlight bash %}
$ rqt_graph
{% endhighlight %}
**RVIZ**:  ROS의 3차원 시각화 도구, 각종 데이터를 보기 좋게 인포그래픽 스타일로 표시
