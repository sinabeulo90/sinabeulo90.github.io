---
layout: post
title:  "[실습] 허프변환 기반 차선 찾기"
date:   2021-01-25 02:00:00 +0900
categories: "Grepp/KDT"
tag: OpenCV
---

## 패키지 생성

Ros workspace의 src 폴더에서 hough_drive 패키지를 만든다.

{% highlight bash %}
$ catkin_create_pkg hough_drive std_msgs rospy
{% endhighlight %}

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── hough_drive
        └── src
            ├── hough_find.py       # 허프 변환으로 차선을 찾는 코드
            ├── hough_track.avi     # 트랙을 촬영한 동영상 파일
            └── steer_arrow.png     # 핸들 그림
{% endhighlight %}



## 프로그램 흐름도(hough_find.py)

1. hough_track.avi 동영상에서 영상 프레임 획득
2. 영상 프레임을 OpenCV 함수로 넘겨 처리
3. OpenCV 영상 처리
    - Grayscale: 흑백 이미지로 변환
    - Gaussian Blur: 노이즈 제거
    - Canny Edge: 외곽선 Edge 추출
    - ROI: 관심 영역 잘라내기
    - HoughLinesP: 선분 검출
4. 차선의 위치 찾고 화면 중앙에서 어느 쪽으로 치우쳤는지 파악
5. 핸들을 얼마나 꺽을 지 결정(조향각 설정 각도 계산)
6. 화면에 핸들 그림을 그려서 차량의 주행 방향 표시

{% highlight Python %}
#!/usr/bin/env python

import rospy
import numpy as np
import cv2, random, math, time


# 영상 사이즈: 640 x 480
Width = 640
Height = 480

# ROI 영역: 세로 480부터, 640 x 40
Offset = 420
Gap = 40


# draw lines: 허프 변환 함수로 검출된 모든 선분을 알록달록하게 출력
def draw_lines(img, lines):
    global Offset
    for line in lines:
        x1, y1, x2, y2 = line[0]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = cv2.line(img, (x1, y1+Offset), (x2, y2+Offset), color, 2)


# draw rectangle
def draw_rectangle(img, lops, rpos, offset=0):
    center = (lpos + rpos) / 2
    # lpos 위치에 녹색 사각형 그리기
    cv.rectangle(img, (lpos-5, 15+offset), (lpos+5, 25+offset), (0, 255, 0), 2)
    # rpos 위치에 녹색 사각형 그리기
    cv.rectangle(img, (rpos-5, 15+offset), (rpos+5, 25+offset), (0, 255, 0), 2)
    # lpos, rpos 사이에 녹색 사각형 그리기
    cv.rectangle(img, (center-5, 15+offset), (center+5, 25+offset), (0, 255, 0), 2)
    # 화면 중앙에 빨강 사각형 그리기
    cv.rectangle(img, (315, 15+offset), (325, 25+offset), (0, 0, 255), 2)


# left lines, right lines
def divide_left_right(lines):
    global Width
    
    low_slope_threshold = 0
    high_slope_threshold = 10

    slope = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2 - y1) / float(x2 - x1)
        
        # 선분의 기울기를 구해서 기울기 절대값이 10 이하인 것만 추출
        if (abs(slope) > low_slope_threshold) and (abs(slope) < high_slope_threshold):
            new_lines.append(line[0])

    # divide lines left to right
    # 허프 변환 함수로 검출한 선분들의 기울기를 비교하여, 왼쪽/오른쪽 차선 구분
    left_ines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = new_lines[j]
        slope = slopes[j]
        x1, y1, x2, y2 = Line

        # OpenCV 좌표계에서 아래방향으로 y가 증가하므로, 기울기 계산법이 다르다.
        # 화면의 왼쪽에 있는 선분 중, 기울기가 음수인 것들만 추출
        if (slope < 0) and x2 < Width/2 - 90):
            left_lines.append([Line.tolist()])
        # 화면의 오른쪽에 있는 선분 중에서 기울기가 양수인 것들만 추출
        elif (slope > 0) and (x1 > Width/2 + 90):
            right_lines.append([Line.tolist()])
    
    return left_lines, right_lines


# get average m, b of lines
# - 허프 변환 함수로 찾아낸 직선을 대상으로 Parameter Space(m, b 좌표계)에서
#   m의 평균 값을 먼저 구하고, 그걸로 b의 값을 구한다.
# - m의 평균 값을 구하는 이유는 허프 변환 함수의 결과로 하나가 아닌
#   여러 개의 선이 검출되기 때문에, 찾은 선들의 평균 값을 이용하려고 한다.
def get_line_params(lines):
    # sum of x, y, m
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)
    if size == 0:
        return 0, 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
    
        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)
    
    x_avg = x_sum / (size * 2)
    y_avg = y_sum / (size * 2)
    m = m_sum / size
    b = y_av - m * x_avg

    return m, b


# get lpos, rpos
def get_line_pos(img, lines, left=False, right=False):
    global Width, Height
    global Offset, Gap

    m, b = get_line_params(lines)

    # 차선이 인식되지 않을 경우,
    # left = 0
    # right = Width(640)
    if m == 0 and b == 0:
        if left:
            pos = 0
        if right:
            pos = Width
    else:
        # 직선의 방정식에서 x, y 좌표 계산
        y = Gap / 2
        pos = (y - b) / m
    
        b += Offset
        x1 = (height - b) / float(m)
        x2 = ((Height/2) - b) / float(m)

        # 640x480 원본 이미지의 맨 아래 Height의 x1과 이미지 중간 Height/2의 x2를 구해서,
        # (x1, 480), (x2, 320) 두 점을 잇는 파란색 선을 그린다.
        cv2.line(img, (int(x1), Height), (int(x2), (Height/2)), (255, 0, 0), 3)
    
    return img, int(pos)


# show image and return lpos, rpos: 카메라 영상처리
def process_image(frame):
    global Width
    global Offset, Gap

    # Gray 색상 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur 처리
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Canny Edge 외곽선 따기
    low_threshold = 60
    high_threshold = 70
    edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)

    # HoughLinesP으로 ROI 영역에서 선분 찾기
    roi = edge_img[Offset:Offset+Gap, 0:Width]
    # cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)
    all_lines = cv2.HoughLinesP(roi, 1, math.pi/180, 30, 30, 10)

    # 선분의 왼쪽/오른쪽 것으로 분리
    if all_lines is None:
        return 0, 640
    left_lines, right_lines = divide_left_right(all_lines)

    # 선분의 정보를 받아서 이미지에 차선과 위치 구하기
    frame, lpos = get_line_pos(frame, left_lines, left=True)
    frame, rpos = get_line_pos(frame, right_lines, right=True)

    # ROI 영역 안에서 허프 변환을 통해 차선을 구해서, 랜덤한 색상으로 그리기
    frame = draw_lines(frame, left_lines)
    frame = draw_lines(frame, right_lines)
    frame = cv2.line(frame, (230, 235), (410, 235), (255, 255, 255), 2)

    # 차선과 화면 중앙에 사각형 그리기
    frame = draw_rectangle(frame, lpos, rpos, offset=Offset)

    return (lpos, rpos), frame


def draw_steer(image, steer_angle):
    global Width, Height, arrow_pic

    # 조향 이미지 읽어들이기    
    arrow_pic = cv2.imread("steer_arrow.png", cv2.IMREAD_COLOR)

    origin_Height = arrow_pic.shape[0]
    origin_Width = arrow_pic.shape[1]
    # 이미지 크기 축소를 위한 크기 계산
    steer_wheel_center = origin_Height * 0.74
    arrow_Height = Height / 2
    arrow_Width = (arrow_Height * 462) / 728

    # steer_angle에 비례하여 회전
    matrix = cv.getRotationMatrix2D((origin_Width/2, steer_wheel_center), (steer_angle) * 2.5, 0.7)

    # 이미지 크기를 영상에 맞춤
    arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_Width+60, origin_Height))
    arrow_pic = cv2.resize(arrow_pic, dsize=(arrow_Width, arrow_Height), interpolation=cv2.INTER_AREA)

    # 전체 그림 위에 핸들 모양의 그림 오버레이
    gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)
    
    arrow_roi = image[arrow_Height:Height, (Width/2-arrow_Width/2):(Width/2+arrow_Width/2)]
    arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
    
    res = cv2.add(arrow_roi, arrow_pic)
    image[(Height-arrow_Height):Height, (Width/2-arrow_Width/2):(Width/2+arrow_Width/2)] = res

    # "steer" 타이틀로 된 화면에 표시
    # 원본 사진 + 검출 차선 + 평균 차선 + 차선 위치 표시 + 화면 중앙 표시 + 핸들 그림 + 초향각 화살표 표시
    cv2.imshow("Steer", image)


def start():
    global image, Width, Height

    # 동영상 파일 열기
    cap = cv2.VideoCapture("hough_track.avi")

    while not rospy.is_shutdown():
        # 동영상 파일에서 이미지 한장 읽기
        ret, image = cap.read()
        time.sleep(0.03)

        # 허프 변환 기반으로 영상 처리 진행 + 차선을 찾고 위치에 표시
        pos, frame = process_image(image)

        # 왼쪽/오른쪽 차선의 중점과 화면 중앙과의 차이를 가지고 핸들 조향각을 결정해서 핸들 그림 표시
        center = (pos[0] + pos[1]) / 2
        angle = 320 - center
        steer_angle = angle * 0.4
        draw_steer(frame, steer_angle)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    start()
{% endhighlight %}



## hough_find.py 실행

~/xycar_ws/src/hough_drive/src 폴더 아래에 `hough_track.avi` 파일과 `steer_arrow.png` 파일을 준비한다.

{% highlight bash %}
$ chmod +x hough_find.py
$ python hough_find.py
{% endhighlight %}



## 실행 결과

- 왼쪽과 오른쪽 차선에 녹색 사각형
- 왼쪽/오른쪽의 중간 위치에 녹색 사각형
- 화면 중앙에 빨간색 사각형
- 아래 중앙에 핸든 모양과 화살표
- 처음부터 직선이 아예 검출되지 않으면, 화면에 아무것도 표시되지 않음
