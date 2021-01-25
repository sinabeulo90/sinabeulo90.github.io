---
layout: post
title:  "[실습] 평행이동, 확대/축소"
date:   2021-01-25 04:00:00 +0900
categories: "Grepp/KDT"
tag: OpenCV
plugins: mathjax
---

## 패키지 생성

Ros workspace의 src 폴더에서 sliding_drive 패키지를 만든다.

{% highlight bash %}
$ catkin_create_pkg sliding_drive std_msgs rospy
{% endhighlight %}

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── sliding_drive
        └── src
            ├── girl.png
            └── chess.png
{% endhighlight %}



## Translation 변환(평행 이동)

이미지를 이동하려면 원래 있전 좌표에 이동시키려는 거리만큼 더하면 된다.

$$
x_{new} = x_{old} + d_1
y_{new} = y_{old} + d_2
$$

이를 행렬식으로 표현하면 아래와 같다.

$$
\begin{pmatrix}
1 & 0 & d_1 \newline
0 & 1 & d_2
\end{pmatrix}

\cdot

\begin{pmatrix}
x \newline
y \newline
1
\end{pmatrix}

=

\begin{pmatrix}
d_1 + x \newline
d_2 + y
\end{pmatrix}
$$

즉 이미지의 좌표를 이동하려면 2x3 행렬을 사용하면 되고, 이를 변환행렬이라고 말한다.



### 변환 행렬을 사용하는 OpenCV 함수

`dst = cv2.warpAffine(src, matrix, dsize, dst, flags, borderMode, borderValue)`
- src: 원본 이미지, numpy 배열
- matrix: 2x3 변환 행렬, dtype=float32
- dsize: 결과 이미지의 크기, (width, height)
- options
    - dst: 결과 이미지
    - flags: 보간법 알고리즘 플래그
        - cv2.INTER_LINEAR: 인접한 4개 픽셀 값에 거리 가중치 사용, default
        - cv2.INTER_NEAREST: 가장 가까운 픽셀 값 사용
        - cv2.INTER_AREA: 픽셀 영역 관계를 이용한 resampling
        - cv2.INTER_CUBIC: 인접한 16개 픽셀 값에 거리 가중치 사용
    - borderMode: 외곽영역 보정 플래그
        - cv2.BORDER_CONSTANT: 고정 색상 값
        - cv2.BORDER_REPLICATE: 가장자리 복제
        - cv2.BORDER_WRAP: 반복
        - cv2.BORDER_REFLECT: 반사
    - borderValue: 외곽영역 보정 플래그가 cv2.BORDER_CONSTANT일 경우 사용할 색상 값, default=0
- ex. `dst = cv2.warpAffine(img, M, (cols, rows))`



### translation.py

{% highlight Python %}
import cv2
import numpy as np

img = cv2.imread("girl.png")
rows, cols = img.shape[0:2]     # 영상의 크기

dx, dy = 100, 50                # 이동할 픽셀 거리

mtrx = np.float32([[1, 0, dx],  # 변환 행렬 생성
                    0, 1, dy]])

# 단순 이동
# 영상이 이동했는지 알기 위해, 출력되는 영상의 크기를 조정: (cols + dx, rows + dy)
dst = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy))

# 탈락된 외곽 픽셀을 파랑색으로 보정
dst2 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255, 0, 0))

# 탈락된 외곽 픽셀을 원본을 반사시켜서 보정
dst3 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT)

cv2.imshow("original", img)
cv2.imshow("trans", dst)
cv2.imshow("BORDER_CONSTANT", dst2)
cv2.imshow("BORDER_REFLECT", dst3)
cv2.waitKey(0)
cv2.destoryAllWindows()
{% endhighlight %}



### translation.py 실행

{% highlight bash %}
$ chmod +x translation.py   # rosrun으로 사용할 경우
$ python translation.py
{% endhighlight %}



## Scaling 변환(확대 축소)

일정 비율로 확대 및 축소시키기 위해, 기존 좌표에 특정 값을 곱하면 된다.

$$
x_{new} = a_1 x x_{old}
y_{new} = a_2 x y_{old}
$$

이를 행렬식으로 표현하면 아래와 같다.

$$
\begin{pmatrix}
a_1 & 0 & 0 \newline
0 & a_2 & 0
\end{pmatrix}

\cdot

\begin{pmatrix}
x \newline
y \newline
1
\end{pmatrix}

=

\begin{pmatrix}
a_1 x \newline
a_2 y
\end{pmatrix}
$$

즉 이미지를 축소/확대하려면 2x3 행렬을 사용하면 되고, 이를 변환행렬이라고 말한다.



### scaling.py

{% highlight Python %}
import cv2
import numpy as np

img = cv2.imread("girl.png")
rows, cols = img.shape[0:2]     # 영상의 크기

m_small = np.float32([[0.5, 0, 0],  # 축소 변환 행렬 생성
                       0, 0.5, 0]])
m_big = np.float32([[2, 0, 0],      # 확대 변환 행렬 생성
                     0, 2, 0]])

# 보간법 없이 축소
dst1 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)))

# 보간법 적용한 축소: INTER_AREA 권장
dst2 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)), None, cv2.INTER_AREA)

# 보간법 없이 확대
dst3 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)))

# 보간법 적용한 확대: INTER_CUBIC 권장, 단 시간이 많이 걸린다.
dst4 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)), None, cv2.INTER_CUBIC)

cv2.imshow("original", img)
cv2.imshow("small", dst1)
cv2.imshow("small INTER_AREA", dst2)
cv2.imshow("big", dst3)
cv2.imshow("big INTER_CUBIC", dst4)
cv2.waitKey(0)
cv2.destoryAllWindows()
{% endhighlight %}


### scaling.py 실행

{% highlight bash %}
$ chmod +x scaling.py   # rosrun으로 사용할 경우
$ python scaling.py
{% endhighlight %}



### 크기 조정 OpenCV 함수

`cv2.resize(src, dsize, dst, fx, fy, interpolation)`
- src: 입력 원본 이미지
- dsize: 확대/축소를 원하는 목표 이미지 크기
    - **생략하면 fx, fy 배율 적용**
- dst: 결과 이미지
- fx, fy: 변경할 크기 배율
    - **dsize가 주어지면, dsize를 우선 적용**
- interpolation: 보간법 알고리즘 선택 플래그, cv2.warpAffine 함수에서 사용하는 것과 동일



### resizing.py

{% highlight Python %}
import cv2
import numpy as np

img = cv2.imread("girl.png")
rows, cols = img.shape[0:2]     # 영상의 크기

# 크기 지정으로 축소
dst1 = cv2.resize(img, (int(width*0.5), int(height*0.5)), interpolation=cv2.INTER_AREA)

# 배율 지정으로 확대: 가로 0.5배, 세로 1.5배
dst2 = cv2.warpAffine(img, None, None, 0.5, 1.5, cv2.INTER_CUBIC)

cv2.imshow("original", img)
cv2.imshow("small", dst1)
cv2.imshow("big", dst2)
cv2.waitKey(0)
cv2.destoryAllWindows()
{% endhighlight %}



### resizing.py 실행

{% highlight bash %}
$ chmod +x resizing.py  # rosrun으로 사용할 경우
$ python resizing.py
{% endhighlight %}
