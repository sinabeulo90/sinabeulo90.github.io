---
layout: post
title:  "[실습] 회전, Affine, 원근 변환"
date:   2021-01-25 05:00:00 +0900
categories: "Grepp/KDT"
tag: OpenCV
plugins: mathjax
---

## Rotation 변환(회전)

이미지를 회전하기 위해서 sin, cos함수를 이용해서 변환하면 되지만, 일반적인 회전 행렬은 2x2 행렬이므로 Affine에서는 사용할 수 없다. 2x3 행렬을 만들어서 사용해야 하므로, `cv2.getRotationMatrix2D` 함수를 사용하여 2x3 행렬을 구해서 사용한다.

일반적인 회전 행렬

$$
\begin{pmatrix}
x' \newline
y'
\end{pmatrix}

=

\begin{pmatrix}
\cos\theta & -\sin\theta \newline
\sin\theta & \cos\theta
\end{pmatrix}

\cdot

\begin{pmatrix}
x \newline
y \newline
\end{pmatrix}
$$

2x3 Affine 변환용 행렬

$$
\begin{pmatrix}
\alpha & \beta & (1 - \alpha) \cdot x_{center} - \beta \cdot y_{center} \newline
-\beta & \alpha & \beta \cdot y_{center} + (1 - \alpha) \cdot x_{center}
\end{pmatrix}
$$

$$
\alpha = scale \cdot \cos\theta,\; \beta = scale \cdot \sin\theta
$$


우리가 일반적으로 사용하는 행렬과 Affine 변환용 행렬을 비교해보면, $\cos\theta$ 부분과 $\beta$ 부분의 부호가 반대로 되어 있음을 알 수 있는데, 이는 우리가 일반적으로 상요하는 데카르트 좌표계와 영상에서 사용하는 좌표계가 서로 $x$축 대칭이기 때문이다.



### rotation1.py

{% highlight Python %}
import cv2
import numpy as np

img = cv2.imread("girl.png")
rows, cols = img.shape[0:2]     # 영상의 크기


d45 = 45.0 * np.pi / 180        # 45도 회전 각도를 라디안 값으로 변경
d90 = 90.0 * np.pi / 180        # 90도 회전 각도를 라디안 값으로 변경

# (0, 0) 기준으로 시계방향으로 45도 회전한 뒤 (cols//2, -1*rows//2)만큼 이동하는 회전 행렬
m45 = np.float32([[np.cos(d45), -1*np.sin(d45), cols//2],
                  [np.sin(d45), np.cos(d45), -1*rows//4]])
# (0, 0) 기준으로 시계방향으로 90도 회전한 뒤 (cols, 0)만큼 이동하는 회전 행렬
m90 = np.float32([[np.cos(d90), -1*np.sin(d90), cols],
                  [np.sin(d90), np.cos(d90), 0]])


r45 = cv2.warpAffine(img, m45, (cols, rows))    # 45도 회전
r90 = cv2.warpAffine(img, m90, (cols, rows))    # 90도 회전

cv2.imshow("original", img)
cv2.imshow("r45", r45)
cv2.imshow("r90", r90)
cv2.waitKey(0)
cv2.destoryAllWindows()
{% endhighlight %}



### rotation1.py 실행

{% highlight bash %}
$ chmod +x rotation1.py   # rosrun으로 사용할 경우
$ python rotation1.py
{% endhighlight %}



### 회전 행렬을 구하는 OpenCV 함수

`mtrx = cv2.getRotationMatrix2D(center, angle, scale)`
- center: 회전 축 중심 좌표(x, y)
- angle: 회전할 각도, 60진법
- scale: 확대 및 축소 비율
- 회전 축과 회전 각도를 정하여, 얼만큼의 확대/축소할지에 대한 행렬을 만듬



### rotation2.py

{% highlight Python %}
import cv2
import numpy as np

img = cv2.imread("girl.png")
rows, cols = img.shape[0:2]     # 영상의 크기

# 중앙 회전축, 45도 반시계 방향 회전, 0.5배 축소 행렬
m45 = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 0.5)
print(m45)
# 중앙 회전축, 90도 반시계 방향 회전, 1.5배 확대 행렬
m90 = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1.5)
print(m90)

r45 = cv2.warpAffine(img, m45, (cols, rows))    # 45도 회전, 0.5배 축소
r90 = cv2.warpAffine(img, m90, (cols, rows))    # 90도 회전, 1.5배 확대

cv2.imshow("original", img)
cv2.imshow("r45", r45)
cv2.imshow("r90", r90)
cv2.waitKey(0)
cv2.destoryAllWindows()
{% endhighlight %}



## Affine 변환

크기 변환, 이동 변환, 회전 변환에서 영상의 직선들이 평행한 상태를 유지하듯이, Affine은 평행한 특성을 유지하면서 좀더 자유로운 형태를 만들 수 있다. 3개의 점을 통해 변환 2x3 행렬을 만들 수 있는데, 직접 만들기 보다는 `cv2.getAffineTransform` 함수를 통해 얻을 수 있다.



### affine.py

{% highlight Python %}
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("chess.png")
rows, cols = img.shape[0:2]     # 영상의 크기

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])     # 변환 전 3개 점의 좌표
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])   # 변환 완료 후 3개 점의 좌표

# 기존 점이 새로운 점으로 이동시킬 때 필요한 행렬 찾기
M = cv2.getAffineTransform(pts1, pts2)
print(M)

# 구해진 행렬을 적용하여 이미지 변환
dst = cv2.wrapAffine(img, M, (cols, rows))

plt.subplot(121), plt.imshow(img), plt.title("Input")
plt.subplot(122), plt.imshow(dst), plt.title("Output")
plt.show()
{% endhighlight %}



### affine.py 실행

{% highlight bash %}
$ chmod +x affine.py   # rosrun으로 사용할 경우
$ python affine.py
{% endhighlight %}



## Perspective 변환(원근 변환)

원근법을 적용한 변환으로, 직선의 성질만 유지되고, 선의 평행성은 유지가 되지 않는 변환이다. 예를 들면, 기차길은 서로 평행하지만 원근 변환을 거치면 평행성은 유지되지 못하고 하나의 점에서 만나는 것처럼 보인다. 반대의 변환도 가능하기 때문에, 차선 추출에도 사용할 수 있다.

원근 변환에서 사용할 변환 행렬은 `cv2.getPerspectiveTransform` 함수를 통해 얻을 수 있는데, 이때는 이동할 4개 점의 좌표가 필요하다. 원근 변환 행렬은 3x3행렬이며, `cv2.warpPerspective` 함수에 적용하여 이미지를 변환을 한다.

예시.

{% highlight Python %}
# 변환 전과 후의 기준이 되는 4개 점의 좌표값 지정
# 점 표시 방향: [좌상단, 좌하단, 우상단, 우하단]
#   이동 전        변환 후
#   1 -- 3       1 --- 2
#   |    |      /       \
#   |    |     /         \
#   2 -- 4    3 --------- 4
pts1 = np.float32([ 이동 전 4점의 좌표 ])
pts2 = np.float32([ 변환 후 4점의 좌표 ])

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (cols, rows))
{% endhighlight %}



### perspective.py

{% highlight Python %}
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("chess.png")
rows, cols = img.shape[0:2]     # 영상의 크기

# 변환 전 4개 점의 좌표
pts1 = np.float32([[20, 20], [20, 280], [380, 20], [380, 280]])
# 변환 완료 후 4개 점의 좌표
pts2 = np.float32([[100, 20], [20, 280], [300, 20], [380, 280]])

# 4개 점의 위치에 다른 색깔로 원 그리기
cv2.circle(img, (20, 20), 20, (255, 0, 0), -1)
cv2.circle(img, (20, 280), 20, (0, 255, 0), -1)
cv2.circle(img, (380, 20), 20, (0, 0, 255), -1)
cv2.circle(img, (380, 380), 20, (0, 255, 255), -1)

# 4개 점의 이동 정보를 가지고 행렬 계산
M = cv2.getPerspectiveTransform(pts1, pts2)
print(M)

# 구해진 행렬을 적용하여 이미지 변환
dst = cv2.wrapAffine(img, M, (cols, rows))

plt.subplot(121), plt.imshow(img), plt.title("Input")
plt.subplot(122), plt.imshow(dst), plt.title("Output")
plt.show()
{% endhighlight %}



### perspective.py 실행

{% highlight bash %}
$ chmod +x perspective.py  # rosrun으로 사용할 경우
$ python perspective.py
{% endhighlight %}
