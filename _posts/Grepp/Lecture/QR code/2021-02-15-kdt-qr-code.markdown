---
layout: post
title:  "QR코드 이해와 활용"
date:   2021-02-15 01:00:00 +0900
category: "Grepp/KDT"
tag: "QR Code"
---

## QR코드 소개

### 자율주행 자동차와 QR코드
- 교통 표지판 역할
- 유용한 정보의 전달 방법


### 로봇과 QR코드
- 경로 안내
- 주변 상황 인지
- ex. Magnetic Guide, QR Code, Laser SLAM, Camera


### QR코드
- Quick Response code
- 정보를 담고 있는 2차원 그래픽 이미지


### QR코드에 담긴 데이터
- 숫자 최대 7,089자
- 문자 최대 4,296자
- 한자 최대 1,817자


### QR코드 vs. 바코드
- 바코드
    - 일상 생활에서 많이 저바는 정보 저장 이미지
    - 1차원으로 20여자의 숫자 정보 저장 가능
- QR코드
    - 2차원의 이미지 패턴으로 정보를 나타내는 코드
    - 2차원 가로/세로 형태로, 숫자는 최대 7,089자, 문자는 4,296자, 한자는 최대 1,817자를 넣을 수 있음
    - 개발사인 덴소웨이브가 특허권을 행사하지 않기 때문에 무료로 이용 가능
    - 정사각형 형태로 기울어져 있어서 상관 없다.


### QR코드의 특징
- 작은 공간에 인쇄 가능: 바코드와 동일한 정보량을 1/10 크기로 표시
- 어느 방향에서도 인식 가능
    - QR코드 안에 3개의 '위치 검출 패턴'이 들어있음
    - A, B, C 어느 방향에서라도 백색과 흑색 셀의 비율이 반드시 1:1:3:1:1이 된다.
    - 따라서 360° 모든 방향에서 고속 인식과 판독이 가능
- 오염과 손성에 강함
    - 오류 복원 기능이 있음
    - 최대 30% 복원 가능
    - QR코드에 디자인 삽입 가능
- 연속 기능 제공
    - 하나의 데이터를 분한해서 여러 개의 QR코드로 표시 가능
    - 최대 16개로 분할 가능하므로 좁고 긴 띠 모양으로 표시 가능


### QR코드의 규격
- QR코드를 구성하는 최소 단위를 셀이라고 함.(흑백의 정사각형)
- QR코드는 셀의 조합으로 표시됨
- 최소 심벌 크기: 21 x 21 셀
- 최대 심벌 크기: 177 x 177 셀
- 최대 데이터 용량
    - 숫자 최대 7,089자
    - 문자 최대 4,296자
    - 한자 최대 1,817자


### QR코드의 2가지 모델
- 모델 1
    - 최초로 제작된 QR코드
- 모델 2
    - 위치 조정을 개선하도록 얼라인먼트 패턴이 추가
    - 모델 1보다 많은 데이터를 수용
- [꺼진(줄 알았던) 기술도 다시 보자!… ‘1994년생 QR코드’ 이야기](http://bit.ly/2AJzlQG)


### QR코드의 규격
- 경계: 마진
    - QR코드 주위의 공백 부분
    - 모델 1, 모델 2에서는 4셀 분의 공백이 필요
    - Micro QR코드에서는 2셀 분의 공백이 필요
- 파인더: 위치 검출 패턴
    - QR코드 안에 3개의 '위치 검출 패턴'이 들어 있음
    - A, B, C 어느 방향에서라도 백색과 흑색 셀의 비율이 반드시 1:1:3:1:1이 된다.
    - 따라서 360° 모든 방향에서 고속 인식과 판독이 가능
- 얼라인먼트 패턴
    - 위치 검출 패턴보다 작은 크기
    - 시야각에 따른 왜곡을 보정하기 위해 사용(Perspective transformation)
    - 모델 2에 적용
- 타이밍 패턴
    - 백색과 흑색 셀이 교대로 배치되어, 심벌 내의 셀 좌표를 결정하는데 사용

- ![QR코드의 6대 구성요소와 의미](/assets/grepp/qrcode.jpg)


### QR코드 만들기
- [네이버 QR코드](https://qr.naver.com)
- [MQR Free Formatted Text](http://mqr.kr/generate/text/)



## QR코드 응용 프로그램 만들기


### QR코드 인식을 위한 소프트웨어
- [ZBar bar code reader](http://zbar.sourceforge.net)
- [pyzbar](https://pypi.org/project/pyzbar/)


### ZBar 파이썬 라이브러리 설치
- ZBar 파이썬 라이브러리인 pyzbar 패키지 설치
{% highlight Python %}
$ sudo apt install python-pip
$ pip install pyzbar
{% endhighlight %}


### 이미지에서 QR코드 인식하기
- QR_code1.png 이미지에서 QR코드를 인식하는 파이썬 프로그램

{% highlight bash %}
$ mkdir -p qr_code/src
{% endhighlight %}

{% highlight bash %}
~/xycar_ws/
├── build
├── devel
└── src
    └── qr_code
        └── src
            └── qr_read_img.py
{% endhighlight %}

- qr_read_img.py
    - 주어진 이미지 파일(QR_code1.png)에서 QR코드를 찾아서
    - 테두리에 빨간색으로 그리고
    - 담고 있는 정보를 위에서 표시(터미널에도 표시)

{% highlight Python %}
import cv2
from pyzbar import pyzbar   # pyzbar 라이브러리를 임포트

image = cv2.imread("QR_code1.png")  # QR코드 이미지 파일 읽어들이기
qrcodes = pyzbar.decode(image)      # QR코드 찾기

for qrcode in qrcodes:
    (x, y, w, h) = qrcode.rect      # QR코드 크기 정보 가져오기
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)    # QR코드 테두리 그리기
        
    text = qrcode.data          # QR코드에 담긴 정보 가져오기
    cv2.putText(image, (x, y-6), cv2.FONT_HERSHY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print("[INFO] Found", text) # QR코드에 담긴 정보 표시하기

cv2.imshow("image", image)      # image를 화면에 표시
cv2.waitKey(0)                  # 키 입력을 기다림
{% endhighlight %}


### 카메라 영상에서 QR코드 인식하기

- qr_read_cam.py
    - OpenCV에서 카메라 영상을 가져와서 QR코드 인식하기
    - 카메라 영상 이미지에 QR코드 위치와 내용 표시
        
{% highlight Python %}
import cv2
from pyzbar import pyzbar

cap = cv2.VideoCapture(0)   # 카메라에서 영상 정보 가져올 준비

while (True):
    # 카메라에서 영상 정보를 가져와서, QR코드 찾기
    ret, image = cap.read()
    qrcodes = pyzbar.decode(image)

    for qrcode in qrcodes:
        # QR코드 크기 정보를 획득하고, 테두리를 빨간색으로 그리기
        (x, y, w, h) = qrcode.rect
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # QR코드에 담긴 정보를 획득해서 표시하기
        text = qrcode.data
        cv2.putText(image, (x, y-6), cv2.FONT_HERSHY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print("[INFO] Found", text)
    
    # 'q' 키보드 입력이 없으면 계속 반복
    # 있으면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
{% endhighlight %}


### ROI 영역에서 QR코드 인식하기

- qr_read_roicam.py
    - 영상의 지정된 영역에서 QR코드 인식
        
{% highlight Python %}
import cv2
from pyzbar import pyzbar

cap = cv2.VideoCapture(0)   # 카메라에서 영상 정보 가져올 준비

while (True):
    ret, image = cap.read() # 카메라에서 영상 정보를 가져오기

    cv2.line(image, (319, 0), (319, 479), (0, 255, 0), 5)   # 이미지 중간에 세로줄 긋기

    # 왼쪽 반쪽 영역만 잘라내서 QR코드 찾기
    image2 = image[0:479, 0:319]
    qrcodes = pyzbar.decode(image2)

    for qrcode in qrcodes:
        # QR코드 크기 정보를 획득하고, 테두리를 빨간색으로 그리기
        (x, y, w, h) = qrcode.rect
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # QR코드에 담긴 정보를 획득해서 표시하기
        text = qrcode.data
        cv2.putText(image, (x, y-6), cv2.FONT_HERSHY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print("[INFO] Found", text)
    
    # 'q' 키보드 입력이 없으면 계속 반복
    # 있으면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
{% endhighlight %}
