---
layout: post
title:  "[테스트] 차선 인식 - ver 0.1"
date:   2021-02-21 00:00:00 +0900
category: "Grepp/KDT"
tags:
    - "Image Processing"
    - "Testbed"
---

<iframe width="740" height="363" src="https://www.youtube.com/embed/g0QGmmCVCO4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## 적용 기술
- Calibration
- Adaptive Threshold
- Bird Eye View
- Sliding Window


## 느낀 점
- Calibration
    - 원본 카메라 영상과 Calibration을 적용한 영상을 비교했을 때, 확실히 Calibration을 적용한 영상이 Sliding View를 쓰기 좋았다.
- Adaptive Threshold
    - 이 영상처리 기법 자체가 비용이 많이 드는 듯 하다. 2개 이상 넣으면 검출 속도가 너무 느려진다. 좀 더 빠른 방법으로 검출이 되는 방법을 찾아보는 중이지만, 아직 이것만큼 좋은 성능을 내는 방법을 아직 찾지 못했다.
- Sliding window
    - 하단부터 Scan 영역에 존재하는 nonzero pixel의 갯수를 알아내서, threshold 이상일 경우 차선의 시작점으로 간주한다.
    - Flood fill 알고리즘과 유사한 방법으로, 시작점에서 좌측/좌측상단/상단/우측상단/우측의 scan 영역을 마찬가지로 nonzero pixel의 갯수를 알아내서 threshold 이상일 경우 차선 후보군에 추가한다.
    - 현재 환경 조건에서 영상의 좌측/우측의 차선은 일단 한 개씩만 존재한다고 가정하므로, 어떻게 봐도 차선이 아닌 것들을 걸러내야 한다.
    - Canny 이미지에도 적용했지만, 이렇게 되면 nonzero pixel 갯수로 판별하기 위해 scan 영역의 크기가 작아야 하는데, 그러면 탐색해야 하는 면적이 넓어져서 연산량이 매우 늘어난다.
- 현재 환경에서는 결국 영상 가공에서 거의 모든 것이 좌우 된다고 생각하고 있다. 영상처리를 매우 빠르게 해서 실시간으로 rough하게 정보를 얻고 후처리에 더 신경을 쓸 것인지, 아니면 좀 더 영상처리에 비용을 증가시키더라도 후처리 비용을 감소시킬 것인지도 관건인 것 같다.


## 개선 사항
- Calibration
    - calibration을 적용한 영상의 좌측 영역에 약간의 왜곡이 있었기 떄문에, calibration 다시 수행
- Sliding window sequence
    - 영상의 하단부터 시작되는 차선을 찾지 못한다면: 가장자리 하단에서 시작되는 차선 검출 시도
    - 검출된 점들에 가장 잘 맞는 3차함수 구하기

![3차 함수를 사용해야 하는 예](/assets/grepp/lane_cubic.png)


## 머리에 맴 도는 문제
- Camera
    - Exposure를 약간 낮춘다면, 빛 반사 간섭을 조금 약화시킬 수 있을까?
- Sliding window
    - 영상의 하단부터 시작되는 차선을 찾았지만, 차선의 흐름이 급격한 경우에 대한 문제
        - ex. 하단에서 찾은 차선 시작점부터 왼쪽으로 검출되다가, 감자기 오른쪽으로 검출되는 경우 존재
        - 왼쪽으로 검출되는 차선과 오른쪽으로 검출되는 차선을 분리하고, 하단부터 시작되는 추세를 가진 차선만 추출할 수 있을까?
    - Sobel을 활용하면 영상처리 비용이 적게 든다는 글을 보았는데, 가로/세로/대각선 kernel을 적용한 edge 결과에서 & 연산으로 교점들을 활용할 수 있을까?
- Filter
    - Sliding Window에서 검출된 정보의 오차를 걸러내기 위해서도 필요하고, 앞으로도 계속 필요할 것 같음
    - 강화학습을 적용하던, 초음파를 적용하던 반드시 사용해야 할 문제라고 점점 확신이 들고 있음
