---
layout: post
title:  "딥러닝 기초"
date:   2021-02-22 01:00:00 +0900
category: "Grepp/KDT"
tag: "Reinforcement Learning"
---
## Deep Learning

### AL / ML / DL

> Artifical Intelligence: 인간의 학습능력과 추론능력, 지각능력, 자연언어의 이해 능력 등을 컴퓨터 프로그램으로 실현한 기술
>> Machine Learning: 인공지능의 연구 분야 중 하나로, 인간의 학습능력과 같은 기능을 컴퓨터에서 실현하고자 하는 기술 및 기법
>>> Deep Learning: 다층 구조 형태의 신경망을 기반으로 하는 머신러닝의 한 분야로 다양의 데이터로부터 높은 수준의 추상화 모델을 구축하고자 하는 기법


### Machine Learning

- Machine Learning의 학습 방법
    - Supervised Learning: 정답을 아는 데이터를 이용하여 학습을 수행
        - Classification
        - Regression
    - Unsupervised Learning: 정답을 모르고 데이터의 특징만을 이용하여 학습 수행
        - Clustering
        - Semi-supervised Learning
    - Reinforcement Learning: 보상을 통해 학습을 수행
        - Game: AlphaGo, AlphaStar
- 2012년 부터 급격한 발전이 일어나게 된 이유
    - Algorithm
        - Back Propagation
        - Convolutional Neural Network
        - Recurrent Neural Network
        - Dropout
    - GPU
    - Big Data
- DeepLearning about 2012
    - 손글씨 인식
    - 표지판 인식
    - 텍스트 생성


### Artifical Neural Network(ANN)

- 사람의 뇌와 유사한 방법으로 정보를 처리하는 알고리즘
- 인간의 두뇌는 뉴런(Neuron)이라는 기본 단위로 집합체로 구성
- ANN은 Perception이라는 뉴런을 모방한 기본 단위의 집합으로 구성
- Non-Linear Functions(Activation functions)
    - ReLU
- 입력층(Input Layer), 은닉층(Hidden Layer), 출력층(Output Layer)
- Forward Propagation, Back Propagation
- ANN의 학습 과정
    - Loss Function(손실 함수, 오류 함수)
    - 손실 함수를 최소화하도록 weight와 bias를 학습하는 것이 인공신경망 학습의 목표
    - Mean Squared Error, Huber Loss, Corss Entropy, ...
    - Gradient descent
        - Error를 최소로 하는 최적의 weight 학습을 위해 최적화 기법인 gradient descent 이용
        - Mini-batch Gradient Descent: 전체 데이터에서 임의로 일부 데이터를 추출하여 학습
- Fitting
    - Underfitting: 학습이 부족하여 제대로 된 예측을 수행하지 못함
    - Good Fitting: 적절하게 학습이 수앻되어 좋은 예측 수행
    - Overfitting
        - 과도하게 학습이 수행되어 training set에 대해서는 좋은 예측
        - 처음 보는 데이터에 대해 나쁜 성능(Training과 test set으로 데이터를 나눔)
- Dataset
    - Training Set: 학습을 위해 사용하는 데이터셋(전체 데이터의 약 70~80%)
    - Test Set: 학습이 완료된 후 성능을 테스트하기 위해 사용하는 데이터셋(전체 데이터의 약 10~20%)
    - Validation Set: 하이퍼 파라미터 튜닝과 오버피팅 여부를 확인하기 위해 사용(전체 데이터의 약 10~20%)
- Overfitting이 발생하는 경우 Training set의 손실함수 값은 줄어들지만, Validation set의 손실합수 값은 어느 지점부터 증가하기 시작(학습되지 않은 데이터에 대해 성능이 감소함)
- Dropout
    - Overfitting을 방지하는 대표적인 기법 중 하나
    - 특정 확률에 따라 뉴런을 0으로 만들어서 네트워크의 결과에 변화를 주는 기법
- ANN의 문제점
    - Input의 일부만 변경되어도 다른 Input으로 인식
    - 모든 데이터에 대응하기 위해서는 많은 데이터가 필요
    - 학습에 대한 시간이 많이 필요하고 성능도 제한적


### Convolutional Neural Network (CNN)
- Convolution
    - Filter를 이용하여 합성곱 연산을 수행
    - 필터에 따라 이미지의 다른 특징을 추출
    - 1개의 필터는 1개의 Convolution Output을 도출
    - 각각의 필터마다 다른 특징의 결과 도출: 각각 convolution output에서 나타나는 특징이 다름
- Convolution 관련 용어
    - Channel: 이미지의 경우 3차원 데이터(높이x폭xChannel수)
    - 흑백 이미지의 경우: 1channel
    - 컬러 이미지의 경우: 3channel(RGB 채널)
    - Kernel size
        - convolution을 하는 필터의 크기
        - 특징을 추출할 영역의 크기를 결정
    - Stride
        - 필터가 한번에 이동하는 정도를 나타내는 파라미터
        - Stride의 크기에 따라 convolution output의 크기 변화
    - 일반적으로 Convolution을 수행할 경우, output의 크기가 줄어든다.
    - Padding: convolution output의 크기를 조절하기 위해 바깥에 테두리를 쳐주는 기법
        - Padding을 하지 않는 경우: 32x32 이미지에 5x5 filter를 적용한 결과: 30x30
        - Padding을 하는 경우: 36x36 이미지에 5x5 filter를 적용한 결과: 32x32
            - 이미지 크기를 유지할 수 있음
    - Pooling
        - 특정 영역 내부의 정보를 압축하는 방식
        - Max polling: Window 내부의 값 중 가장 큰 값만을 선택
        - Average polling: Window 내부의 값을 평균
        - Convolution과 마찬가지로 filter size나 stride 조절 가능
- 예시
    1. 입력: 80x80x3
    2. 합성곱
        - 필터 크기: 3x3x3
        - 필터 수: 4개
        - 스트라이드: 2x2
        - 패딩: Same
    3. 피쳐맵: 40x40x4
    4. 합성곱
        - 필터 크기: 3x3x4
        - 필터 수: 8개
        - 스트라이드: 2x2
        - 패딩: Same
    5. 피쳐맵: 20x20x8
    6. 합성곱
        - 필터 크기: 3x3x8
        - 필터 수: 16개
        - 스트라이드: 2x2
        - 패딩: Same
    7. 피쳐맵: 10x10x16
    8. 벡터로 변환: 데이터 크기: 1600
    9. 인공신경망 연산: 은닉층 크기: 512
    10. 인공신경망 연산: 은닉층 크기: 256
    11. 출력: 출력의 크기 4
- 이미지 분석, 패턴 분석 등 locality를 가지는 신호에 주로 이용
- 이미지를 convolution한 결과를 ANN의 input에 대입
- 이미지의 특징을 이용
    - 다양한 filter를 이용하므로 여러 특징 도출 가능
    - 이미지의 주된 특징만 이용하므로 변형에 강인

### Classification 코드(MNIST)
- MNIST
    - 0~9로 이루어진 손글씨 숫자 데이터
    - 28x28 크기의 흑백 이미지
    - 60,000개의 학습 데이터, 10,000개의 테스트 데이터로 구성
- [Pytorch](https://pytorch.org)
    - 파이썬을 위한 오픈소스 딥러닝 라이브러리
    - 페이스북에서 개발
    - 구현이 직관적이고 간결하여 처음 딥러닝을 시작하는 사용자들에게 좋음
- 목표: Pytorch를 이용해 MNIST 데이터를 분류하는 CNN 네트워크 구현
    - ex. 입력 이미지가 7인 경우, 딥러닝 분류 결과 7을 출력하도록
- 네트워크 구조
    - 3번의 convolution, 3번의 인공신경망 연산 수행
    - 출력은 총 10개의 값을 도출
        - 각 값은 입력이 0~9 중 각 숫자일 확률을 나타냄
        - ex. 출력: [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
            - 입력이 0일 확률: 0%
            - 입력이 1~8일 확률: 10%
            - 입력이 9일 확률: 20%

{% highlight Python %}# 라이브러리 불러오기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np

# gpu가 있는 경우 딥러닝 연산을 gpu로 수행, 그렇지 않은 경우 cpu로 수행
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 파라미터 설정
batch_size = 128            # 한번 학습시 128개의 데이터를 통해 학습
num_epochs = 10             # 모든 데이터에 대해 10번 학습 수행
learning_rate = 0.00025     # 학습 속도 결정
                            # 너무 값이 작으면, 학습 속도가 느림
                            # 너무 값이 크면, 최적으로 학습하지 못합

# MNIST 데이터 다운로드
trn_dataset = datasets.MNIST("./mnist_data/",
                             download=True,
                             train=True,    # 학습 데이터로 사용
                             transform=transforms.Compose([
                                 transforms.ToTensor()]))  # Pytorch 텐서(Tensor)의 형태로 데이터 출력
val_dataset = datasets.MNIST("./mnist_data/",
                              download=False,
                              train=False,
                              transform=transforms.Compose([transforms.ToTensor()]))

# DataLoader 설정
trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# CNN 네트워크
class CNNClassifier(nn.Module):
    def __init__(self):
        # 네트워크 연산에 사용할 구조 설정
        super(CNNClassifier, self).__init__()   # 항상 torch.nn.Module을 상속받고 시작

        # Conv2d(입력의 채널, 출력의 채널, Kernel size, Stride)
        self.conv1 = nn.Conv2d(1, 16, 3, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)

        # Linear(입력 노드의 수, 출력 노드의 수)
        self.fc1 = nn.Linear(64*4*4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10) 

    def forward(self, x):
        # 네트워크 연산 수행

        # Convolution 연산 후 ReLU 연산
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Convolution의 연산 결과 (batch_sizex(64*4*4))를 (batch_sizex64*4*4)로 변환
        x = x.view(x.size(0), -1)

        # Fully connected 연산 수행 후 ReLU 연산
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Fully connected 연산
        x = self.fc3(x)

        # 최종 결과에 Softmax 연산 수행
        # - 출력 결과를 확률로 변환(합=1이 되도록)
        # - 입력에 대한 결과의 확률으 ㄹ알 수 있
        return F.softmax(x, dim=1)
    
# 정확도 도출 함수
# - y: 네트워크의 연산 결과
#   - 각 숫자에 대한 확률을 나타냄
#   - 하나의 입력에 대해 10개의 값을 가짐: batch_sizex10
# - label: 실제 결과
#   - 현재 입력이 어떤 숫자인지의 값을 보여줌: batch_sizex1
def get_accuracy(y, label):
    # argmax 연산을 통해 확률 중 가장 큰 값의 인덱스를 반환하여 label과 동일한 형식으로 변환
    y_idx = torch.argmax(y, dim=1)
    result = y_idx-label

    # 모든 입력에 대해 정답을 맞춘 갯수를 전체 개수로 나눠주어 정확도를 반환
    num_correct = 0
    for i in range(len(result)):
        if result[i] == 0:
            num_correct += 1
    return num_correct / y.shape[0]


# 네트워크, 손실함수, 최적화기 선언

# 네트워크 정의
# - CNNClassifier 클래스 호출
# - 설정한 device에서 딥러닝 네트워크 연산을 하도록 설정
cnn = CNNClassifier().to(device)

# 솔신 함수 설정
# - Cross Entropy 함수: 분류 문제에서 많이 사용하는 손실함수
criterion = nn.CrossEntropyLoss()

# 최적화기(Optimizer) 설정
# - 딥러닝 학습에서 주로 사용하는 Adam Optimizer 사용
# - cnn 네트워크의 파라미터 학습, 학습률 설정
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
# 한 epoch에 대한 전체 미니 배치의 
num_batches = len(trn_loader)


# 학습 및 검증 수행

# 학습 수행
for epoch in range(num_epochs):     # epoch 반복문
    # 학습시 손실함수 값과 정확도를 기록하기 위한 리스트
    trn_loss_list = []
    trn_acc_list = []
    # 1 epoch 연산을 위한 반복문
    # - data: 각 배치로 나누어진 데이터와 정답
    for i, data in enumerate(trn_loader):
        # 데이터 처리
        cnn.train()     # 네트워크를 학습을 위한 모드로 설정

        # 학습 데이터(x: 입력, label: 정답)를 받아온 후, device에 올려줌
        x, label = data
        X = x.to(device)
        label = label.to(device)

        # 네트워크 연산 및 손실함수 계산
        model_output = cnn(x)   # 네트워크 연산 수행 후 출력값 도출
                                # 입력: x, 출력: model_output

        loss = criterion(model_output, label)   # 손실함수 값 계산
                                                # 네트워크 연산 결과와 실제 결과를
                                                # cross entropy 연산하여 손실함수 값 도출

        # 네트워크 업데이트
        optimizer.zero_grad()   # 학습 수행 전 미분값을 0으로 초기화
                                # 학습 전에 꼭 수행
        loss.backward()         # 가중치 W와 b에 대한 기울기 계산
        optimizer.step()       # 가중치와 편향 업데이트

        # 학습 정확도 및 손실함수 값 기록
        trn_acc = get_accuracy(model_output, label) # 네트워크의 연산 결과와 실제 정답 결과를 비교하여 정확도 도출
        trn_loss_list.append(loss.item())           # 손실함수 값을 trn_loss_list에 추가
                                                    # item: 하나의 값으로 된 tensor를 일반 값으로 바꿔줌
        trn_acc_list.append(trn_acc)                # 정확도 값을 trn_acc_list에 추가
        
        # 검증 수행
        # 학습 진행 상황 출력 및 검증셋 연산 수행
        if (i+1) % 100 == 0:    # 매 100번째 미니배치 연산마다 진행상황 출력
            cnn.eval()          # 네트워크를 검증 모드로 설정
            with torch.no_grad():   # 학습에 사용하지 않는 코드들은 해당 블록 내에 기입
                # 검증시 손실함수 값과 정확도를 저장하기 위한 리스
                val_loss_list = []
                val_acc_list = []

                # 검증셋에 대한 연산 수행
                for j, val in enumerate(val_loader):
                    val_x, val_label = val
                    
                    val_x = val_x.to(device)
                    val_label = val_label.to(device)

                    val_output = cnn(val_x)

                    val_loss = criterion(val_output, val_label)
                    val_acc = get_accuracy(val_output, val_label)

                    val_loss_list.append(val_loss.item())
                    val_acc_list.append(val_acc)
            
                # 학습 및 검증 과정에 대한 진행상황 출력
                print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | trn acc: {:.4f} | val acc: {:.4f}".format(
                    epoch+1, num_epochs, i+1, num_batches, np.mean(trn_loss_list), np.mean(val_loss_list), np.mean(trn_acc_list), np.mean(val_acc_list)))
{% endhighlight %}


## Deep Q Network(DQN)
### Q-Learning
### Deep Q Network