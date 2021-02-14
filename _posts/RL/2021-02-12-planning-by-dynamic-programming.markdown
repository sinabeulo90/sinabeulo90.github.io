---
layout: post
title:  "Planning by dynamic programming"
date:   2021-02-12 00:00:00 +0900
category: "Reinforcement Learning"
tags: YouTube
---

## 참고

- video
    - [[강화학습 3강] Planning by Dynamic Programming](https://youtu.be/rrTxOkbHj-M)
- slide



## Planning by dynamic programming
- planning: MDP에 대한 모든 정보를 알고 있을 때, MDP안에서 최적의 policy를 찾는 것


## Outline
- Policy Evaluation (Prediction 문제)
	- policy가 정해질 때, MDP안에서의 value function 값을 찾는 것
	- ex. 미로 찾기에서 현재 위치에서 마지막 까지 평균적으로 얼만큼의 value function을 가지는 지 찾는 것
	- Policy를 평가
- Policy Iteration, Value Iteration (Control 문제)
	- 어떤 iterative한 방법론을 통해서 최적의 Policy를 찾아가는 2가지 방법


## What is Dynamic Programming?
- 복잡한 문제를 푸는 어떤 방법론
- 큰 문제가 있으면, sub problem으로 나누고, 작은 문제에 대해서 답을 찾고, 그 답을 모아서 큰 문제를 푸는 일반적인 방법론
- 강화학습은 매우 큰 문제이고, 그 안에 model-free, model-based로 나뉜다.
	- model-free: environment가 어떤 정보를 던져줄 지 모를때(완전한 정보가 없을 때)
	- model-based: environment에 대한 모델이 있음, 내가 어떤 행동을 하면 어떤 확률로 어떤 state에 있게 되는지를 알고 있다. 여기서 planning을 풀기 위해 dynamic programming이 쓰인다.


## Requirements for Dynamic Programming
- Dynamic programming을 쓰기 위해 2가지 조건이 필요
- 1. Optimal substructure: 전체 큰 문제에 대한 optimal solution이 작은 문제들로 나뉠 수 있어야 한다.
- 2. Overlapping subproblems: 한 subproblem을 풀고 그 값을 저장해 둔 뒤에, 재사용 할 수 있어야 한다.
	- ex. 길찾기: A부터 B로 가기 위해, A에서 중간 지점 M까지, M부터 B까지 가는 최적의 경로를 알고 있다면, A에서 B까지 가는 최적의 경로를 풀 수 있을 것이다.
- Markov Decision Process는 위의 2개 조건을 모두 만족하기 때문에, Dynamic programming을 적용할 수 있다.
	- Bellman equation이 recursive하게 엮여있는 형태가 subproblem으로 나뉘는 것에 해당
	- 작은 문제들에 대한 해에 해당하는 value function을 저장하고, 재사용할 수 있다.


## Planning by Dynamic Programming
- Dynamic Programming은 MDP에 대한 모든 지식을 알고 있다고 가정한다. State transition probability, reward 등
- planning: MDP를 푸는 것
	- Frediction
		- MDP와 policy, or MRP에서 value function을 찾는 것
		- 최적의 policy와는 관계 없이. 어떤 바보같은 policy일지라도, MDP안에서 어떤 state에서부터 끝날때까지 얼만큼의 return을 받을지 예측
	- Control: MDP에서 optiaml value function, optimal policy를 찾는 것
 


-----

## Iterative Policy Evaluation
- Policy Evaluation(prediction): 지금 policy를 따랐을 때의 value function이 어떻게 되는지를 찾는 문제
- bellman expectaion equation을 이용해서 계속 반복 적용해서 구한다.
- 처음에 random한 v를 설정
	- ex. 모든 state의 value function을 0으로 초기화
	- 1번 iterative한 방법을 사용해서 v_2를 만들고, v_2에서 v_3를 만들어가면 v_\pi에 수렴한다. v_\pi는 \pi에 대한 value function으로, 궁극적으로 학습하고 싶은 것
	- 점화식과는 조금 다르게 거친다.
- backup: cache와 비슷하게 메모리에 저장해 두는 것
	- synchronous backup: 모든 state에 대해서 매 iteration마다 업데이트를 한다.(full sweep), 현재	 state에 있는 value를 다음 state에 대한 value 값을 이용해서 조금씩 더 정확하게 만든다.
	- asynchronous backup: 5단원에서 나옴


## Iterative Policy Evaluation(2)
- Bellman expectation equation
- 한 state의 value 값을 계산할 때, 다음 state의 정확하지 않은 value 값을 이용해서 업데이트한다. 정확한 reward가 있기 떄문에 점점 정확한 정보가 들어가게 되면서, 수렴될 때 까지 계속 반복한다.


## Evaluating a Random Policy in the Small Gridworld
- Prediction 문제이기 때문에, policy가 주어져야하고, MDP가 주어져야 한다.
- random policy: 1/4 확률로 행동을 선택
- 2번 state에서 얼마의 확률로 에피소드가 종료될까? 알기 매우 어려운 문제이다.
- reward가 -1일 때는 각 state의 value는 뭐가 될까?


## Iterative Policy Evaluation in Small Gridworld
- 처음에 0으로 초기화, 이론적으로 어떻게 초기화를 하던, true value function에 도달
- 우리는 바보같은 policy를 평가했을 뿐인데, 그 과정에서 평가한 value function을 가지고 greedy하게 움직이면(다음 state들 중 가장 좋은 state로 움직이면), optimal policy를 얻게 된다.
- 즉 모든 문제들에 대해서 greedy하게만 움직이는 policy를 찾기만 해도 더 나은 policy를 찾을 수 있다.


## Iterative Policy Evaluation in Small Gridworld(2)
- evaluation하는 과정에서 optimal policy를 찾아버렸고, 심지어 3번 iteration만에 찾아버렸다. 


## How to Improve a Policy
- policy iteration: 평가하고, greedy하게 움직이는 policy를 찾고, 그 policy를 평가하고 그에 대한 greedy한 policy를 찾으면, 최적의 policy를 찾을 수 있다.
- 1. Evaluate the policy \pi: 해당 policy에 대한 value function을 찾음
- 2. 그 value function에 대해 greedy 하게 움직이는 새로운 policy를 찾는다. (1의 policy보다 더 나은 policy)
- 1과 2를 반복하면, 작은 Gridworld 문제에서는 금방 optimal policy를 찾아내지만, 일반적으로믄 1, 2에 대해 좀더 많은 iteration을 거쳐야 한다. 하지만 이 policy iteration은 항상 optimal policy에 수렴한다.


## Policy Iteration
- 처음 바보같은 value function v, policy \pi가 있으면, 처음에 evaluate를 하고, 그에 대한 greedy하게 움직이는 \pi를 찾고, 다시 그 \pi에 대해 evaluation하고, 다시 greedy한 \pi를 찾고 이 과정을 반복한다.


## Jack's Car Rental
- 두 렌트카 장소가 있고, 한 장소에는 최대 20대의 차가 있을 수 있다.
- A 지점에는 포아송 분포에 따라 고객들이 온다.
	- 포아송 분포: 정해진 단위 시간동안 사건이 발생할 확률 분포
- A 지점: 하루에 3번 렌트 요청이 오고, 3번 반납
- B 지점: 하루에 4번 렌트 요청이 오고, 2번 반납
- 밤 중에 A 지점에서 B 지점으로, 혹은 B 지점에서 A 지점으로 계속 차를 옮겨야 한다.
- 하나를 빌려줄 때 마다 10달러의 수익을 얻는다.
- 이때 수익을 최대화 하기 위해서, B지점에서 수요가 더 많기 떄문에 A 지점의 차가 좀 더 적어도 B로 차량을 옮기는 것이 더 나을 수 있을 것이다.


## Policy Iteration in Jack's Car Rental
- x축은 B지점에 있는 차의 수, y축은 A지점에 있는 차의 수
- +5: 각 state에 대해서 A지점에서 B지점으로 5대의 차량을 옮기는 policy
- 마지막 그림은 evaluation을 할 때의 가치(420, 612: accumulate sum을 한 reward. 즉 value function)
- 등고선은 greedy한 policy를 표현


## Policy Improvement
- 증명: evaluation한 value function에 대해 greedy하게 움직이는 policy(policy improvement)가 무조건 이전보다 더 나은 policy가 되는가? 나아진다! 라는 증명
- 
