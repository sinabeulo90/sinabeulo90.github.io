---
layout: post
title:  "Model Free Control"
date:   2021-03-01 02:00:00 +0900
category: "Reinforcement Learning"
tags: YouTube
---

## 참고

- video
    - [[강화학습 5강] Model Free Control](https://youtu.be/2h-FD3e1YgQ)
- slide
	- [Lecture 5: Model-Free Control](https://www.davidsilver.uk/wp-content/uploads/2020/03/control.pdf)


## Use of Model-Free Control
- Model-free control이 쓰이는 곳
- 엘리베이터 알고리즘, 로봇 축구, 포토폴리오 관리 등
- MDP를 모르거나, 알고 있더라도 샘플링을 이용하는 것 이외에는 쓰기가 어려울 떄 사용한다.


## On and Off-Policy Learning
policy에는 최적화하고자 하는 policy와 behavior policy, 즉 environment에서 경험으로 쌓는 policy가 있는데, 이 때 이 두 policy가 같으면 on-policy, 다를 때 off-policy learning 이다.
- On-policy learning: 최적화하고자 하는 policy가 있고, 그 policy로부터 어떤 행동을 하고 경험을 쌓아서, 그 경험으로 배우는 방법
- Off-policy learning: 최적화하고자 싶은 policy가 아니라 다른 agent가 행동한 behavior policy로부터 어떤 행동을 해서 경험을 쌓았을 때, 그 경험으로부터 배우는 방법
	- 전혀 다른 policy를 사용


## Generalised Policy Iteration (Refresher)
- Policy iteration은 결국 최적의 policy를 찾는 control의 방법론
- 2단계를 번갈아 가면서 반복한다.
	- 1. policy evaluation: policy가 고정되어 있을 때, 그 policy를 평가하여 V_\pi를 측정
	- 2. policy improvement: 평가된 V_\pi를 바탕으로 더 좋은 policy 발전
		- Greedy policy improvement: 평가된 V_\pi를 바탕으로 greedy하게 움직이는 policy 생성


## Generalised Policy Iteration With Monte-Carlo Evaluation
- Policy evaluation: Monte-Carlo policy evaluation을 사용
	- MC는 model-free이어도 evaluation할 수 있음
- Policy improvement: Greedy policy improvement
- 이렇게 두면 Control 문제가 풀리는 게 아닐까?
- 안된다. 왜냐하면, MC로 V_\pi를 학습한다는 말은 현재 state가 있고, 다음 state가 무엇인지 알 때, 다음 state의 V 값 중에 제일 V 값이 큰 state로 가는 것인데, 다음 state를 안다는 것은 MDP를 안다는 것이다. MDP를 모를 때는, 다음 state가 뭐가 되는지는 가보지 않고는 모른다. 어떤 확률도 무엇이 될지도 아무것도 모르는데, 지금 V만 알고서는 greedy하게 policy를 improvement하게 학습할 수가 없다. 왜냐하면 model-free이기 떄문에, policy evaluation을 할 수 있는데, 다음 state가 뭔지를 모르기 때문에 greedy policy improvement할 수 없다. 내가 갈 수 있는 곳 중에서 제일 좋은 곳으로 가야하는데..


## Model-Free Policy Iteration Using Action-Value Function
- 모델을 알아야 V에 대한 greedy policy improvement 할 수 있다.
- 만약 state value function이 아닌, action value function을 policy evaluation 단계에서 평가를 했다고 한다면, Q에 대한 greedy policy imporvement는 가능할까?
	- V: state 마다 값이 하나씩 있는 것
	- Q: state 갯수 x action 갯수만큼 값이 있는 것
		- Q의 value는 MC 방법으로 알 수 있다. value function 자체는 Q, V 모두 구할 수 있다. 왜냐하면 state에서 action을 취할 때, 나온 return들의 평균을 취한 것이 Q이기 때문이다. Q(s, a)를 구하는 것은 어렵지 않다.
- Q에서는 greedy policy imporvement 할 수 있다. 왜냐하면 state에서 할수 있는 action이 몇 가지인지 알수 있기 떄문에, 할 수 있는 action들 중 Q 값이 제일 높은 action을 policy를 새로운 policy로 삼으면 되기 때문이다.


## Generalised Policy Iteration with Action-Value Function
- Policy evaluation: MC를 이용해서 Q를 평가
- Policy improvement: Greedy policy improvement 진행
	- Greedy policy improvement를 하려면, 모든 state를 충분히 봐야한다.
	- 그런데, 모든 state를 보지 않으면, 그냥 우리가 막힐 수 있다.(you can stuck)
	- Greedy 하게만 움직이면, 충분히 많은 곳을 가볼 수가 없게 된다.(exploration이 총분히 되지 않는다.)


## Example of Greedy Action Selection
- Greedy하게 action을 선택하는 것의 문제점 예
- 2개의 문 중 왼쪽 문을 열었더니 reward가 0을 받았다고 하자. 오른쪽 문을 열었을 때는 1을 받았다고 하면, 그 다음 문을 열게 될 때, 오른쪽 문을 열 것이다. 그리고 오른쪽 문을 열었더니 3을 받았다고 하자. 그렇게 greedy하게 오른쪽 문만 열게 된다.
- 그럼, 오른쪽 문이 제일 좋은 문이라는 확신이 있을까? 왼쪽 문을 한번 더 열때 100을 받을수도 있을 지 모른다.
- 따라서, 우리가 Greedy하게 행동을 선택하게 되면, 이런 문제가 생길 수 있다.


## \epsilon-Greedy Exploration
- policy improvement할 때, \epsilon의 확률(5%)로 랜덤하게 다른 행동을 선택한다. 95%의 확률로 제일 좋은 행동을 선택한다.
- 장점:
	- 모든 action에 대해 exploration을 보장할 수 있다.
	- policy가 발전함을 보장할 수 있다. 왜냐하면 1 - \epsilon일 확률일 떄는 제일 좋은 행동을 선택하기 때문이다.


## \epsilon-Greedy Policy Improvement
- \epsilon-greedy를 사용해도, improvement가 되는지에 대한 내용
- 이전 3강에서 greedy하게 value function을 선택하는것이 실제로 value function이 발전하게 되는 것을 증명하는 내용이 있던 것 처럼, 여기서는 greedy 자리에 \epsilon-greedy가 들어갔을 때 똑같이 value function이 개선되는 효과가 있는지를 보여주는 내용이다.
- m: action의 갯수
- q_\pi(s, \pi'(s)): 현재 action 선택은 \pi'으로 선택하고, 이후에는 \pi를 따라갔을 때의 value


## Monte-Carlo Policy Iteration
- Policy evaluation: MC를 이용해서 Q를 평가
- Policy improvement: Greedy policy improvement 대신 \epsilon-greedy policy improvement를 사용한다.
- 이렇게 하면 된는데, 조금 더 효율적으로 할 수 있는 방법은 없을까?


## Monte-Carlo Control
- MC로 policy evaluation을 할 때 수렴할 때까지 evaluation을 할수도 있겠지만, MC는 episode 단위로 평가가 진행되기 때문에, 딱 한 episode만 쌓고 그것으로 MC policy evaluaction을 한 뒤 바로 policy improvement 단계로 가기 떄문에, 화살표가 Q방향으로 끝까지 가지 않는다.
- policy evaluation 단계일 때, 끝까지 가지 않는 이유는 에피소드가 하나 끝나면 바로 한 에피소드 만큼의 경험을 갖고 policy를 개선할 수 있는데, 더 평가할 때까지 기다리지 않고 바로 policy improvement해도 되는 것 아닌가에 대한 아이디어이다.


## GLIE
- 좀 더 빠르게 수렴할 수 있기 위한 성질: GLIE의 property
- GLIE
	- 1. exploration에 관한 것: 모든 state-action pair들이 무한히 많이 반복되어야 충분히 explore할 수 있다.
		- \epsilon-greedy policy에서 만족
	- 2. exploitation에 관한 것: 예를 들어 \epsilon이 5%일 경우, 학습을 10만년을 하여 제일 좋은 policy를 찾더라도, 5%로 바보같은 행동을 할 것이다. 그러하면 그것은 최적의 policy가 아니다. 이는 \epsilon-greedy가 들어가면서 이런 상황이 생긴 것이지만, 결국에는 greedy-policy로 수렴을 해야 한다.
- 위 2개 조건을 만족시키기 위해, \epsilon을 1/k로 한다.
	- 그렇다면 \epsilon은 있으면서 그 값은 점점 줄어들 것이다.


## GLIE Monte-Carlo Control
- \epsilon = 1/k: GLIE의 조건을 만족시키는 Monte-Carlo Control
- evaluation 단계
	- N(S_t, A_t): S_t에서 A_t를 선택한 갯수
	- Q(S_t, A_t): 1/N(S_t, A_t) 대신 고정된 \alpha가 있을 수 있다.
- improvement 단계
	- \epsilon: 1/k
	- 위의 \epsilon을 사용한 \epsilon-greedy improvement
- GLIE Monte-Carlo control을 하면, optimal action-value function으로 수렴함이 정리되었다.


## Back to the Blackjack Example
- 4장에서는 어떤 policy가 주어졌을 때, value를 evaluation을 통해 학습
- 여기서는 control에 대한 예시


## Monte-Carlo Control in Blackjack
- optimal policy와 optimal value가 학습된 것을 나타냄
- Ace가 있을 때와 없을 때
- state: ace가 있는가 없는가 / 딜러가 어떤 카드를 보여주고 있는가 / 내 카드의 합이 몇인가(11~21)
- GLIE의 Monte-Carlo를 학습한 결과


## MC vs. TD Control
- Monte-Carlo를 쓰는 자리에 TD를 쓰면 되지 않을까? 가능하다.
- TD: variance가 작고, online 학습을 할 수 있고, 에피소드가 끝나지 않아도 학습할 수 있다.


## Updating Action-Value Functions with Sarsa
- 한 스텝 action을 한 뒤, 그 action이 끝날 때 R을 받고, Q(S, A) 자리에 업데이트
- \alpha: 얼만큼 업데이트할 지 나타냄
- TD error: R + \gammaQ(S', A') - Q(S, A)
	- TD target: R + \gammaQ(S', A')
	- \gammaQ(S', A'): 한 스텝을 더 가서, 거기서 예측하는 예측치
