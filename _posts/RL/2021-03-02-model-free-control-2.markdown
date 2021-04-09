---
layout: post
title:  "Model Free Control(2)"
date:   2021-03-02 01:00:00 +0900
category: "Reinforcement Learning"
tags: YouTube
---

## 참고

- video
    - [[강화학습 5강] Model Free Control](https://youtu.be/2h-FD3e1YgQ)
- slide
	- [Lecture 5: Model-Free Control](https://www.davidsilver.uk/wp-content/uploads/2020/03/control.pdf)


## On-Policy Control With Sarsa
- 이전 그림보다 좀 더 극단적인 형태
- Policy evaluation: Sarsa를 이용해서 Q를 평가
- Policy improvement: Greedy policy improvement 대신 \epsilon-greedy policy improvement를 사용한다.
- 한 에피소드마다 가는 것이 아닌, 한 step마다 진행
	- 더 짧은 간격으로 iteration을 수행한다.


## Sarsa Algorithm for On-Policy Control
1. Q(s, a)를 임의로 initialize
	- Q(s, a)는 lookup table
2. 처음 state S에서 \epsilon-greedy policy로 한 action을 선택한다.
	- Q(s, a) 중 가장 큰 값을 가지는 action을 선택하거나,
	- 낮은 확률로 랜덤하게 선택한다.
3. reward와 다음 state s'을 관측한다.
4. 그럼 s'에서 \epsilon-greedy policy로 한 action a'을 선택한다.
5. s', a'에서 얻은 예측치를 통해 업데이트한다.
6. 2를 반복한다.


## Convergence of Sarsa
- Sarsa는 optimal action-value function에 수렴한다.
- 조건
	1. GLIE를 만족해야 한다.
		- GLIE: 모든 state-action을 방문하면서, \epsilon이 줄어들어서 greedy policy로 수렴해야 한다.
	2. Robbins-Monro sequence: step size \alpha에 관한 얘기
		- \alpha: 얼만큼 업데이트 할지를 나타내는 값
		- \alpha를 1에서 무한히 더하면 무한해야 한다.
			- Q를 먼 곳으로 데려갈 수 있도록 step size가 충분히 크다는 의미
			- 만약 Q value의 실제 값이 1억인데, 0으로 initialize 했으면, 1억까지 업데이트를 할 수 있어야 한다는 의미
		- \alpha의 제곱을 1에서 무한히 더하면 무한보다 작다.
			- Q value를 수정하는 것이 점점 작아져 결국에는 수렴할 수 있어야 한다는 의미
		- 실질적으로 실제 쓸 때는, 이 2개 조건들을 고민하지 않아도 잘 수렴한다.
		- 이론적으로 수렴성을 만족하기 위해서는, 이 조건들을 만족해야 한다는 정도로 알아두면 될 듯 하다.


## Windy Gridworld Example
- 0: 바람x
- 1: 위로 한 칸 올려주는 바람
- 2: 위로 두 칸 올려주는 바람
- S에서 G까지 가는 최적의 길 찾기
- 한 step을 갈때마가 reward = -1
- discount factor을 사용하지 않음


## Sarsa on the Windy Gridworld
- x축: 총 업데이트한 횟수
- y축: G에 도달할 때마다 1씩 증가
- 2000번 움직여야 첫 에피소드가 걸린다.
	- 랜덤하게 너무 바보같이 움직이기 때문이다.
- 우연히 한번 도달하기 시작하면, reward 때문에 정보가 전파되기 시작하면서 그 다음번 에피소드를 진행할 때는 훨씬 적은 time step을 소모한다.
	- 한번 도달하는 순간 reward가 발생하고, 그 지식이 bootstrapping 되므로 전파되어 더 잘 찾아간다.
- Q 테이블의 칸 갯수: 행동의 갯수 4 x 가로 10 x 세로 7 = 280개


## n-Step Sarsa
- 지난 시간에 TD(0), TD(n), TD(\lambda), TD Forward/Backward-View 배운것을 여기에 적용할 수 있다.
- n-step TD: n까지 reward를 보고, 거기서 bootstrapping을 하는 것


## Forward View Sarsa(\lambda)
- TD(\lambda): 모든 stap에 대해서 geometric mean을 한 것
	- 1 step: 1 - \lambda
	- n step: (1 - \lambda)*\lambda^{n-1}


## Backward View Sarsa(\lambda)
- Eligibility traces 사용
	- 어떤 state를 방문할 때마다 2가지 관점에서 책임 사유를 묻는 것
		- 1. 가장 최근에 방문했는가?
		- 2. 여러번 방문했는가?
	- Q처럼 각 state-action pair하게 값을 갖고 있으며, 각 state-action을 방문하면 1 증가, 시간이 지날 때마다 감쇠시켜준다.
- E_t(s, a)가 크면 책임사유가 크다고 판단하고 더 많이 업데이트 되고, 적으면 더 적게 업데이트 된다.
- Forward View와 수학적으로 동일함이 증명됨


## Sarsa(\lambda) Algorithm
- 이전 Sarsa 알고리즘과 비슷한데, 중간에 For-loop이 추가되었다.
- 이전에는 한 state-action을 하면, 해당 칸만 업데이트만 해줬는데, 여기서는 한 state-action을 하면 Sarsa(\lambda)는 모든 s, 모든 a에 대해 모든 칸을 업데이트 한다.
	- 왜냐하면 과거에 지나갔던 칸도 Eligibility traces값이 있기 때문에, 한번 action을 하면 모든 칸들을 업데이트 한다.
	- 계산량이 좀 더 들지만, 대신 정보 전파는 빠르다.


## Sarsa(\lambda) Gridworld Example
- 중간 그림: one-step Sarsa
	- 목적지까지 진행하는 동안 -1의 reward를 받는다.
	- 목적지에 도착하는 순간 평소와는 다른 reward를 받고 종료한다.
	- 이때, 목적지까지 도착하는 순간의 것만 업데이트 한다.
	- 이전 state에서 위로 가는 것이 좋다는 것만 배운다.
- 오른쪽 그림: Sarsa(\lambda)
	- 지나온 모든 경로에 대해 eligibility traces값 만큼 비례해서 책임을 물으면서 업데이트 된다.
	- 직전 칸은 많이 업데이트, 과거로 갈 수록 점점 작은 업데이트


## Off-Policy Learning
- behavior policy \mu, target policy \pi
	- behavior policy \mu: 실제 action을 샘플링하는 policy
		- stochastic policy(확률분포가 있는 policy)에 주사위를 던져서 그 확률에 비례하는 action을 선택, 즉 샘플링한다.
- \mu와 \pi가 다르다.
- 예를 들어 target policy \pi를 evaluate하여 value function(state, state-action)을 구하고 싶은데, \pi를 따라서 움직여야 하는데, \pi가 아닌 \mu를 따라 움직여야 햐는 상황인 것
- 다른 agent의 행동을 관찰하는 것으로부터 배울 수 있음
- ex. 사람이 이렇게 한다고 했을 때, 그 사람이 이렇게 한 policy를 만드는 것이 아니라, 그 사람의 결과를 보고 따라하거나, 다른 행동을 해야겠다 등을 배우는 것
	- 사람은 일단 경험을 제공해 주었고, 그 경험안에서 최적을 배우는 것이다.
- 시뮬레이션은 못하지만, 기존에 수집한 어떤 데이터가 있는 상황에서 학습을 할 때 사용
- On-policy의 경우 한번 경험을 쌓고 난 뒤, 그 경험을 버린다. 왜냐하면 그 경험으로부터 policy를 업데이트하고 나면, policy가 조금 바뀌기 때문에 새로 업데이트된 policy를 가지고 새롭게 action을 해야 의미가 있는 경험이다.
	- 경험하는 쌓는 agent와 학습하는 agent가 같은 policy 여야 한다.
- 경험하는 쌓는 agent와 학습하는 agent가 달라도 된다.
	- 옛날에 했던 경험들, 혹은 다른 agent가 겪은 경험들을 재사용 가능하다.
	- 굉장히 효율적으로 경험을 이용할 수 있다.
- 나는 탐험적인 행동을 하면서, 동시에 optimal policy를 배울 수 있다.
	- exploration, exploitation이 서로 trade off 관계에 있는데, 이 것을 잘 조율할 수 있다.
	- 탐함적인 행동은 하면서도 안전하게 최적의 policy를 배워나갈 수 있다.
- 한 policy를 따르면서, 여러개의 policy를 학습할 수 있다.


## Importance Sampling
- X~P: 확률분포 P에서 샘플링 되는 어떤 X(state)들
	- ex. 1 ~ 6: 주사위의 눈, P = 1/6
- X~Q
	- ex. 6: 50%, 1~5: 10%, 비뚤어진 주사위 Q
- f(X): 어떤 뭐가 될수 있는 함수
	- ex. X=3
		- f(X)=3, f(X)=6, f(X)=9, f(X)=12, ...
- E_{X~P}[f(X)]: 확률분포 P를 따르는 어떤 X에 대해, f(X)의 기댓값
- 하고 싶은 것은 P 주사위를 이용해서 f(X)의 기댓값을 구하고 싶은데, P를 이용해서 던져 놓고선, 다른 주사위 Q를 이용했을 때의 기댓값을 구하고 싶은 것
	- E_{X~P}[f(X)]는 확률분포 Q에서 샘플링 되는 어떤 X에 대해 P(X)/Q(X)*f(X)의 기댓값과 같다.
	- 다른 분포로 구하고 싶다면, 첫번째 분포와 두번째 분포의 비율을 곱해주면 된다.


## Importance Sampling for Off-Policy Monte-Carlo
- MC: policy \mu를 이용해서 게임이 끝나면, return G_t를 받게 되는데 여기에 G_t를 그냥 쓰는 것이 아니라, G_t를 얻기까지 나왔던 모든 action의 확률의 비율(\mu와 \pi의 비율)을 계속 곱해주면, 다른 policy를 따랐음에서 policy \pi를 따랐을 때의 return을 구할 수 있다.
- target 자리에 G_t^{\pi/\mu}가 들어가게 되어, 정확하게 업데이트 할 수 있다.
- 주사위를 한번만 던지면 P/Q를 한번만 곱해지는데, 실제 게임이 끝날때까지 주사위를 계속 던지므로, 각 던질때 마다 교정 term이 들어가서 action수 만큼 교정 term이 들어간다.
- 하지만 이 방법은 상상속에나 존재하는 방법론이다. 이 방법론은 쓸 수 없다. 왜냐하면 이 교정 term의 variance가 극도록 크기 때문에, 도저히 현실에서 쓸 수 없는 방법론이다.
	- 교정 term이 계속 곱해지기 때문에, 값이 폭발하거나, 매우 작게 수렴될 것이기 때문이다.
	- \mu가 0일 경우 쓸 수 없다.


## Importance Sampling for Off-Policy TD
- MC 대신 TD에 적용한다.
- TD는 한 스텝마다 업데이트하기 때문에, 앞에 곱해지는 importance sampling ratio가 1개 밖에 없다.
	- action 하나 한것에 대해서 target policy의 확률과 behavior policy의 확률을 나눈 것을 곱해준다.
- 이 방법은 variance가 훨씬 적기 때문에 가능하다.


## Q-Learning
- Atari 게임에 썼던 방법론
- importance sampling을 쓰지않고, Action-value 인 Q를 off-policy로 학습하고 싶다.
- behavior policy를 이용해서 action을 하나 뽑아 실제 action을 하고, S_{t+1}에 도착한다. TD target에서 S_{t+1}의 Q value가 쓰이고, R_{t+1} + \gamma Q(S_{t+1}, A')의 A' 자리에 behavior policy의 action이 아닌, target policy의 action을 넣어준다.
	-  behavior policy를 통해 얻은 S_{t+1}의 Q value로 target policy를 따를 때의 action a'에 대한 Q(S_{t+1}, a')을 사용한다.
	- 원래 우리가 알던 식과 동일하고, 굉장히 말이 되는 업데이트이다.
	- S에서 A를 할 때의 value를 한 스텝 가보고 그 스텝에서 추측하는 것으로 업데이트를 하는 것이 TD learning이었는데, 그 추측치가 behavior policy를 안써도 되니까 이 식이 성립한다.
- A_{t+1}을 \mu로 골라서 선택했고, A_t도 \mu를 골라서 선택했다. S_t가 있을 때 \mu를 통해 A_t를 골라서 S_{t+1}이동했기 때문에 이에 대한 action-value를 구하고 싶은데 이 때, S_{t+1}에 대한 action a'은 behavior policy가 아닌 target policy를 통한 행동을 사용한다.


## Off-Policy Control with Q-Learning
- behavior policy를 하나 정해야 하는데, taget policy 처럼 behavior policy도 improvement 되었으면 좋겠다. 그러면서 behavior policy는 탐험적인 행동을 했으면 좋겠다.
- 따라서 target policy는 무조건 잘하면 되기 때문에 greedy Q로 정하고, behavior policy는 다양하게 해주면 되기 때문에 \epsilon-greedy Q로 정한다.
	- 이렇게 정한 것이 Q-learning으로 더 많이 쓰인다.
	- 이렇게 하면 behavior policy도 점점 좋아진다.


## Q-Learning Control Algorith
- Sarsa는 실제 policy의 actio에 대해 한 개 S_{t+1}을 사용하기 때문에 밑에 점이 하나 있었지만, Q-learning은 할 수 있는 action 중에 max 값을 선택하기 때문에 호가 그려져 있다.
- Q-learning control은 optimal action-value function에 수렴함이 정리됬다.
- Sarsa max라고도 불림


## Relationship Between DP and TD
