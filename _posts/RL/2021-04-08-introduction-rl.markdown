---
layout: post
title:  "Lecture 1: Introduction to Reinforcement Learning"
date:   2021-04-08 18:05:14 +0900
category: "Reinforcement Learning"
tags: YouTube
plugins: mathjax
---

David Silver 님의 [Introduction to reinforcement learning](https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) 강의 내용을 [팡요랩 Pang-Yo Lab](https://www.youtube.com/channel/UCwkGvF7xKz2E0Lv-fZ9wv2g)의 [강화학습의 기초 이론](https://youtube.com/playlist?list=PLpRS2w0xWHTcTZyyX8LMmtbcMXpd3s4TU)에서 풀어서 설명하기에, 이를 바탕으로 공부하여 정리한 내용입니다.


---

## About RL


### Branches of Machine Learning

기계학습(Machine Learning)의 유형
- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning


### Characteristics of Reinforcement Learning

강화학습이 다른 기계학습 패러다임과 다른 점

1. No supervisor: 임의의 상황에 대해 정답을 알려주는 사람이 없다.
2. Only a reward signal: 오직 Reward 신호만을 사용해서 좋은 방법론을 찾아 간다.
	- Reward를 주는 행위가 지도학습에서 정답을 알려주는 행위와 비슷하게 느껴질 수도 있다.
	- 하지만, Reward는 어떤 목적을 위해 설정해 놓은 것일 뿐, 구체적으로 어떻게 행동을 해야 한다는 것을 알려주지는 않는다.
	- Supervised Learning은 어떤 상황에서 어떤 행위를 해야하는지 알려주기 때문에, 이는 강화학습과 약간의 차이가 있다.
	- Reward만 설정해준다면, 오히려 Supervisor가 정답으로 알고있는 방법들보다 더 나은 방법을 찾아낼 수도 있을 것이다.
3. *Feedback is delayed, not instantaneous: 피드백이 즉시 주어지지 않고, 지연되서 주어진다. 이 부분이 강화학습을 어렵게 만든다.*{: style="color: red"}
4. Time really matters(sequential, non [i.i.d](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) data): 순서가 있는 데이터이기 때문에, 시간이 중요하다.
5. Agent's actions affect the subsequent data it receives: 어떤 행동을 하느냐에 따라 얻게되는 데이터가 그때그때 달라진다.


---


## The RL Problem


### Reward


#### Rewards

- Scalar feedback signal: 피드백에 대한 숫자 한 개
- 매 $t$ 번째 시간 마다 Agent의 행동에 대한 보상이 주어진다.
- Agent의 목적은 축적된 보상의 합이 최대가 되도록 하는 것이다.
- Reward Hypothesis
	- 모든 목적은 축적된 보상이 극대화 되는 것으로 설명할 수 있다.
	- 강화학습은 reward hypothesis에 기초로 돈다.
- 만약 여러가지 목적을 위해 reward를 여러 개 주고 싶을때, 각각의 reward에 대해 가중치를 주어 하나의 reward로 표현한다면 강화학습에 적용할 수 있을 것이다.


#### Sequential Decision Making

- 목표: 미래 보상의 총 합을 극대화하는 행동을 선택하는 것
- 행동들은 장기적은 결과를 만들어 낼 것이다.
- 보상은 지연될 것이다.
- 지금 당장은 손해보더라도, 장기적으로 더 큰 보상을 얻으려는 것이 좋을 것이다.


<br>


### Environments


#### Agent and Environment

- Agent: 뇌
- Environment: Agent 외의 모든 것
- Action: Agent가 선택한 행동

- Agent가 action을 하면, environment는 observation과 reward를 준다.
- Agent는 observation과 reward를 받고, action을 한다.
- 다시 environment는 action을 받고, observation과 reward를 준다.
- Agent와 environment는 서로 상호작용한다.
	- Agent: Step $t$일 때, observation $O_t$와 reward $R_t$를 받고, action $A_t$를 한다.
	- Environment: action $A_t$를 받으면, observation $O_{t+1}$와 reward $R_{t+1}$을 준다.


<br>


### State


#### History and State

- History: observations, actions, rewards의 나열
	- $H_t = O_1, R_1, A_1, \dots, A_{t-1}, O_t, R_t$
	- Agent: History를 보고 action을 정한다.
	- Environment: History를 보고, observation과 reward를 정한다.
- State: 다음에 무엇을 할지 결정하기 위해 사용하는 정보
	- Agnet가 어떤 행동을 결정할 때, state에 근거해서 선택한다.
	- Environment가 observation과 reward를 계산할 때, state에 근거해서 선택한다.
	- *결국 state는 history에 대한 함수이다.*{: style="color: red"}
	- $S_t = f(H_t)$


#### Environment State

- Environment state $S^e_t$: Observation과 reward를 계산하기 위해 쓰이는 모든 정보들
- Agent는 이 정보들을 볼 수 없고, 만약 볼 수 있다하더라도, 전혀 관련 없는 정보가 포함되어 있을 수 있다.
- Ex: Atari 게임에서, 화면을 observation으로, 점수를 reward로 제공된다고 하자. Agent는 이 2개의 정보를 받아 action을 선택한다. 그럼 이제 내부적인 계산을 통해 다음 observation을 만든텐데, 이 내부적인 계산에 사용되는 값들이 environment state로 생각하면 된다.


#### Agent State

- Agent state $S^a_t$: Agent가 다음 행동을 선택하기 위해 사용하는 정보들(History)
- $S^a_t = f(H_t)$


#### Information State(Markov State)
{: style="color: red"}

- Markov state: 어떤 결정을 할 때, 바로 이전 상태만 의존하면 되는 정보
- Ex: 헬리콥터의 현재 위치, 각도, 각/가속도, 바람 세기 등의 정보가 주어졌을 때, 이 헬리콥터가 균형을 유지하기 위해서 오직 바로 지금의 상태만을 알면 될 때, 이 상태를 Markov하다고 말한다.
- *다음 state를 결정하기 위해 오직 현재 state만을 사용하고, 이전 state들은 사용하지 않는다.*{: style="color: red"}
	- $\mathbb{P}[S_{t+1} \| S_t] = \mathbb{P}[S_{t+1} \| S_1, \dots, S_t]$
	- $\mathbb{P}[S_{t+1} \| S_1, \dots, S_t]$: $S_1, \dots, S_t$가 모두 주어졌을 때, $S_{t+1}$로 갈 확률
- $H_{1:t}$가 $S_t$를 결정하고, $S_t$가 $H_{t+1:\infty}$을 결정한다.
	- $H_{1:t} \rightarrow S_t \rightarrow H_{t+1:\infty}$
- *Agent state가 markov state가 아닌 경우*{: style="color: red"
}: 도로에서 현재 위치, 현재 시간, 현재 조향각을 알고 있을 때, 1틱이 지나고 엑셀을 완전히 밟으면, 다음 틱에 이 차의 속도가 얼마나 될지 알 수 있을까? 현재 속도를 모른다면, 현재 위치와 이전 틱의 위치를 알면 1초 동안 얼마나 움직였는지 알아내야 다음 속도를 알 수 있다. 이렇게 이전 틱을 알아야만 할 경우, 이 상태는 Markov하지 않다고 할 것이다. 만약 현재 속도를 포함해서 정의된다면 Markov하다고 할 수 있다.


#### Fully Observable Environments

- Agent가 직접 environment state를 관찰할 수 있다.
- Observation state = Abent state = Environment state
- Markov decision process


#### Partially Observable Environments

- Agent가 간접적으로 environment state를 관찰할 수 있다.
- Agent state ≠ environment state
- Partially observable Markov decision process(POMDP)
- Agent는 state의 표현형을 구축해야 한다.
	- Ex: History, Recurrent neural network


---


## Inside An RL Agent

### Major Components of an RL Agent

- Agent의 구성 요소: 3가지를 가질 수 있고, 1가지만 가질 수도 있다.
	- Policy: Agent의 행동을 정한다.
		- State를 넣으면, action을 준다. (mapping)
		- Deterministic policy: State를 넣으면, action 1개가 결정되어 반환한다.
			- $a = \pi(s)$
		- Stochastic policy: State를 넣으면, 각 action별로 확률을 반환한다. 
			- $\pi(a\|s) = \mathbb{P}[A_t = a \| S_t = s]$
	- Value function: 상황이 얼마나 좋은지를 나타낸다.
		- 게임이 끝날 때까지 받을 수 있는 미래의 reward 총합을 예측한다.
		- 현재 state에서 어떤 정책 $\pi$를 따랐을 때, 게임이 끝날 때까지 얻을 reward 총합의 기댓값
		- 기댓값이라고 표현하는 이유는 Stochastic policy일 경우도 있지만, Deterministic policy라고 하더라고 environment 자체에 확률적인 요소가 있기 때문이다.
		- $v_\pi(s) = \mathbb{E_\pi} [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \| S_t = s]$
			- Discount factor $\gamma$: 미래의 reward는 불확실할 수 있고, 또한 미래의 가치는 현재의 가치보다 적기 때문에 조금씩 줄여서 예측하기 위해 붙여준다.
	- Model: 환경이 어떻게 될 것인지를 예측한다.
		- 환경을 정확히 알기 어려울 때, agent가 환경이 어떻게 될 것인지 예측하기 위해 모델링한다.
		- Reward와 state transition 예측
			- Transition model: $\mathcal{P}^a_{ss'} = \mathbb{P}[S_{t+1} = s' \| S_t = s, A_t = a]$
			- Reward model: $\mathcal{R}^a_s = \mathbb{E}[R_{t+1} \| S_t = s, A_t = a]$
		- 이 모델링 기법은 쓰일 수도(model-based) 있고, 쓰이지 않을 수도(model-free) 있다.


### Categorizing RL agents(1)

- Value Based
	- Value function만 있으면 어떻게 이동해야 할지 알 수 있다.
	- 어느 위치가 좋은지 알면, 제일 좋은 위치로 이동할 수 있기 때문이다.
- Policy Based
	- Policy만 있으면 어떻게 이동해야 할지 알 수 있다.
- *Actor Critic*{: style="color: red"}
	- Policy와 value function을 사용해서 이동한다.


### Categorizing RL agents(2)

- Model Free
	- 내부적으로 모델을 만들지 않고, 오직 policy 또는 value function만을 사용해서 이동한다.
- Model Based
	- 내부적으로 환경에 대한 모델을 만들고, 그것에 근거해서 움직인다.
Categorizing RL agents(1)

- 위/아래 분류를 조합하여 총 6개 종류의 agent를 만들 수 있다.


---


## Problems within RL


### Learning and Planning

- Sequential decision making의 두 가지 근본적인 문제
	- Reinforcement Learning
		- 환경을 모른채로, 상호작용을 통해 policy를 개선한다.
		- Ex: Atari 게임의 규칙을 모른채로 계속 점수를 높이도록 학습한다.
	- Planning(Search)
		- 환경을 알고 있을 때, 즉 reward와 transition을 알고 있을 때, agent가 실제 environment에서 행동하지 않고, 내부적으로 계산을 통해 policy를 개선한다.
		- Monte Carlo tree search, Tree search
		- Ex: Emulator에 쿼리를 보내면서 다음 상황과 reward를 알아내어 최적의 policy를 찾아낸다.


### Exploration and Exploitation

- Exploration: Environment에 대해 더 많은 정보를 모으는 것
	- 이전과 다른 행동을 선택하여, 이전보다 더 좋은 곳을 발견하기 위해 탐험한다.
	- 장기적으로 좋을 수는 있지만, 지금 당장은 어떻게 될지 모른다.
- Exploitation: 지금까지 얻은 정보를 바탕으로 최선의 선택을 내리는 것
- 이 둘은 서로 trade-off 관계에 있다.


### Prediction and Control

- Prediction
	- Policy가 주어졌을 때, 미래를 평가한다.
	- Value function을 잘 학습시키는 것을 의미한다.
- Control
	- 미래를 최적화한다.
	- 최적의 policy를 찾는 것을 의미한다.


---

## References
- Video: [[강화학습 1강] 강화학습 introduction](https://youtu.be/wYgyiCEkwC8?t=3494)
- Slide: [Lecture 1: Introduction to Reinforcement Learning](https://www.davidsilver.uk/wp-content/uploads/2020/03/intro_RL.pdf)
