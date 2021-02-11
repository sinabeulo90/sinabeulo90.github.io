---
layout: post
title:  "Introduction to Reinforcement Learning"
date:   2021-02-10 00:00:00 +0900
category: "Reinforcement Learning"
tags: YouTube
---

## 참고

- video
    - [[강화학습 1강] 강화학습 introduction](https://youtu.be/wYgyiCEkwC8?t=3494)
    - [RL Course by David Silver - Lecture 1: Introduction to Reinforcement Learning](https://youtu.be/2pWv7GOvuf0)
- slide
    - [Lecture 1: Introduction to Reinforcement Learning](https://www.davidsilver.uk/wp-content/uploads/2020/03/intro_RL.pdf)


## 내용 정리중

기계학습:

- supervised Learning
- unsupervised Learning
- Reinforcement Learning

강화학습이 다른 머신러닝 패러다임과 다른 점

- No superviseor: 임의의 상황에 대해 정답을 알려주는 사람이 없다
- only a reward signal: 오직 Reward 신호만을 사용해서 좋은 방법론을 찾아 간다.
- Reward 를 주는 행위가 지도학습을 알려주는 행위와 비슷하게 느껴진다. Reward는 어떤 목적을 위해서 설정해 놓은 것이지만, 구체적으로 어떻게 해야 하는 방법들은 알려주지 않는다. Supervisor 방법론은 어떤 상황에서 어떤 행위를 해야만 하는지를 알려줘야 하기 때문에 약간의 차이가 생긴다. Reward만 설정해준다면 오히려 Supervisor가 알고있는 방법들보다 더 나은 방법을 찾아낼 수도 있을 것이다.
- Feedback is delayed, not instaneous: 피드백이 즉각적으로 주어지지 않고, 지연되서 주어진다. 이 부분이 강화학습을 어렵게 만든다.
- Time really matters: 시간이 중요하다. 순서가 있는 데이터이다.
- Agent's actions affect the subsequent data it receives: 어떤 행동을 하느냐에 따라 그때그때 데이터가 달라진다. 


reward

- scalar feedback signal: 숫자 한개
-  t번째 시간에 보상이 주어짐
- 축적된 모든 보상을 최대화 하는 것이 agent의 목적
- Reward Hypothesis: 모든 목적은 축적된 보상이 극대화 되는 것으로 설명할 수 있다.
- 여러가지의 목적이 있기 때문에 reward를 여러개 주고 싶다면, 각각의 reward에 가중치를 줌으로써 하나의 reward로 표현할 수 있다면, 강화학습으로 적용할 수 있을 것이다.

Sequential Decision Making

- Goal: 미래의 보상의 총합을 극대화하는 행동을 선택하는 것
- 행동들은 장기적인 결과를 만들어 낼 수 있다.
- 보상은 지연될 것이다.
- 초반의 보상을 희생하더라도 더 장기적으로 더 큰 보상을 얻으려는 것이 좋을 것
- 


Environment

- Agent: 뇌
- Environment: Agent외의 모든 것
- Action: agent가 선택한 행동
- agent가 action을 하면 environment는 reward, observation을 준다.
- Agent는 observation, reward를 받고, action을 실행한다.
- environment는 action을 받고, observation, reward를 제공한다.
- 서로 상호작용한다.


History and State

- History: observation, action, rewards의 나열
- Agent: History를 보고 action을 정한다
- environment는 History를 보고 observation, reward를 정한다.
- state: 다음에 무엇을 할지 결정하기위해서 쓰이는 정보
	- agent가 어떤 행동을 할 때, state를 근거로 해서 선택한다.
	- environement가 observation, reward를 계산하기 위해 쓰이는 state를 근거로 해서 선택한다.
	- 결국 state는 history에 대한의 함수이다. 


Environment State

- observation, reward를 계산하기 위해 쓰이는 모든 정보들
- agent는 볼수 없고, 가령 알수 있더라도, 전혀 관련 없는 정보가 포함되어 있을 수 있다.
- 예를 들어 Atari 게임에서 화면을 observation으로써, 점수를 reward로써 제공되면, 2개 정보를 받음으로써 agent는 action을 선택한다. 이때 다음 observation을 표현하기 위해서, 내부적인 계산을 통해서 observation을 만들텐데, 이때의 내부적인 계산에 사용되는 숫자들을 environmenet state라고 생각하면 된다.

agent state

- 내가 다음 행동을 선택하기 위해 사용하는 정보들(history)

Information State

- Markov state: 내가 결정을 할 때, 바로 이전의 상태만 의존하면 되는 정보
- ex: 헬리콥터의 현재 위치, 각도, 각속도, 가속도, 바람의 세기 등이 주어졌을 때, 이 헬리콥터가 균형을 유지하기 위해서 오로지 바로 지금의 상태를 알면 될 때, 이때 이 상태는 Markov 하다고 한다.
- 이전 틱의 State는 관심이 없고, 다음 틱의 state는 오로지 지금 state에 의해서만 결정
	- P[S_{t+1}|S_t]: S_t가 S_{t+1}로 될 확률
	- P[S_{t+1}|S_1, ..., S_t] : S_1, ... S_{t+1} 이 모두 주어졌을 때, S_{t+1}로 갈 확률
- S_t가 있으면, H_{1:t}는 버려도 되고, S_t가 H_{t+1:\inf} 를 결정한다.
- agent state가 markov state가 아닌 예시: 도로에서 현재 위치, 현재 시간, 현재 조향각을 알고 있을 때, 1틱이 지나고 엑셀을 풀로 밟으면, 다음 틱에 이 차의 속도가 얼마나 될지를 알 수 있을까? 만약 현재 차의 속도를 모르면 알수 없겠지만, 알고 있으면, 알수 있을 것이다. 현재 내가 속도를 모르니까, 현재 위치와 이전 틱의 위치 시간을 알면 1초동안 얼마나 움직였는지 알고 다음 속도를 알수 있을 텐데, 이렇게 이전 틱을 알아야만 할 경우는 이 상태가 Markov하지 않다고 할 것이다.
  Markov하게 정하려면, 현재의 속도를 포함해서 정의한다면 Markov하다고 할 것이다.


Fully Observable Environments

- agent가 직접 environment state를 관찰할 수 있음
- Observation state = Agent state = Environment state
- Markov decision process

Paritally Obervable Environments

- agent가 간접적으로 environment state를 관찰할 수 있음
- Agent state ≠ environment state
- partially observable Markov decision process(POMDP)
- 이때 agent는 state의 표현형을 구축해야 한다.
	- History
	- Recurrent neural network

Major Components of an RL Agent

- Agent의 구성 요소: 3가지를 가질 수도 있고, 1가지만 가질 수도 있다.
	- Policy: agent의 행동을 규정
		- state를 넣어주면, action을 뱉는다.(mapping)
		- Deterministic policy: 상태를 넣으면 action 1개가 결정적으로 반환
		- Stochastic policy: 상태를 넣으면 각 action별로 확률을 반환
	- Value: 상황이 얼마나 좋은지를 나타냄
		- ex. 게임이 끝날 때 까지 받을 수 있는 총 미래의 reward의 합산을 예측
		- 현재 state에서 **어떤 정책 \pi를 따라갔을 때**, 게임이 끝날때 까지 얻을 총 reward의 기댓값
		  --> policy가 확률적일 수 있고, environment에 확률적인 요소가 있을 수 있으므로 꼭 기댓값이 있다고 볼수 있다.
		- \gamma: discount factor, 미래의 보상은 불확실할 수 있고, 미래의 가치는 현재의 가치보다 적기 때문에 조금씩 줄여서 예측한다.
	- Model: 환경이 어떻게 될 것인지 예측
		- 환경을 정확히 알기 어려울 때, agent가 환경이 어떻게 될 것인지 예측하기 위해 모델링을 한다.
		- 1. reward 예측, 2. state transition 예측
		- 이 모델링 기법은 쓰일 수 있고(model-based), 안쓰일 수(model-free) 있다.

ETC

transition model P^a_{ss'}: s에서 a를 선택할때 s'으로 갈 확률
reward model R^a_s : s에서 a를 선택할때 얻는 reward 값

Categorizing RL agents(1)

- Value Based
	- value function만 있어도 agent 역할을 할 수 있다.
	- 어느 위치가 좋은지 알면, 제일 좋은 위치로 이동할 수 있기 때문이다.
- Policy Based
	- Policy만 있으면, 어떻게 이동할 지 알 수 있다.
- Actor Critic
	- Policy와 value function을 사용해서 이동한다.

Categorizing RL agents(2)

- Model Free
	- 내부적으로 모델을 만들지 않고, policy 또는 value function만 을 사용해서 이동한다.
- Model based
	- 내부적으로 환경에 대한 모델을 만들고, 그것에 근거해서 움직인다.

총 6개 종류의 agent가 생긴다.

Learning and Planning

- Reinforcement Learning
	- 환경을 모른채로, 환경과 interaction을 통해 policy를 개선
	- Atari: 게임의 규칙을 모른채로, 계속 점수를 높이도록 학습
- Planning, search
	- 환경을 알고 있을 때(reward, transition을 알고 있을 때), agent가 실제 environment에서 행동하지 않고, 내부적으로 computation을 통해 policy 개선
	- ex. Monte carlo tree search
	- Emulator가 있어서, 쿼리를 날리면서 다음 상황과 reward를 알아내면서 최적의 policy를 찾아내는 것
	- ex. tree search

Exploration and Exploitation

- 환경로부터 정보를 얻어서, 환경을 이해하는 과정에서 지금까지 얻은 정보를 통해 최선의 선택을 내리는 과정이 있는데, 정보를 모으는 것은 Exploration, 모인 정보를 바탕으로 최선의 선택을 내리는 것을 Exploitation
- Exploration, Exploitation은 서로 trade off 관계에 있다.
- Exploration는 Exploitation와는 다른 과정이다. 가보는 곳은 어떤 곳일지 모르고, 더 좋을 지도 모르고 탐험하는 것이기 때문에, 장기적으로 좋을 수는 있어도, 당장은 어떻게 될지는 모르는 관점이다.


Prediction and Control

- prediction
	- policy가 주어졌을 때, 미래를 평가
	- value function을 잘 학습 시키는 것
- control
	- 미래를 최적화하는 것
	- 최적의 policy를 찾는 것
