---
layout: post
title:  "Markov Decision Process"
date:   2021-02-11 00:00:00 +0900
category: "Reinforcement Learning"
tags: YouTube
---

## 참고

- video
    - [[강화학습 2강] Markov Decision Process](https://youtu.be/NMesGSXr8H4)
    - [RL Course by David Silver - Lecture 2: Markov Decision Process](https://youtu.be/lfHX2hHRMVQ)
- slide
    - [Lecture 2: Markov Decision Processes](https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf)



Introduction to MDPs

- MDP: RL에서의 environment을 표현, environment를 모두 관측할 수 있는 상황, 현재 state가 완전히 표현되는 것
- 거의 모든 강화학습 문제는 MDP형태로 만들 수 있다.
- ex. optimal control, bandits(카지노의 슬롯머신)

Markov Property

- s_t가 s_{t+1}로 갈 확률이 s_1 ~ s_t가 주어졌을 때, s_{t+1}로 갈 확률과 같다.
- 이전 과거를 다 버릴 수 있다. state가 history의 모든 관련된 정보를 가지고 있어서 state만 필요할 뿐,  history는 버릴 수 있다. state는 미래에 대한 충분한 통계적 표현형이다.



State Transition Matrix

- Markov process, Markov reward process, Markov decision process에 쓰이는 개념
- environment를 설명하기 위해 필요한 것
- State Transition probability: 시간이 t일때 S에 있을 때, Markov process에서는 action이 없이 매 step마다 확률적으로 다음 S로 옮겨간다. t에 있을 때, t+1에 될수 있는 여러개의 state로 전이될 수 있는 확률
- State Transition probability를 matrix로 표현되면 State Transition matrix이다.
	- state가 총 n개
	- P_{11} : state 1에서 state 1로 갈 확률


Markov Process(Markov chain)

- S: state들의 집합(n개), P: process의 전이 확률(n^2개)가 있으면 완전히 정의가 된다.
- memoryless: 어느 경로를 통해 여기로 왔는지 관계 없이, 여기로 온 순간 미래가 정해진다.
	- Markov property
- random process: 샘플링을 할 수 있다.
	- 어느 state에서 시작해서 주사위를 던지면서 이동해가면, state의 sequence가 생길 것이고, 다른 state에서 시작해서 주사위를 던지면, 또 다른 state의 sequence가 생길 것이다.
- 참고: state Transition Matrix가 어떤 조건들을 만족(Ergodicity)하면 Markov process이 최종 분포가 stationary가 된다.
	- 예를 들어 1억 명을 각 state에 골고루 퍼트려 놓고 충분히 전이시키면, 각 state 마다 있는 사람의 수가 일정하게 된다.
	- 어떤 조건인지는 알려주지 않음
- state의 집합과 transition matrix로 완전히 표현이 가능
- RL 관점에서 environment의 dynamic를 말한다.: environment가 어떤 원리로 동작하는지 dynamic에 대한 설명


Example: student Markov chain

- state 수: 7개
- state의 전이확률: 화살표
-  terminal state: 자신에게 돌아오는 마지막 종료 state
- FSM?
- Episode: 한번 어느 state에서 시작해서  termial state까지 가는 것
- 샘플링: 어떤 확률변수로부터 이벤트가 발생한 것

Example: student markov chain transition matrix

- transition probability를 matrix로 표현
- 그림이 없어도, matrix만 있으면 설명이 가능


---
Markov Reward Process

- 강화학습은 environment의 reward를 최대화하는 문제이다.
- S, P,(Markov process) R, \gamma가 있으면 완전히 표현 가능
	- S: state의 집합
	- P: state transition matrix
	- R: reward 함수, 어떤 state에 도달하면 어떤 reward 값을 주라고 state별로 정의
		- n개
	- \gamma: discount factor


Example: student MRP

- 빨간 것: reward
- state마다 reward가 주어진다.


Return

- reward와 다른 용어, reward를 maximize하는 것이 아니다.
- return을 최대로 하는 것이 정확한 표현
- sample된 경로 동안에 받는 reward의 총합
	- \gamma: 감가상각
		- 0에 가까울 수록 근시적인 보상에 집중
		- 1에 가까울 수록 장기적인 보상에 집중
	- 미래에 받는 reward를 감가상각 시켜서 더해준다.

Why discount?

- discount때문에 수렴성이 증명이 되기 때문에 수학적으로 편리하다.
	- 수렴해야 대소 비교를 할 수 있기 때문이다.
- 만약 모든 episode sequence가 종료됨이 보장된다면, \gamma를 1로 설정해도 될 때가 있다.
- silver: 문제에 따라서 필요할 때가 있고, 필요 없을 때도 있는데, 네가 동의하지 않는다면 알아서 잘 써


Value function

- return의 기댓값
- state가 input일때, 그 state에서 계속 샘플링하면서 episode를 만들어지면 episode마다 return이 생긴다.
- 같은 c_1에서 시작하더라도 어떻게 샘플링이 되느냐에 따라 return 값이 달라지는데, 이것을 평균낸 것
- state s에 왔을 때, 조건부 G_t의 기댓값

Example: State-Value Function for Student MRP

- 어떻게 구하는 지는 나중에 알려준다.


Bellman Equation for MRPs

- Value Function을 관장하는 식, value function이 학습되는 건 bellman equation에 근거해서  iterative하게 학습이 되는 것
- s에서의 value는 1step을 간 뒤에 얻는 R_{t+1}과 다음 state의 value에 \gamma를 곱한 것의 합과 같다. 즉 한 스텝을 더 간 뒤를 본다.


Bellman Equation in Matrix Form

- Bellman 방정식을 matrix로 표현 될 수 있다.

Solving the Bellman Equation

- R, P, \gamma가 주어져있으면, value function을 한번에 구할 수 있다.
- 계산 복잡도는 n^3이므로, 작은 MRP에서만 직접 계산이 가능하다.
- 큰 MRP는 iterative한 방법을 사용한다.

---
Markov Decision Process

- MRP에서 action이 추가된다.
- A: action들의 집합

Example: Student MDP

- action 마다 reward가 주어진다.
- action을 하면 무조건 다른 state를 가는 것이 아니라, 확률적으로 다른 state로 가게 된다. 무조건 다른 state를 가다고 하더라도, 그것은 1.0의 확률로 다음 state를 간다는 것을 의미한다.
-  MDP는 environment 이기 때문에 Policy는 나타나 있지 않음. Policy는 agent의 영역이다. 즉 agent가 어떤 Policy를 가지고 MDP를 돌아다닐 때, 가장 최대의 reward를 갖는 지를 찾아 내는 것이 MDP를 푼다는 뜻이다. 

Policy(1)

- MRP에서는 action을 안하기 때문에 policy가 없다.  state에 있으면,  state로 가는 확률 분포에 의해서 자동으로 시스템이 다음 state를 넘겨준다.
- action을 하는 것에 따라 다음 state가 달라지기 때문에, 어떤 policy에 따라 행동을 받을 지가 중요하다.
- state s에 있을 때, a를 선택할 확률 분포
-  agent의 행동을 완전히 결정해 주는 것
-  MDP의 policy: 현재 state에서 의존해서 행동을 결정

Policy(2)

-  MDP에서 어떤 Policy \pi를 가지고 움직인다고 할 때, 한 state에서 한 행동을 선택해서 다음 state로 움직이는 과정들을 Markov Process로 표현할 수 있다. 왜냐하면 policy \pi가 고정되면, state에서 다음 state로 갈 확률을 계산할 수 있기 때문이다.
- 그리고 state에서 다음 state로 갈 때 reward를 받고 있고, 그 reward를 계산 할 수 있기 때문에, MRP로 계산 할 수 있다.

Value Function

- 어떤 state 에서 policy \pi에 따른 episode를 여러개 만들 때, 그때의 평균 return
- State-Value function: input에 state만 들어간 뒤, 정책 \pi를 따라 움직일 때 return의 기대값
- Action-value function: input에 state, action이 들어간 뒤, 정책 \pi를 따라 움직일 때 return의 기댓값
	- Q 함수라고도 한다.
	- ex. Q-Learning, DQN의 Q

Bellman Expectation Equation

- state value function: 한 step을 가고, 그 다음 state부터 \pi를 따라 가는 것과 같다
- action value function: state에서 action을 통해 reward를 하나 받고, 다음 state에서 다음 action을 하는 것의 기댓값과 같다.

Bellman Expection Equation for V^\pi

- state에서 action을 할 확률이 있고 action을 했을 때,  action-value function의 가중치 합이 value function이다.
- V를 Q로 표현
Bellman Expection Equation for Q^\pi

- state에서의 action을 할때의 가치는 reward를 갖고 + \gamma * (state에서 action을 할 때 다음 state로 떨어질 확률 * 그 state의 가치들을 sigma로 더해 준 값)
- Q를 V로 표현


Bellman Expection Equation for V^\pi(2)

- 이전 두 식의 q(s, a)를 대입한 결과


Bellman Expectation Equation(Matrix Form)

- value function만 구했지, action은 어떻게 해야하는지는 구하지 않았다. 이것은 \pi를 따랐을 때,  value가 어떻게 되느냐만 구한 것


---
Optimal Value Function

- optimal state-value function: 가능한 모든 policy를 따르는 value function중 가장 나은 value function
- optimal action-value function: 가능한 모든 policy를 따르는 action-value function 중 가장 나은 action-value function
- MDP에서의 가능한 가장 좋은 성능을 나타낸 것이고, 이것을 아는 순간 MDP는 풀렸다고 말한다. 
- 작은 문제도 풀기가 쉽지 않다. optimal value function은 형렬 연산으로 풀수가 없기 때문이다. 행렬 형대로 표현이 되지 않는다.

Optimal Policy

- partial ordering: 어떤 2개의 policy가 주어졌을 때, 항상 2개 비교할 수 있는 것은 아니다. 대신, 어떤 2개에 대해서 이것이 더 낫다고 할 때가 존재하는데, 모든 s에 대해서 v_\pi가 v_\pi'보다 클 때, \pi가 \pi'보다 더 좋다.
- 정리:
	- MDP에 대해서 모든 policy에 대해서 optimal policy가 존재한다.
	- 모든 optimal policy들을 따르면 optimal value function가 된다. action-value function도 마찬가지


Finding an optimal policy

- q_*를 나는 순간 optimal policy 한개는 반드시 찾아진다. 즉 state에서 q_*를 알때, q_*의 action을 할 확률이 1이고 나머지가 0인  policy가 있다면(단지 q_*를 따라가는 policy) 그 policy는 optimal policy가 된다.
- 모든 MDP문제에서 policy는 본디 각 action에 대해 확률을 알려주기 때문에 stochastic하다. 위와 같이 하나의 행동이 1이고, 나머지가 0인 경우를 deterministic하다고 한다. 따라서 위와 같이 MDP에 대해서 deterministic optimal policy가 존재한다. 즉 정해진 답을 가지는 pollicy가 존재한다는 뜻
- q_*를 알고 있다면, optimal policy를 알고 있는 것이다.


Example: Optimal Policy for student MDP

- state가 1만개 있다면, 다음 state로 가는 선을 찾는 것이 어려운 문제이다.


Bellman Optimality Equation for v_*

- s에서의 optimal value는 q_*의 max를 구하면 된다.


Bellman Optimality Equation for V_*(2)

- max가 있기 때문에 linear equation이 아니기 때문에 역행렬을 구해서 계산할 수 없다.



Solving the Bellman Optimality Equation

- 벨만 최적 방정식은 non-linear이기 때문에 closed form 해가 없다.
- 따라서 여러개의 반복적인 해결법이 존재
	- value iteration (DP 방법론)
	- policy iteration (DP 방법론)
	- q-learning
	- sarsa