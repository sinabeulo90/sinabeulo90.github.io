---
layout: post
title:  "Lecture 2: Markov Decision Processes"
date:   2021-04-09 17:57:53 +0900
category: "Reinforcement Learning"
tags: YouTube
plugins: mathjax
---

David Silver 님의 [Introduction to reinforcement learning](https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) 강의 내용을 [팡요랩 Pang-Yo Lab](https://www.youtube.com/channel/UCwkGvF7xKz2E0Lv-fZ9wv2g)의 [강화학습의 기초 이론](https://youtube.com/playlist?list=PLpRS2w0xWHTcTZyyX8LMmtbcMXpd3s4TU)에서 풀어서 설명하기에, 이를 바탕으로 공부하여 정리한 내용입니다.

- Video: [[강화학습 2강] Markov Decision Process](https://youtu.be/NMesGSXr8H4)
- Slide: [Lecture 2: Markov Decision Processes](https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf)


---

## Markov Processes


### Introduction


#### Introduction to MDPs

- Markov decision process: RL에서의 environment를 표현한다.
    - Environment를 모두 관측할 수 있는 상황
    - 현재 state만으로 완전히 표현되는 것
- 거의 모든 RL 문제는 MDP형태로 만들 수 있다.
- Ex: Optimal control, Bandits(카지노의 슬롯머신)


<br>


### Markov Property


#### Markov Property
{: style="color: red"}

- $S_t$에서 $S_{t+1}$로 갈 확률은 $S_1, \dots, S_t$가 주어졌을 때, $S_{t+1}$로 갈 확률과 같다.
    - $\mathbb{P} [S_{t+1} \| S_t] = \mathbb{P} [S_{t+1} \| S_1, \dots, S_t]$
- State는 history의 모든 관련 정보를 가지고 있기 때문에, state만 필요할 뿐, history는 버릴 수 있다.
- 즉 state는 미래에 대한 충분한 통계적 표현형이다.


#### State Transition Matrix

- Markov process, Markov reward process, Markov decision process에 사용되는 개념이며, environment를 설명하기 사용한다.
- State transition probability
    - 시간 $t$일 때, 어떤 $S$에 있을 때, 다음 step인 $t+1$이 될 때 다른 여러 개의 state로 전이 될 확률
    - $\mathcal{P_{ss'}} = \mathbb{P} [S_{t+1} = s' \| S_t = s]$
    - Markov process에서는 action 없이 매 step마다 확률적으로 다음 $S$로 옮겨간다.
- State transition matrix: State transition probability를 행렬로 표현한 것
    - $\begin{array}{ccc}
                                    & \text{to} \newline
        \mathcal{P} = \text{from}   & \begin{bmatrix}
                                        \mathcal{P_{11}} & \cdots & \mathcal{P_{1n}}    \newline
                                        \vdots           &        &                     \newline
                                        \mathcal{P_{n1}} & \cdots & \mathcal{P_{nn}}
                                    \end{bmatrix}
    \end{array}$
    - state의 갯수는 총 $n$개이고, 행렬의 각 행의 합은 1이다.
    - $\mathcal{P}_{11}$: state 1에서 state 1로 이동할 확률


<br>


### Markov Chains


#### Markov Process(Markov chain)
{: style="color: red"}

- State들의 집합 $\mathcal{S}$ ($n$개)와 각 state들의 전이확률 $\mathcal{P}$ ($n^2$개)가 있으면 완전히 정의된다.
    - $\langle \mathcal{S}, \mathcal{P} \rangle$
    - State의 집합과 Transition matrix로 완전히 표현 가능하다.
- A Markov process is a memoryless random process. i.e. a sequence of random states $S_1, S_2, \dots$ with the Markov property
    - Memoryless: 어느 경로를 통해 어떤 위치로 왔는지 관계없이, 위치한 순간 미래가 정해진다.(Markov property)
    - Random procss: 샘플링 할 수 있다.
        - 어떤 state에서 주사위를 던지면서 이동해가면, state의 sequence가 생길 것이다.
        - 다른 state에서 주사위를 던지면서 이동해가면, 또 다른 state의 sequence가 생길 것이다.
- Transition matrix가 어떤 조건들을 만족하면, Markov process의 최종 분포가 stationary가 된다.
    - Ex: 1억 명을 각 state에 골고루 위치시키고 충분히 전이시키면, 각 state에 있는 사람의 수가 일정하게 된다.
    - ~~어떤 조건인지는 알려주지 않았다.~~
- RL관점에서 environment가 어떤 원리로 동작하는지, dynamic에 대해 설명한다.


#### Example: Student Markov Chain

![Markov Process Example](/assets/rl/markov_process_ex.png)

- 7개의 State를 가지고 있고, state의 전이확률은 화살표로 표시되어 있다.
- Terminal state: 자신에게 돌아오는 마지막 종료 state
- Episode: 어떤 state에서 시작해서 terminal state까지 가는 것
- Sampling: 어떤 확률변수로부터 발생한 이벤트


#### Example: Student Markov Chain Transition Matrix

![State Transition Matrix](/assets/rl/markov_process_transition_matrix.png)

- Transition probability를 행렬로 표현할 수 있고, 행렬로 표현할 수 있으면 그림이 없어도 설명이 가능하다.


---


## Markov Reward Processes


### MRP


#### Markov Reward Process

- MRP는 Markov chain에서 가치 값이 포함된 것이다.
    - RL은 environment의 reward를 최대화하는 문제이다.
- $\mathcal{S}, \mathcal{P}$(Markov process)와 $\mathcal{R}, \gamma$가 있으면 완전히 정의된다.
    - $\langle \mathcal{S}, \mathcal{P}, \mathcal{R}, \gamma \rangle$
    - $\mathcal{S}$: State의 집합
    - $\mathcal{P}$: State transition matrix
        - $mathcal{P_{ss'}} = \mathbb{P} [S_{t+1} = s' \| S_t = s]$
    - $\mathcal{R}$: Reward 함수
        - 어떤 state에 도달하면, 어떤 reward를 주어야 할지 state별로 정의한다.
        - $\mathcal{R_s} = \mathbb{E} [R_{t+1} \| S_t = s]$
    - $\gamma$: Discount factor
        - $\gamma \in [0, 1]$


#### Example: Student MRP

![Markov Reward Process](/assets/rl/markov_reward_process_ex.png)

- Reward: 빨간색 값이며, 각 state마다 reward가 주어진다.


#### Return
{: style="color: red"}

- $G_t = R_{t+1} + \gamma R_{t+2} + \dots = \sum^\infty_{k=0}\gamma^k R_{t + k + 1}$
    - Sampling된 경로 동안 받은 reward의 총 합
    - $\gamma$: 감가상각
        - 0에 가까울 수록 근시안적인 reward에 집중
        - 1에 가까울 수록 장기적인 reward에 집중
    - 미래의 reward에 감가상각을 적용하여 더해준다.
- RL의 목표는 Return을 maximize로 하는 것이 정확한 표현이다. reward를 maximize하는 것이 아니다.


#### Why discount?

- Discount를 사용하면 수렴성이 증명되어 대소비교를 할 수 있기 때문에, 수학적으로 편리하다.
- 만약 모든 episode sequence가 종료된다는 것이 보장된다면, $\gamma$를 1로 설정해도 될 때가 있다.
- 문제에 따라서 필요할 때가 있고, 필요 없을 때도 있으므로, 알아서 잘 사용하면 된다.


<br>


### Value Function


#### Value Function

- $v(s) = \mathbb{E} [G_t \| S_t = s]$
    - State $s$에서 시작할 때, *return의 기댓값*{: style="color: red"}
- State s에서부터 계속 sampling하여 episode들이 만들어 지면, 각 episode마다 return이 계산된다.
- 같은 $s_1$에서 시작하더라도, 어떻게 샘플링되느냐에 따라 return 값이 달라지는데, 이 return을 평균 낸 것이다.


<br>


### Bellman Equation
{: style="color: red"}


#### Bellman Equation for MRPs

- $\begin{aligned}
    v(s) &= \mathbb{E} [G_t \| S_t = s] \newline
         &= \mathbb{E} [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \| S_t = s] \newline
         &= \mathbb{E} [R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \dots) \| S_t = s] \newline
         &= \mathbb{E} [R_{t+1} + \gamma G_{t+1} \| S_t = s] \newline
         &= \mathbb{E} [R_{t+1} + \gamma v(S_{t+1}) \| S_t = s]
\end{aligned}$
    - *Value function이 학습되는 것은 Bellman equation에 근거해서 iterative하게 학습되는 것이다.*{: style="color: red"}
    - $s$의 value는 한 step 간 뒤에 얻는 $R_{t+1}$과 다음 $s'$의 value에 $\gamma$를 곱한 것의 합과 같다.
    - 즉 한 step 앞의 value를 사용한다.


#### Bellman Equation in Matrix Form

- $v = \mathcal{R} + \gamma \mathcal{P}v$
- $\begin{bmatrix}
        v(1)    \newline
        \vdots  \newline
        v(n)
    \end{bmatrix}
    = 
    \begin{bmatrix}
        \mathcal{R_{1}} \newline
        \vdots          \newline
        \mathcal{R_{n}}
    \end{bmatrix}
    +
    \gamma
    \begin{bmatrix}
        \mathcal{P_{11}} & \dots & \mathcal{P_{1n}} \newline
        \vdots           &       &                  \newline
        \mathcal{P_{n1}} & \dots & \mathcal{P_{nn}}
    \end{bmatrix}
    \begin{bmatrix}
        v(1)    \newline
        \vdots  \newline
        v(n)
    \end{bmatrix}$
- Bellman 방정식을 행렬로 표현할 수 있다.


#### Solving the Bellman Equation

- $\begin{aligned}
    v &= \mathcal{R} + \gamma \mathcal{P}v      \newline
    (1 - \gamma \mathcal{P})v &= \mathcal{R}    \newline
    v &= (1 - \gamma \mathcal{P})^{-1} \mathcal{R}
\end{aligned}$
- $\mathcal{R}, \mathcal{P}, \gamma$가 주어졌다면, Bellman equation은 선형방정식이므로 value function을 한번에 구할 수 있다.
    - 계산 복잡도: $O(n^3)$
    - 작은 MRP에서만 가능하다.
- 큰 MRP는 iterative 방법을 사용한다.


---


## Markov Decision Processes


### MDP


#### Markov Decision Process

- MDP는 Markov reward process에서 action이 포함된 것이다.
- $\mathcal{S}, \mathcal{P}, \mathcal{R}, \gamma$(Markov reward process)와 $mathcal{A}$가 있으면 완전히 정의된다.
    - $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$
    - $\mathcal{A}$: Action의 집합
    - $\mathcal{P}$: State transition matrix
        - $\mathcal{P^a_{ss'}} = \mathbb{P} [S_{t+1} = s' \| S_t = s, A_t = a]$
    - $\mathcal{R}$: Reward 함수
        - 어떤 state에 도달하면, 어떤 reward를 주어야 할지 state별로 정의한다.
        - $\mathcal{R^a_s} = \mathbb{E} [R_{t+1} \| S_t = s, A_t = a]$


#### Example: Student MDP

![Markov Decision Process](/assets/rl/markov_decision_process_ex.png)

- Action마다 reward가 주어진다.
- *Action을 하면 항상 그 다음 state로 가는 것은 아니고, 확률적으로 다른 state로 가게 된다. 항상 다음 state로 간다고 하면, 그것은 1.0의 확률로 다음 state로 간다는 것을 의미한다.*{: style="color: red"}
- MDP는 environment이기 때문에 여기서는 policy가 존재하지 않는다.
    - Policy는 agent의 영역이다.
    - Agent가 어떤 policy를 가지고 MDP를 돌아다닐 때, 최대의 return를 갖는 policy를 찾았다면, 이때 MDP를 풀었다고 말한다.


<br>


### Policies


#### Policies (1)

- *MRP에서는 action을 하지 않기 때문에 policy가 없다.*{: style="color: red"} 즉 state에서 다음 state로 가는 확률분포에 의해 시스템이 자동으로 다음 state를 알려준다.
- MDP에서는 action에 따라 다음 state가 달라지기 때문에, 어떤 policy에 따라 어떤 행동을 선택할 지가 중요하다.
- $\pi(A \| s) = \mathbb(P) [A_t = a \| S_t = s]$
    - State $s$에 있을 때, action $a$를 선택할 확률 분포
    - 현재 state에 의존해서 agent의 행동을 완전히 결정해준다.


#### Policies (2)

- MDP에서 어떤 policy $\pi$를 통해 행동을 선택한다고 할 때, 한 state에서 어떤 행동을 선택해서 다음 state로 움직이는 과정들은 Markov process로 표현할 수 있다. 왜냐하면 policy $\pi$가 고정되면, 현재 state에서 다음 state로 갈 확률을 계산할 수 있기 때문이다.
    - Markov process $\langle \mathcal{S}, \mathcal{P^\pi} \rangle$
    - $\mathcal{P_{ss'}^{\pi}} = \sum_{a \in \mathcal{A}} \pi(a \| s) \mathcal{P_{ss'}^{a}}$
- 마찬가지로, 한 state에서 어떤 행동을 선택해서 다음 state로 움직일 때 reward를 얻게되는 과정들을 MRP로 표현할 수 있다.
    - Markov reward process $\langle \mathcal{S}, \mathcal{P^\pi}, \mathcal{R^\pi}, \gamma \rangle$
    - $\mathcal{R_{s}^{\pi}} = \sum_{a \in \mathcal{A}} \pi(a \| s) \mathcal{R}_{s}^{a}$


<br>


### Value Functions


#### Value Function

- 어떤 state에서 policy $\pi$를 따르는 episode를 여러 개 만들 때, 이 때 return의 기댓값
- State-value function: 입력으로 state를 받고 policy $\pi$를 따라 움직일 때, return의 기댓값
    - $v_\pi(s) = \mathbb{E_\pi} [G_t \| S_t = s]$
- Action-value function: 입력으로 state와 action을 받고 policy $\pi$를 따라 움직일 때, return의 기댓값
    - $q_\pi(s, a) = \mathbb{E_\pi} [G_t \| S_t = s, A_t = a]$
    - Q-function이라고도 하며, Q-Learning의 Q, DQN의 Q와 같은 의미이다.


<br>


### Bellman Expectation Equation
{: style="color: red"}


#### Bellman Expectation Equation

- State-value function: 한 step을 가고, 다음 step부터 policy $\pi$를 따라갈 때, 이 때 return의 기댓값
    - $v_\pi(s) = \mathbb{E_\pi} [R_{t+1} + \gamma v_\pi(S_{t+1}) \| S_t = s]$
- Action-value function: 한 state에서 어떤 action을 통해 reward를 하나 받고, 다음 state에서 policy $\pi$를 따라 action을 선택할 때, 이 때 return의 기댓값
    - $q_\pi(s, a) = \mathbb{E_\pi} [R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) \| S_t = s, A_t = a]$


#### Bellman Expectation Equation for $V^\pi$

![Bellman Expectation Equation for $V^\pi$](/assets/rl/bellman_expectation_equation_v1.png)

- $v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \| s) q_\pi(s, a)$
    - State-value function $v$를 action-value function $q$로 표현할 수 있다.
    - 모든 action에 대해, (한 state에서 $\pi$를 통해 어떤 action을 선택할 확률) * (해당 action에 대한 action-value function)의 모든 합(가중치 합) = state-value function


#### Bellman Expectation Equation for $Q^\pi$

![Bellman Expectation Equation for $Q^\pi$](/assets/rl/bellman_expectation_equation_q1.png)

- $q_\pi(s, a) = \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^a} v_\pi(s')$
    - action-value function $q$를 State-value function $v$로 표현할 수 있다.
    - (한 state에서 어떤 action을 수행할 때 얻은 reward) + $\gamma$ * [현재 state에서 이동가능한 다음 state에 대해, (한 state에서 다음 state로 이동할 확률) * (다음 state의 state-value function)의 모든 합(가중치 합)] = action-value function


#### Bellman Expectation Equation for $V^\pi$ (2)

![Bellman Expectation Equation for $V^\pi$](/assets/rl/bellman_expectation_equation_v2.png)

- $\begin{aligned}
    v_\pi(s) &= \sum_{a \in \mathcal{A}} \pi(a \| s) q_\pi(s, a)    \newline
             &= \sum_{a \in \mathcal{A}} \pi(a \| s) \left(\mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^a} v_\pi(s') \right)
\end{aligned}$
    - 위 두개의 식 $v_\pi$와 $q_\pi$를 대입한 결과


#### Bellman Expectation Equation for $Q^\pi$ (2)

![Bellman Expectation Equation for $Q^\pi$](/assets/rl/bellman_expectation_equation_q2.png)

- $\begin{aligned}
    q_\pi(s, a) &= \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^a} v_\pi(s')     \newline
                &= \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^{a}} \sum_{a' \in \mathcal{A}} \pi(a' \| s') q_\pi(s', a')
\end{aligned}$
    - 위 두개의 식 $v_\pi$와 $q_\pi$를 대입한 결과


#### Bellman Expectation Equation (Matrix Form)

- [Solving the Bellman Equation](#solving-the-bellman-equation) 장에서 MRP에 대해서 행렬로 한번에 풀었던 것과 유사한 방법으로 풀수 있다.
- $\begin{aligned}
    v_\pi &= \mathcal{R^\pi} + \gamma \mathcal{P^\pi}v      \newline
    (1 - \gamma \mathcal{P^\pi})v_\pi &= \mathcal{R^\pi}    \newline
    v_\pi &= (1 - \gamma \mathcal{P^\pi})^{-1} \mathcal{R^\pi}
\end{aligned}$

- 지금까지는 $\pi$를 따랐을 때 value function이 어떻게 되는가에 대해서만 계산했다.
- 아직 action은 어떻게 해야하는지는 고려하지 않았다.


<br>


### Optimal Value Function


#### Optimal Value Function

- Optimal state-value function: 가능한 모든 policy를 따르는 value function 중 가장 나은 value function
    - $v_*(s) = \max_\pi v_\pi(s)$
- Optimal action-value function: 가능한 모든 policy를 따르는 action-value function 중 가장 나은 action-value function
    - $q_*(s, a) = \max_\pi q_\pi(s, a)$
- MDP에서 가장 좋은 성능을 나타낸 것이고, 이것을 아는 순간 MDP가 풀렸다고 말한다.
- 하지만 작은 문제도 풀기가 쉽지 않다. Optimal value function은 행렬 형태로 표현되지 않으므로, 행렬 연산으로 풀수 없기 때문이다.



#### Optimal Policy

- *Partial ordering*{: style="color: red"}: 어떤 2개의 policy가 주어졌을 때, 항상 2개를 비교할 수 있는 것은 아니다. 대신, 어떤 policy가 다른 policy보다 더 낫다고 할 때가 존재한다. 따라서 모든 state에 대해서 $v_\pi$가 $v_\pi'$보다 클 때, $\pi$가 $\pi'$보다 더 좋다고 할 수 있다.
    - $\pi \geq \pi' \  \text{if} \  v_\pi(s) \geq v_{\pi'}(s), \forall s$
- MDP에서, 모든 policy에 대해 optimal policy가 존재한다.
    - $\pi_* \geq \pi, \forall \pi$
- 모든 optimal policy들을 따르면 optimal value function과 optimal action-value function이 된다.
    - $v_{\pi_\*}(s) = v_\*(s)$
    - $q_{\pi_\*}(s, a) = q_\*(s, a)$


#### Finding an Optimal Policy

- $\pi_\*(a \| s) = \begin{cases}
                        1 \  & \text{if} \  a = \arg\max_{a \in \mathcal{A}} q_\*(s, a)    \newline
                        0 \  & \text{otherwise}
                    \end{cases}$
    - 모든 state에서 $q_\*$를 알 때, $q_\*$가 가장 큰 action을 선택할 확률이 1이고, 나머지 action을 선택할 확률이 0인 policy($q_\*$를 따라가는 policy)가 있다면, 해당 policy는 optimal policy가 된다.
        - 모든 MDP문제에서의 policy는 본래 각 action에 대해 확률을 알려주기 때문에 stochastic하다.
        - 위와 같이 한 행동을 선택할 확률이 1이고, 나머지가 0인 경우 deterministic하다고 한다.
        - $q_\*$를 알고 있고, 위와 같이 정해진 답을 가지는 policy를 deterministic optimal policy라고 한다.
- $q_\*$를 아는 순간 optimal policy 1개는 반드시 알 수 있다.


<br>

### Bellman Optimality Equation
{: style="color: red"}


#### Bellman Optimality Equation for $V^\*$

![Bellman Optimality Equation for $V^\*$](/assets/rl/bellman_optimality_equation_v1.png)

- $v_\*(s) = \max_a q_\*(s, a)$
- $s$에서의 optimal value는 구하려면 $q_\*$의 max를 구하면 된다.


#### Bellman Optimality Equation for $V^\*$ (2)

![Bellman Optimality Equation for $V^\*$](/assets/rl/bellman_optimality_equation_v2.png)

- $v_\*(s) = \max_a \left( \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^a} v_\*(s') \right)$


#### Solving the Bellman Optimality Equation

- Bellman optimality equation에 있는 $\max$로 인해 non-linear가 되어, closed form solution이 없다.
- 따라서 다양한 iterative 방법들로 해를 구한다.
    - Value iteration (Dynamic Programming 방법론)
    - Policy iteration (Dynamic Programming 방법론)
    - Q-learning
    - Sarsa
