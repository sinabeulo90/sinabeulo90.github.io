---
layout: post
title:  "Lecture 3: Planning by Dynamic Programming"
date:   2021-04-12 13:55:53 +0900
category: "Reinforcement Learning"
tags: YouTube
plugins: mathjax
---

David Silver 님의 [Introduction to reinforcement learning](https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) 강의 내용을 [팡요랩 Pang-Yo Lab](https://www.youtube.com/channel/UCwkGvF7xKz2E0Lv-fZ9wv2g)의 [강화학습의 기초 이론](https://youtube.com/playlist?list=PLpRS2w0xWHTcTZyyX8LMmtbcMXpd3s4TU)에서 풀어서 설명하기에, 이를 바탕으로 공부하여 정리한 내용입니다.

- Video: [[강화학습 3강] Planning by Dynamic Programming](https://youtu.be/rrTxOkbHj-M)
- Slide: [Lecture 3: Planning by Dynamic Programming](https://www.davidsilver.uk/wp-content/uploads/2020/03/DP.pdf)


---

## Outline

- *Planning: MDP에 대한 모든 정보를 알고 있을 때, MDP 안에서의 최적 policy를 찾는 것이다.*{: style="color: red"}
- Policy evaluation: Prediction 문제
	- Policy가 정해질 때, MDP 안에서의 value function을 찾아 policy를 평가한다.
	- Ex: 미로찾기에서 현재 위치부터 마지막 위치까지 평균적으로 얼만큼의 value function을 갖는지 찾는 것이다.
- Policy iteration(Value iteration): Control 문제
	- 어떤 iterative한 방법론을 통해, 최적 policy를 찾아가는 2가지가 방법이 있다.


---

## Introduction


### What is Dynamic Programming?

- 복잡한 문제를 푸는 한 방법론으로, 큰 문제를 작은 문제들로 나눠서 답을 찾고, 이 답들을 모아 큰 문제를 푸는 일반적인 방법론이다.
- 강화학습은 매우 큰 문제이고, 그 안에서 model-free와 model-based로 나뉘어서 풀 수 있다.
	- Model-free: environment가 어떤 정보를 던져줄지 모를 경우, 즉 완전한 정보가 없을 때 푸는 방법이다.
	- Model-based: environment에 대한 모델이 있어서 agent가 어떤 행동을 하면 어떤 확률로 어떤 state로 움직이게 되는지 알고 있을 때 푸는 방법이다.
		- Ex: Dynamic programming


### Requirements for Dynamic Programming

- Dynamic programming을 사용하기 위해서는 2가지 조건이 필요하다.
	1. Optimal substructure: 큰 문제에 대한 optimal solution이 작은 문제들로 나뉠 수 있어야 한다.
	2. Overlapping subproblems: 한 subproblem을 풀면, 그 값을 저장해 둔 뒤 재사용 할 수 있어야 한다.
		- Ex: A에서 B까지 가는 최적의 길찾기 문제를 풀 때, A에서 중간 지점 M까지, M부터 B가지 가는 최적의 경로를 알고 있다면 A에서 B까지 가는 최적 경로를 풀 수 있을 것이다.
- Markov decision process는 위 2개 조건을 모두 만족하므로, dynamic programming을 적용할 수 있다.
	1. Bellman equation의 recursive하게 엮여 있는 형태이므로, 위 조건의 1에 해당한다.
	2. 작은 문제들에 대한 해인 value function을 저장하여 재사용할 수 있으므로, 위 조건의 2에 해당한다.


### Planning by Dynamic Programming

- Dynamic programming은 MDP에 대한 모든 지식을 알고 있다고 가정한다.
	- State transition probability, Reward 등
- Planning: MDP를 푸는 것을 의미한다.
	- Prediction
		- MDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$와 policy $\pi$ 또는 MRP $\langle \mathcal{S}, \mathcal{P^\pi}, \mathcal{R^\pi}, \gamma \rangle$ 로부터 value function $v_\pi$를 찾는다.
		- 최적의 policy인지 아닌지는 상관없이 어떤 바보같은 policy라도, MDP안에서 어떤 state에서 시작해서 끝날때 까지 얼만큼의 return을 받을지 예측한다.
	- Control
		- MDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$ 로부터 optimal value function $v_\*$와 optimal policy $\pi_\*$를 찾는다.


---

## Policy Evaluation


### Iterative Policy Evaluation


#### Iterative Policy Evaluation

- Policy evaluation(Prediction)
	- 어떤 policy를 따랐을 때, value function을 찾는 문제이다.
	- Ballman expectation equation을 반복적으로 적용하여 값을 찾는다.
	-  $v_1 \rightarrow v_2 \rightarrow \cdots \rightarrow v_\pi$
- Synchronous backups
	- Backup: Cache와 비슷하게 메모리에 저장해 두는 것을 의미한다.
	- 모든 state에 대해 iteration마다 업데이트한다. (Full sweep)
	- 현재 step에 있는 value를 계산할 때, 다음 step의 value를 이용하여 조금씩 더 정확하게 만든다.
		- 처음에는 모든 state의 value function $v_1$을 임의의 값 또는 0으로 초기화한다.
		- 한 번 iterative한 방법을 사용해서 $v_2$를 만들고, 다시 한번 반복하여 $v_2$에서 $v_3$를 만든다. 이것을 계속 반복하면, $v_\pi$에 수렴한다.
		- 이때, $v_\pi$는 policy $\pi$에 대한 value function이다.
	- 점화식은 현재 state를 이용해서 다음 state를 갱신하므로, 점화식과는 다른 방식이다.
- Asynchronous backups: [뒤에서 설명](#asynchronous-dynamic-programming)


#### Iterative Policy Evaluation (2)

![Iterative Policy Evaluation](/assets/rl/iterative_policy_evaluation.png)

- [Bellman expectation equation]({% post_url /RL/2021-04-09-markov-decision-process %}#bellman-expectation-equation)
	- 어떤 state의 value를 계산할 때, 다음 state의 정확하지 않은 value를 이용해서 갱신되지만, 정확한 reward가 있기 때문에 조금씩 정확한 value로 다가가게 된다.
	- $\begin{aligned}
		v_{k+1}(s) &= \sum_{a \in \mathcal{A}} \pi(a \| s) \left( \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^a} v_k(s') \right)	\newline
		v^{k+1} &= \mathcal{R^\pi} + \gamma \mathcal{P^\pi} v^k
	\end{aligned}$


<br>


### Example: Small Gridworld


#### Evaluating a Random Policy in the Small Gridworld

![Iterative Policy Evaluation](/assets/rl/small_gridworld_1.png)

- Prediction 문제를 풀기 때문에, policy가 있어야 하므로 MDP가 주어져야 한다.
- Random policy: 1/4 확률로 행동을 선택한다.
	- $\pi(\text{North} \| \cdot ) = \pi(\text{East} \| \cdot ) = \pi(\text{South} \| \cdot ) = \pi(\text{West} \| \cdot ) = 0.25$
- reward가 -1일 경우, 각 state의 value는 어떻게 될까?
- 만약 2번 state에서 얼마의 확률로 에피소드가 종료될까?
	- 알기 매우 어려운 문제이다.


#### Iterative Policy Evaluation in Small Gridworld

![Iterative Policy Evaluation](/assets/rl/small_gridworld_2.png)

- 처음에는 모든 state의 value function을 0으로 초기화했지만, 결국 true value function에 도달한다.
	- 이론적으로 어떻게 초기화를 해도 true value function에 도달한다.
- 우리는 random policy를 평가했을 뿐이지만, prediction 과정에서 평가한 value function에 대해 greedy하게 행동하면 optimal policy를 얻게 된다.
	- Greedy: 다음 state 중 가장 높은 value를 가지는 state로 움직이는 것을 의미한다.
	- 즉, 모든 문제에 대해 greedy하게 움직이는 policy를 찾기만 해도 더 나은 policy를 찾을 수 있다.


#### Iterative Policy Evaluation in Small Gridworld (2)

![Iterative Policy Evaluation](/assets/rl/small_gridworld_3.png)

- Policy evaluation 과정에서 단 3번의 iteration으로 optimal policy를 찾게 되었다.


---

## Policy Iteration


### How to Improve a Policy

- Policy iteration
	1. Evaluate the policy $\pi$: policy $\pi$에 대한 value function을 찾는다.
		- $v_\pi(s) = \mathbb{E} [R_{t+1} + \gamma R_{t+2} + \dots \| S_t = s]$
	2. 찾은 value function에 대해 greedy하게 움직이는 이전 policy $\pi$보다 더 나은 새로운 policy $\pi'$를 찾는다.
		- $\pi' = \text{greedy}(v_\pi)$
- Policy iteration을 반복하면, 작은 Gridworld 문제에서는 금방 optimal policy를 내지만, 일반적으로 더 많은 iteration을 수행해야 한다.
- *Policy iteration은 항상 optimal policy에 수렴한다.*{: style="color: red"}


### Policy Iteration

![Policy Iteration](/assets/rl/policy_iteration.png)


<br>


### Example: Jack's Car Rental


#### Jack's Car Rental

- 렌트카 지점 A, B가 있고, 지점마다 최대 20대의 차가 있을 수 있다.
- 차 한 대를 빌려줄 때 마다 10달려의 수익을 얻는다.
- 밤 중에는, A지점에서 B지점으로 또는 B지점에서 A지점으로 계속 차를 옮길 수 있다.
- A지점은 [푸아송 분포](https://ko.wikipedia.org/wiki/푸아송_분포)에 따라 고객들이 찾아온다.
	- 푸아송 분포: 정해진 단위 시간 동안 사건이 발생할 확률 분포
- 하루 평균 A지점은 3번의 렌트 요청과 3번의 반납이 발생하고, B지점은 4번의 렌트 요청과 2번의 반납이 발생한다.
	- 최대 수익을 얻기 위해, B지점에서 수요가 더 많으므로 A지점의 차량을 B지점으로 옮기는 것이 더 나을 수 있을 것이다.


#### Policy Iteration in Jack's Car Rental

- x축: B지점의 차량 수
- y축: A지점의 차량 수
- 등고선: greedy하게 움직이는 policy
	- +5: 현재 state에서 A지점에서 B지점으로 차량 5대를 옮기는 policy를 의미한다.
	- -5: 현재 state에서 B지점에서 A지점으로 차량 5대를 옮기는 policy를 의미한다.
- 마지막 그림: Evaluation을 할 때의 value
	- z축의 [420, 612]: Accumulate sum을 한 reward, 즉 value function을 의미한다.


<br>


### Policy Improvement


#### Policy Improvement

- 증명: Evaluation한 value function에 대해 greedy하게 움직이는 policy는 항상 현재 policy보다 더 나은 policy인가?
	- 더 나은 policy이다.
- $q_\pi(s, \pi'(s)) = \max_{a \in \mathcal{A}} q_\pi(s, a) \geq q_\pi(s, \pi(s)) = v_\pi(s)$
	- $q$: Action-value function
	- $\pi'$: Greedy policy
	- $q_\pi(s, \pi'(s))$: 1-step만 $\pi'$을 따라가고, 다음부터는 $\pi$를 따라갈 때의 action-value function
- $\begin{aligned}
		v_\pi(s) & \leq q_\pi(s, \pi'(s)) = \mathbb{E_{\pi'}} [R_{t+1} + \gamma v_\pi(S_{t+1}) \| S_t = s]	\newline
				 & \leq \mathbb{E_{\pi'}} [R_{t+1} + \gamma q_\pi(S_{t+1}, \pi'(S_{t+1})) \| S_t = s]	\newline
				 & \leq \mathbb{E_{\pi'}} [R_{t+1} + \gamma R_{t+2} + \gamma^2 q_\pi(S_{t+2}, \pi'(S_{t+2})) \| S_t = s]	\newline
				 & \leq \mathbb{E_{\pi'}} [R_{t+1} + \gamma R_{t+2} + \dots \| S_t = s]	= v_{\pi'}(s)
\end{aligned}$
	- 1-step에 대해서, greedy policy가 더 나은 것처럼, iterative하게 움직이면 더 많은 value를 가져다준다.


#### Policy Improvement (2)

- 계속 반복해 나아가다 더 이상 개선이 이루어지지 않는 경우는 Bellman optimality equation이 만족하는 상황이므로, 이때의 $\pi$를 모든 state $s$에 대해 optimal policy라고 할 수 있다.
	- $q_\pi(s, \pi'(s)) = \max_{a \in \mathcal{A}} q_\pi(s, a) = q_\pi(s, \pi(s)) = v_\pi(s)$
	- $v_\pi(s) = \max_{a \in \mathcal{A}} q_\pi(s, a)$
	- $v_\pi(s) = v_\*(s)$ for all $s \in \mathcal{S}$
- *Local이 아닌 optimal policy를 가진다.*{: style="color: red"}


<br>

### Extensions to Policy Iteration


#### Modified Policy Iteration

- Policy evaluation 단계에서 반드시 $v_\pi$에 수렴해야 할까?
	- 수렴하지 않더라고, 좀 더 일찍 멈추면 안될까?
	- Ex: 정해진 횟수만큼만 evaluation을 진행하고, improvement를 진행하면 안될까?
	- Ex: 극단적으로 단 한 번만 평가하고, 단 한 번만 improvement를 하면 안될까?
- 위의 어떤 경우에도 완전히 합리적인 방법이다.


#### Generalised Policy Iteration

![Generalised Policy Iteration](/assets/rl/generalised_policy_iteration.png)

- 어떤 알고리즘을 사용하던 evaluation하면 되고, 어떤 알고리즘을 사용하던 improvement하면 된다.


---

## Value Iteration
{: style="color: red"}


### Value Iteration in MDPs


#### Principle of Optimality

- Optimal policy는 2가지 요소로 나뉠 수 있다.
	1. 처음 optimal action $A_\*$을 선택한다.
	2. 그리고 다음 state $S'$에서 optimal policy를 따라간다.
- Principle of Optimality 정리
	- State $s$에서 도달가능한 모든 state $s'$에 대해, policy $\pi$가 $s'$에서 optimal value를 가지게 되면($v_\pi(s') = v_\*(s')$), policy $\pi(a \| s)$는 $s$에서 optimal value를 가진다($v_\pi(s) = v_\*(s)$).


#### Deterministic Value Iteration

- Subproblem $s'$의 해 $v_\*(s')$를 알고 있을 때, 1-step lookahead로 $v_\*(s)$를 계산할 수 있다.
	- $v_\*(s) \leftarrow \max_{a \in \mathcal{A}} \left( \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^a} v_\*(s') \right)$
	- Bellman optimality equation
- 직관적 접근
	1. 출발지 A부터 목적지 Z까지의 최단 거리를 구하기 위해, 목적지 Z의 이전 경유지들부터 Z까지의 최단 거리를 구한다.
	2. 이때의 최단 거리의 경유지가 M이라고 한다면, 다시 M의 이전 경유지들부터 M까지의 최단 거리를 구한다.
	3. 이를 계속 반복한다.
- Optimal value가 수렴됨이 이미 증명되었다.
- Policy iteration과 value iteration의 차이
	- Policy iteration: Policy가 있으므로, policy evaluation과 policy improvement를 진행한다.
	- Value iteration: Policy 없이 value만을 가지고 iteration을 진행한다.
		- $k=1$인 policy evaluationr과 동일하다.
	

#### Example: Shortest Path

![Shortest Path](/assets/rl/shortest_path.png)

- Policy가 없으므로,  step마다 Bellman optimality equation을 적용한다.
- 간단한 문제의 경우 직접 풀 수도 있지만, 실제 복잡한 문제에서는 terminate state가 어디에 있는지 알기 어렵고, 심지어 존재하는지도 알 수 없기 때문에, 매 step마다 asynchronous하게 full sweep으로 모든 state를 확인한다.
- 매 step마다 업데이트될 때, terminate state에 가까운 state들부터 가장 멀리 있는 state로 value가 확정된다.
- 모든 state에서 value가 확정되면, 이 value를 통해서 policy를 알게 된다.
	- Iteration이 진행되면서 마치 policy가 업데이트되는 것처럼 보이는데, 이는 value iteration이 $k=1$인 policy evaluation과 동일하기 때문이다.
	- 하지만, 현재 step에서 정해진 value를 통해 greedy policy를 구한다고 해도, 다음 step의 value는 이전 greedy policy로 계산되는 값은 아니다.


#### Value Iteration

- Problem: Optimal policy $\pi$ 찾기
- Solution: Bellman optimality backup을 계속 적용한다.
	- $v_1 \rightarrow v_2 \rightarrow \cdots \rightarrow v_\*$
	- Synchronous backup: Iteration $k+1$ 마다 모든 state $s$에 대해, $v_k(s')$을 사용하여 $v_{k+1}(s)$을 업데이트한다.
- Policy iteration과 다르게 policy가 없다.
	- $\begin{aligned}
		v_{k+1}(s) &= \max_{a \in \mathcal{A}} \left( \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^a} v_k(s') \right)	\newline
		v_{k+1} &= \max_{a \in \mathcal{A}} \left( \mathcal{R^a} + \gamma \mathcal{P^a} v_k \right)
	\end{aligned}$
- *Value iteration이 진행되는 동안, 모든 state의 value function은 어떠한 policy에 대한 value가 아니다.*{: style="color: red"}


<br>


### Summary of DP Algorithms


#### Synchronous Dynamic Programming Algorithms

| Problem | Bellman Equation | Algorithm |
|-|-|-|
| Prediction | Bellman Expectation Equation | Iterative Policy Evaluation |
| Control | Bellman Expectation Equation + Greedy Policy Improvement | Policy Iteration |
| Control | Bellman Optimality Equation | Value Iteration |

- 알고리즘이 state-value function $v_\pi(s)$ 또는 $v_\*(s)$를 사용하는 경우
	- 매 iteration마다 complexity는 $O(mn^2)$으로 매우 크다.
	- $m$: action 수
	- $n$: state 수
- 알고리즘이 action-value function $q_\pi(s, a)$ 또는 $q_\*(s, a)$를 사용하는 경우
	- 매 iteration마다 complexity는 $O(m^2n^2)$으로 더 늘어난다.
- Full sweep하기 때문에 매우 비효율적이다.


---

## Extensions to Dynamic Programming


### Asynchronous Dynamic Programming


#### Asynchronous Dynamic Programming

- 지금까지의 DP 방법론은 모두 synchronous backup을 통해 모든 state들이 parallel하게 backup 되었다.
- 하지만, asynchronous하게도 적용할 수 있다.
	- 특정 state만 선택하거나 순서를 다르게 정해서 backup을 수행한다.
	- Computation을 굉장히 줄일 수 있다.
	- 모든 state가 골고루 선택된다면, 수렴이 보장된다. 즉, 무한 번 뽑을 때, 모든 state들이 무한 번 선택되어야 한다.
	- Full sweep하는 synchronous backup보다 개선된 방법이다. 특별하게 튜닝된 알고리즘이 아닌, practical하게 정말 많이 쓰이는 알고리즘이며 매우 일반적인 방법론이다.
- asynchronous dynamic programming의 간단한 3가지 아이디어
	- In-place dynamic programming
	- Prioritised sweeping
	- Real-time dynamic programming


#### In-place dynamic programming

- 기존 backup을 진행하기 위해서는 $n$개의 state에 대해 최소 2개의 table이 필요한데, 여기서는 단 1개의 table만을 사용해서 backup을 진행한다.
	- Synchronous value iteration
		1. $v_\text{new}(s) \leftarrow \max_{a \in \mathcal{A}} \left( \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} v_\text{old}(s') \right)$
		2. $v_\text{old}(s) \leftarrow v_\text{new}(s)$
	- In-place value iteration
		- $ v(s) \leftarrow \max_{a \in \mathcal{A}} \left( \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} v(s') \right)$
		- 코딩 테크닉에 더 가깝다.
- 실제로 수렴한다.


#### Prioritised Sweeping

- 원래는 어떤 순서로 backup해도 상관 없지만, 여기서는 우선순위를 둬서 중요한 state 먼저 backup을 진행한다.
- 우선순위는 Ballman error가 가장 큰 순서로 정의한다.
	- Bellman error: 이전 step의 table과 현재 step의 table에서 차이값
		- $\left\| \max_{a \in \mathcal{A}} \left( \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^a} v(s') \right) - v(s) \right\|$
	- Priority queue를 이용해서 쉽게 구현할 수 있다.


#### Real-Time Dynamic Programming

- State space가 굉장히 크고 agent가 방문하는 곳은 한정적일 때, agent는 움직이게 두고, agent가 방문한 state를 바로바로 업데이트한다.
	- $v(S_t) \leftarrow \max_{a \in \mathcal{A}} \left( \mathcal{R_{S_t}^a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{\mathcal{S_t}s'}^a} v(s') \right)$


<br>


### Full-width and sample backups


#### Full-Width Backups

- DP는 full-width backups을 사용한다.
- Sync 또는 async 관계 없이, 각 backup마다 현재 state $s$에서 갈수 있는 다음 state $s'$와 action을 참조한다.
- 하지만 이 방법은 큰 문제에 대해서는 적용할 수 없다. 방문할 수 있는 $s'$가 매우 많을 경우, 한 개의 $s$를 계산하기 위해 모든 $s'$을 계산해야 하고, 또 다른 state에서 같은 방식으로 계산하기 때문에, 매우 많은 연산량이 필요해진다*(차원의 저주)*{: style="color: red"}.
	- 심지어 한개 backup마저도 매우 많은 비용이 들 수 있다.


#### Sample Backups
{: style="color: red"}

- 다음 강의에서 고려할 개념이다.
- 장점
	- State의 갯수에 상관없이 고정된 적은 비용이 들기 때문에, 차원의 저주를 깰 수 있다.
	- Model-free인 경우에도 적용할 수 있다.
		- Model-based
			- 어떤 state에서 어떤 action을 선택할 때, 다음에 어떤 state로 이동하는지 알고 있는 상태를 말한다.
			- Environment에 대한 충분한 정보가 담긴 model이 있다.
			- 지금까지 우리는 model-based에서 prediction과 control 문제를 풀고 있었다.
		- Model-free
			- 현재 state에서 어떤 action을 선택할 때, 다음에 어떤 state로 이동하는지 모르는 상태를 말한다.
			- 따라서 action을 통해 state들을 샘플링한다.
			- 한 state에서 100번 action을 선택하면 100개의 state에 도달하게 되고,  이 100개의 샘플로 backup을 한다.
- 따라서 sample backup은 full-width backup에 비해 매우 효율적인 방법이다.
