---
layout: post
title:  "Planning by Dynamic Programming"
date:   2021-02-12 00:00:00 +0900
category: "Reinforcement Learning"
tags: YouTube
plugins: mathjax
---

David Silver 님의 [Introduction to reinforcement learning](https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) 강의 내용을 [팡요랩 Pang-Yo Lab](https://www.youtube.com/channel/UCwkGvF7xKz2E0Lv-fZ9wv2g)의 [강화학습의 기초 이론](https://youtube.com/playlist?list=PLpRS2w0xWHTcTZyyX8LMmtbcMXpd3s4TU)에서 풀어서 설명하기에, 이를 바탕으로 공부하여 정리한 내용입니다.

- video: [[강화학습 3강] Planning by Dynamic Programming](https://youtu.be/rrTxOkbHj-M)
- slide: [Lecture 3: Planning by Dynamic Programming](https://www.davidsilver.uk/wp-content/uploads/2020/03/DP.pdf)


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
		- MDP $< \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma >$와 policy $\pi$ 또는 MRP $< \mathcal{S}, \mathcal{P^\pi}, \mathcal{R^\pi}, \gamma >$ 로부터 value function $v_\pi$를 찾는다.
		- 최적의 policy인지 아닌지는 상관없이 어떤 바보같은 policy라도, MDP안에서 어떤 state에서 시작해서 끝날때 까지 얼만큼의 return을 받을지 예측한다.
	- Control
		- MDP $< \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma >$ 로부터 optimal value function $v_\*$와 optimal policy $\pi_\*$를 찾는다.


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
	- 모든 state에 대해 매 iteration마다 업데이트한다. (Full sweep)
	- 현재 step에 있는 value를 계산 할 때, 다음 step의 value를 이용하여 조금씩 더 정확하게 만든다.
		- 처음에는 모든 state의 value function $v_1$을 임의의 값 또는 0으로 초기화한다.
		- 한 번 iterative한 방법을 사용해서 $v_2$를 만들고, 다시 한 번 반복하여 $v_2$에서 $v_3$를 만든다. 이것을 계속 반복하면, $v_\pi$에 수렴한다.
		- 이 때, $v_\pi$는 policy $\pi$에 대한 value function이다.
	- 점화식은 현재 state를 이용해서 다음 state를 갱신하므로, 점화식과는 다른 방식이다.
- Asynchronous backups: 뒤에서 설명


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
	- Greedy: 다음 state들 중 가장 높은 value를 가지는 state로 움직이는 것을 의미한다.
	- 즉, 모든 문제들에 대해 greedy하게 움직이는 policy를 찾기만 해도 더 나은 policy를 찾을 수 있다.


#### Iterative Policy Evaluation in Small Gridworld (2)

![Iterative Policy Evaluation](/assets/rl/small_gridworld_3.png)

- Policy evaluation 과정에서 단 3번의 iteration으로 optimal policy를 찾게 되었다.

