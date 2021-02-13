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
