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
	- 지금의 policy를 따랐을 때, value가 어떻게 되느냐에 따라 값을 구하는 것
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
- q: action value function
- \pi': greedy policy
- 한 step에 대해서는 greedy policy를 따르는게 더 나은 것 처럼, iterative하게 적용하면 모든 step에서 v_\pi'은 v_\pi보다 높을 것이다.
- 모든 state에 대해, \pi'의 value가 \pi 보다 높아서 greedy하게 움직이는 것이 더 많은 value를 가져다 주면서 개선이 일어난다.
- q_\pi(s, \pi'(s)): 첫 step에서 \pi'을 따라가고 이후부터는 \pi를 따라갈 때 action-value function


## Policy Improvement(2)
- 계속 나아지다가 더이상 개선이 이루어 지지 않을 경우에는 bellman optimality equation이 만족하는 상황이기 때문에, 이때의 \pi를 optimal policy라고 할 수 있다.
- local이 아닌 optimal policy를 가진다.


## Modified Policy Iteration
- policy evaluation 단계에서 반드시 v_\pi에 수렴해야 하는가?
- 또는 더 일찍 끝내면 안될까?
- 예를 들어, 정해진 횟수만큼 evaluation을 진행하고, imporvement를 진행하면 안될까?
- 극단적으로 한번 평가하고, 한번 improvement를 하는 것은 안되는가?
- silver: 위의 어떤 경우에도 완전히 합리적인 방법이다.


## Generalised Policy Iteration
- 어떤 알고리즘을 써서든 policy를 평가하면 되고, 어떤 알고리즘을 써서든 improvement하면 된다.


## Principle of Optimality
- 설명할 만큼 이해를 하지 못하였으므로 PASS
- optimal policy는 두 요소로 나뉠 수 있다.
	- 1. 처음 optimal한 action을 선택한다.
	- 2. 그럼 다음 state에서도 optimal policy를 따라간다.
- 어떤 policy \pi는 다음 조건을 만족하면 optimal value를 만족한다.
	- if and only if...
	- s에서부터 도달가능한 모든 state s'에 대해서, \pi가 s'으로부터 optiaml value가 성립하면, v_\pi(s')은 v_*(s')이 된다.


## Deterministic Value Iteration
- subproblems인 s'의 해 v_*(s')를 알면, 한 step lookahead를 통해 v_*(s)를 구할 수 있다.
	- bellman optimality equation: max로 엮여 있다.
- 직관적으로 본다면, 목적지까지의 최단 거리를 구하기 위해, 목적지의 이전 점 들에서 목적지 까지의 촤단 거리를 구하고, 이전 점의 이전 점에서부터 이전 점까지의 최단 거리를 구하는 식으로 볼 수 있다.
- 이미 optimal의 수렴이 보장되었음이 증명되었다고 한다.
- 이전 장과의 차이점:
	- 이전 장에서는 policy가 있어서, policy evaluation, policy improvement을 진행한다.
	- 지금 장에서는 명시적인 policy 없이, value만을 가지고 iteration하게 진행한다.
	- policy iteration에서의 k=1과 value iteration은 동일하다.


## Example: Shortest Path
- Policy가 없다.
- 매 step 마다 bellman optimality equation을 적용
- 간단한 문제들에서는 직접할 수도 있겠지만, 실제 복잡한 문제에서는 끝 주변의 state가 어디인지도 알기 어렵고, 끝이이라는게 있는지도 모르기 때문에 매 step마다 asynchronous하게 full sweep 하게 모든 것을 다 본다.
- 매 step 업데이트 될 때마다, terminate state에 가까운 것들부터 점점 value가 확정된다. 그러면서 가장 멀리 있는 state의 value가 확정된다.
- 이렇게 하면 모든 state에서 value가 구해지고, 이 value를 통해서 policy도 알게 된다.
- iteration이 진행되면서 마치 policy가 업데이트 되는 것 처럼 보이는데, 이는 policy iteration에서의 k=1과 value iteration가 동일하기 떄문이다.
	- 하지만, greedy policy를 구한다고 해도, 다음 step은 greedy policy로 인해 구해지는 value 값이 아니다.


## Value Iteration
- optimal policy를 찾는 문제
- 해법: Bellman optimality backup을 계속 적용
- synchronous backup 이용
	- 각 iteration마다 모든 state에 대해서 v_{k+1}을 업데이트
- policy iteration과 다르게, 여기서는 policy가 없다.
- value iteration이 진행되는 과정 중간에 있는 value functions은 어떤 policy의 value도 아니다.


## Synchronous Dynamic Programming Algorithms
- 알고리즘들이 state-value function에 기반한다면, complexity가 매우 크다. O(mn^2), m: action, n: state
- action-value function일 경우에는 더 늘어난다.
- full sweep하기 떄문에 매우 비효율 적이다.


## Synchronous Dynamic Programming
- 개선할 수 있는 방법들이 있다.
- 특별하게 튜닝된 알고리즘이 아닌 practical하게 정말 많이 쓰이는 알고리즘, 아주 일반적인 방법론이 있다.
- DP 방법론은 모두 synchronous backups을 사용했다. 모든 state들이 parallel하게 backup된다.
- 하지만 Asynchronous하게도 적용할 수 있다.
	- ex. 특정 state만 선택해서 적용, 또는 순서를 다르게 할 경우
	- computational을 굉장히 줄이면서, converge가 보장된다. 대신 이 반복적인 작업을 할때 여러 state가 골고루 뽑혀서 작업되어야 한다. 즉 무한번 뽑으면 모든 경우의 state가 무한번 뽑힐 때
	- 1. In-place dynamic programming
	- 2. Prioritised sweeping
	- 3. Real-time dynamic programming


## In-place dynamic programming
- 코딩 테크닉에 더 가깝다.
- 본래는 n개의 state에 대해서 최소 2개의 table이 필요한데, 여기서는 단 1개의 table만은 사용해서 update를 진행한다.
- 실제로 수렴한다.


## Prioritised sweeping
- 본래는 어떤 순서로 업데이트를 해도 상관이 없지만, 여기서는 우선순위를 두어서 중요한 state를 업데이트 한다. 이때의 중요도는 Bellman error가 가장 큰 순서로 정의된다.
- Bellman error: 이전 step의 table과 다음 step의 table에서 차이값
- priority queue를 이용해서 쉽게 구현 할 수 있다.


## Real-time dynamic programming
- state space가 굉장히 넓고 agent가 가는 곳은 한정적일 때, agent를 움직이게 해 놓고, agent가 방문한 state를 바로바로 업데이트한다.


-----

## Full-Width Bakckups
- 지금까지 한 DP는 full-width backups을 사용한다.
- sync 또는 async든 관계 없이, 각 backup마다 s에서 갈수 있는 모든 s'와 action을 참조해서 업데이트한다.
- 이 방법은 큰 문제에 대해서는 적용할 수 없다. 왜냐하면 방문할 수 있는 s'이 매우 많으면, 그것을 모두 계산해야 겨우 1개의 state를 업데이트 할 수 있는데, 이것을 다른 state에도 적용해야 하기 때문에 매우 많은 계산량이 필요해진다. 즉, 매우 큰 문제의 경우 차원의 저주에 빠지게 된다.(state의 개수가 늘어날 수록 exponentially하게 늘어난다.), 한 backup 조차도 매우 비싸질 수 있다.


## Sample Backups
- 뒤에서 등장하는 내용
- 장점
	- 1. state가 많아져도, 고정된 cost를 들일 수 있다. 비용이 작다. 즉 차원의 저주를 깰 수 있다.
	- 2. Model-free 상황에서도 적용할 수 있다.
		- 지금까지 우리는 Model-based에서 하고 있었다.
		- Model-based: 내가 어떤 state에 있는데, 어떤 action을 하면 어떤 state로 갈지 아는 상황, 즉 모델에 대항 충본한 정보를 알고 있다. 그 상황에서 prediction, control 문제를 풀고 있었다.
		- Model-free: 내가 어떤 action을 하면 어떤 state에 도착할지 모른다. 따라서 action을 통해서 샘플링을 한다. 즉 한 state에서 100번 action을 하면 100개의 state에 도달할텐데, 그 100개의 샘플을 가지고 backup을 한다.
- 따라서 smaple backups에 비해 full-width backup은 매우 비효율 적인 방법이다.
