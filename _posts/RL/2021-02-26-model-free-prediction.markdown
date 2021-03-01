---
layout: post
title:  "Model Free Prediction"
date:   2021-02-26 00:00:00 +0900
category: "Reinforcement Learning"
tags: YouTube
---

## 참고

- video
    - [[강화학습 4강] Model Free Prediction](https://youtu.be/47FyZtBRglI)
- slide
	-[Lecture 4: Model-Free Prediction](https://www.davidsilver.uk/wp-content/uploads/2020/03/MC-TD.pdf)


## Model-Free Prediction
environment를 모를때(환경의 transition을 모르고, 환경의 reward 함수를 모를때), model-free prediction을 풀기 위한 방법론


## Model-Free Reinforcement Learning
- 지난 강의: MDP를 알떄 푸는 방법(prediction, control)
	- control: Dynamic programming을 이용한 policy iteration 방법론


## Monte-Carlo Reinforcement Learning
- MC: 머신러닝을 하던, 통계학을 하던 monte-carlo 라는 단어가 나오면, 직접 구하기 어려운 것을 실제적으로 계속 사건을 실행하면서 나오는 실제 값들을 통해서 추정하는 것
	- 쉽게 말하면 그냥 해보고 세보는 것
- policy는 정해져 있으므로, agent는 환경 안에서 policy를 따라 움직이면서 계속 시도해본다. 그러면 어떤 state에서 게임이 끝났을데 얻을 수 있는 value가 매번 다를텐데, 게임을 계속 해보면서 얻은 return들을 적어 놓고 평균을 내는 것을 MC이다.
- transition, reward를 몰라도 게임을 끝까지 하면서 경험으로부터 직접 배운다.
- 에피소드가 끝나야 return이 정해진다.
	- return: 게임이 끝날 떄 까지 얻은 discount가 적용된 reward들의 accumulate sum이다.
	- 그리고 그런 return들을 평균 낸 것이 value
- 단점: 에피소드가 끝나야지만, return이 확정되므로 MC 기법을 적용할 수 있다.


## Monte-Carlo Policy Evaluation
- Policy evaluation == prediction
- 목적: v_\pi를 배우는 것
	- return: discounted reward의 총계
	- value function: return의 기댓값
		- return은 확률 변수이다.
		- 매번 에피소드를 진행할 때 마다 다르기 어떤 return을 받을지 다르기 떄문
- Monte-Carlo policy evaluation: 실제적으로 수행한 return의 평균을 사용한다.


## First-Visit Monte-Carlo Policy Evaluation
- Monte-Carlo에는 first-visit MC와 every-visit MC가 존재
- state의 갯수만 큼 빈 칸이 있을 때, agent가 해당 state에 방문할 때마다 해당 빈 칸을 하나씩 늘려주고, 게임이 끝 날떄의 return을 그 칸에 적어준다.
	- 이 때, 한 에피소드에서 한 state를 여러번 방문할 수 있는데, 여러번 방문 하는 것을 모두 인정하는 것을 every-visit MC
	- 처음 방문 한 것만 인정하는 것을 first-visit MC
- 에피소드가 처음 방문할 떄만 count를 올려주고, 그 때만 return을 더해주고 평균을 낸다.
- 큰 수의 법칙에 의해 n이 무한으로 가면 V(s)는 V_\pi(s)로 수렴한다.

## Every-Visit Monte-Carlo Policy Evaluation
- state를 방문할 떄마다 count를 올려주는 것만 다를 뿐 똑같은 아이디어
- 둘 중 어떤 것을 사용하던 상관 없다고 한다. 결과는 같다.
- (단지 서로 다른 구현이 있다고 이해하면 될 것 같다.)
- 두개 방법을 사용하기 위한 조건: policy가 골고루 움직이는 정책을 가지기 떄문에, 모든 state를 다 방문한다는 가정이 있어야 한다.
	- 왜냐하면 N(s)는 무한으로 가야하기 떄문에, 어떤 state가 방문하지 않으면 그 state가 갱신되지 않는다.
	- 그리고 우리는 모든 state를 평가하는 것이 목적이기 떄문에 방문하지 않으면, 그 state의 가치를 구하기 위해 참고할 수 있는 통계치가 없기 떄문이다.


## Balckjack Example
- 카드 2장을 받고, 내가 더 받을 수 있고 멈출 수 있지만 최종적으로 21에 가까운 사람이 이기는 게임
- 딜러와 나의 대결
- 딜러는 2장 중 1장만 Open하며, 그 카드를 보고 판단한다.
- 21을 넘어가면 무조건 패배한다.
- state 3개
	- 현재 내 카드의 합
		- 12보다 작은 카드를 2장 받으면, 자동으로 한장을 더 받는다.
	- 딜러가 보여주는 카드
	- 내가 쓸 수 있는 Ace 수
		- Ace는 1 또는 10으로 유리한 쪽으로 셀 수 있다.
- action 2개
	- stick: 카드를 더 받지 않음, 딜러보다 큰 수일 경우 승리
	- twist: 카드를 한장 더 받음, 21보다 큰 수일 경우 패배


## Balckjack Value Function after Monte-Carlo Learning
- policy: 내 카드가 20 이상일 경우 stick, 아닐 경우 twist
	- 이 policy 대로 수행했을 때의 value function 결과
- player의 합계가 21에 가까울 수록 value function이 높음을 알 수 있다.
- 학습이 덜 되었다는 것은 500,000번 에피소드를 진행하고난 결과와 비교했을 때 알 수 있다.


## Incremental Mean
- 평균을 조금 다른 방식으로 쓸 수 있다.
- 이 전까지의 평균과 현재의 평균을 나눠서 표현할 수 있다.
- MC를 생각해보면 각 state를 방문했을 때의 평균값을 저장하고 있어야 할텐데, 만약 10,000번 방문했다면, 10,000개의 값을 가지고 있다가 평균을 낼 수도 있지만, 위와 같이 새로운 값이 나올 때 마다 가지고 있는 값을 가지고 보정할 수 잇다.


## Incremental Monte-Carlo Updates
- 이전 값이 있고 지금 얻은 값이 있을 때, 그 차이의 error term만큼 이전 값을 보정해준다고 볼 수 있다.
- 1/N(S_t)를 \alpha로 고정할 수 있다.
	- 오래된 기억들은 잊을 것이라는 의미
	- 예전에 했던 경험들은 잊는다.
	- 이것을 사용하는 때는 non-stationary problem에서 좋을 수 있다.
		- non-stationary problem: MDP가 조금씩 바뀌는 상황
	- 과거는 잊고 최신의 것들로만 구성하고 싶을 때 사용
	- TD에서도 사용하므로 기억!
- error를 줄이도록 매번 G_t를 향해 조금씩 조금씩 움직이도록 V(S_t)를 업데이트한다.
