---
layout: post
title:  "Value Function Approximation"
date:   2021-03-02 02:00:00 +0900
category: "Reinforcement Learning"
tags: YouTube
---

## 참고

- video
    - [[강화학습 6강] Value Function Approximation](https://youtu.be/71nH1BUjhNw)
- slide
	- [Lecture 6: Value Function Approximation](https://www.davidsilver.uk/wp-content/uploads/2020/03/FA.pdf)


## Large-Scale Reinforcement Learning
- table lookup 방법은 각 칸에 해당하는 table을 만들어 놓고 초기화하는데, 현실의 문제를 푼다고 할 때, state의 갯수는 얼마나 될까?
    - ex. Backgammon: 10^{20}개
    - ex. 바둑: 10^{170}개
    - ex. 헬리콥터: 빈 칸을 만들 수도 없이 연속적인 상태 공간을 갖는다.(무한)
- 그러면 이런 문제에서 어떻게 강화학습을 동작시킬까?
- Model-free method에서 어떻게 scale up 할까?


## Value Function Approximation
- 지금까지는 lookup table 기반이어서, state-value function V는 state s 갯수만큼 빈 칸이 있었고, state-action value function Q는 모든 state-action pair 갯수 만큼 빈 칸이 있었다.
- 말도안되게 큰 state를 메모리에 담을 수 없고, 담더라도 연산상 너무 느리다.
- 따라서 커다란 MDP에 대해서는 function approximation을 사용한다.
- function approximation:
    - V_\pi(s), Q_\pi(s, a)를 실제 value function 이라고 하고, \hat{v}를 v_\pi를 모방하는 function approximaton이라고 하자.
    - \hat{v}(s, w)
        - w: \hat{v}함수 안에 들어있는 파라미터
    - 실제 V와 Q를 학습하는 것이 목적이다.
    - 내가 봤던 state부터 안 본 state까지 generalized 하는 것
        - generalized: 안 본 state에 대해서도 올바른 value와 맞도록 output을 내준다.
    - 신경망 등
- \hat{v}은 w에 의해서 다른 함수가 되므로, 학습한다는 건 결국 w를 업데이트 한다는 것이다. w를 업데이트 하는데 MC 또는 TD를 사용한다.


## Types of Value Function Approximation
- 네모: Blackbox라고 생각
    - w: internal parameters
- value function을 모방
    - 첫 번째 네모: 이 함수로 s를 query로 던지면, 블랙박스 내부에서 w라는 파라미터들이 관장헤서 \hat{v}가 출력된다.
- q value function 모방: 2가지 방법
    - 두 번째 네모: s, a를 query로 던지면, value를 리턴하는 방식
        - action in 형태의 Q
    - 세 번째 네모: s만 넣어주면, s에서 할 수 있는 모든 action들에 대해서 여러개의 output을 내주는 형태로 함수를 만들 수 있다.
        - action out 형태의 Q


## Which Function Approximator?
- 모방하는 함수는 어떤 걸 쓸수 있을까?
    1. Linear combination of features: 각 feature들마다 가중치 w가 있어서 가중 합을 구하는 방식, 선형 방식
    2. Neural network: 비선형 방식
    3. Decision tree
    4. Nearest neighbour
    5. Fourier / wavelet bases


## Which Function Approximator?
- 꼭 미분 가능할 필요는 없지만, 우리는 이 중에서 미분가능한 function approximator을 사용할 것이다.
    1. Linear combinations of features
    2. Neural Network
- 미분 가능해야 gradient를 구해서 update를 업데이트 할 수 있다.
- non-stationary, non-iid data
    - 모분포가 계속 바뀌면서 서로 independent하지 않다.
    - 즉 앞에서 본 데이터와 뒤에 있는 데이터가 서로 연관있기 때문에 그런 점에도 잘 맞는 학습 방법을 사용해야 한다.


## Gradient Descent
- input을 w로 가지는 J함수가 있다.
    - w: 1차원일 필요 x, 일반적인 n차원 벡터
- J라는 함수를 최소화하는 input w를 찾고자 할때, gradient descent를 사용한다.
    - J라는 함수가 convex하면 local minimum은 global minimum이다.
- J라는 함수에 대한 곡면을 그릴 때, 점 x, y를 조금씩 움직여서 J의 output을 더 낮아지도록, 최저점으로 향하게 한다.
    - 반드시 global한 최저점은 아니고, local한 최저점이다. 그 부근에서 제일 작은 점을 향한다.
- J를 w에 대해 gradient(편미분)를 구하면 벡터의 갯수 n개의 항 만큼 값이 나오는데, 이 값들이 벡터를 이루면서 크기와 방향을 나타낸다. 이때 이 방향은 J가 제일 빠르게 변하는 방향이 되므로, -방향으로 step size \alpha 만큼 아주 작게 움직인다. 그리고 다시 그 점에서 gradient를 구하고 다시 조금 움직이고, 그 동작을 반복한다. 그렇게 그 부근에서 제일 작은 방향으로 가는 방법을 경사하강법이라고 한다.
- 1/2은 수학적 편의를 위해 넣어둔 것이다.


## Value Function Approx. By Stochastic Gradient Descent
- 우리의 목적은 value function을 잘 학습하는 것이 목적이다. 즉 \hat{v}이 v를 잘 모방하는 것이 목적이다.
    - \hat{v}와 v의 차이가 적다.
    - 모든 것을 아는 신이 v을 알고 있다고 가정하고, 실제 어떤 state s일때 s에서 실제 value v와 모방한 value \hat{v}을 빼서 제곱하고 기댓값을 구한다.
        - 제곱: 양과 음의 차이를 똑같은 차이라고 계산하기 위함
        - 기댓값: policy \pi를 따랐을 때의 s에 대한 차이를 봐야하기 떄문
    - 우리가 하고 싶은 것은 J를 줄이는 방향으로 w를 업데이트 해줘야 한다.
        - J: loss
        - V_\pi(S): 상수값
- Stochastic gradient descent: expectation을 빼는 방법
    - policy \pi를 따라가면 방문했던 state들의 샘플들이 나오는데, 그것을 input으로 넣어준다는 의미이다.
    - 샘플들을 취해서, 그것을 바탕으로 업데이트를 하는 것을 말한다.
- 만약 policy \pi가 1번을 자주 방문한다고 하면, 1번 샘플이 많이 뽑힐 것이고, 7번을 덜 방문하면, 7번은 적게 뽑히기 떄문에 여러번 시행하면 자연스럽게 expection과 같아지게 된다.
    - 기댓값 갱신은 전체 gradient 갱신과 같아지게 된다.
- 일반적인 function approximator를 이용


## Feature Vectors
- Feature: state s가 있다면 state  s에서 n개의 feature을 만들 수 있다.
    - 사람이 만들어 준다.
    - ex. 주식시장: 현재 state를 넣으면, 이동평균선, 전고가, 볼린져밴드 등의 feature을 만든다.
- state는 실제 문제에서는 추상적인 개념인데, 이를 숫자 몇개로 표현하기 위한 것
    - state를 잘 표현해 주는 feature를 생성해야 할 것이다.


## Linear Value Function Approximation
- Linear value function을 이용한 approximator \hat{v}의 예시
    - \hat{v}는 사실 충분히 flexible한 함수는 아니다. 왜냐하면 첫번째 feature의 제곱에 비례할 수도 있는데, 여기서는 선형합으로 구성되어 있다.
    - 사실 이렇게 표현되지 않으면 에러가 크겠지만, 그래도 상관없다. 단지 우리는 그런 function approximation을 갖고 최선을 다하면 되는 것이기 때문에, flexibility가 부족해서 정확히 fitting은 되지 않을 수도 있다.
    - 우리는 선형으로 모방 함수를 구성해서 처음보다 나은 w를 찾아간다. 
- \hat{v} : feature들을 이용해서, 각 feature들에 대해 w를 곱해서 합한다.
    - x1 feature에 w1 곱하고, x2 feature에 w2 곱하고, ...
    - 내적의 결과가 value
- 처음에는 이상한 값들이 나올 텐데, 그 값들을 실제값들에 근사하게 w를 수정해나간다.
- Objective function(목적함수)는 true value와 \hat{v}의 차의 제곱
- Stochastic gradient descent는 global optimum으로 수렴하게 된다.
    - 왜냐하면 모방함수가 선형함수이기 떄문에 최저점이 하나밖에 없기 때문이다.
- update 규칙: 이전 슬라이드와 마찬가지로 진행한다.
    - update = step size x 틀린 정도(error) x feature value
- value function을 학습하려고 하는데, 모방 함수를 사용하려고 하고, 모방함수의 linear combination을 넣었을 때 어떻게 되는지를 보는 중이다.


## Table Lookup Features
- table lookup과 linear value function이 전혀 다른 2개의 방법이라고 생각하지 않았으면 좋겠다.
    - table lookup은 linear value function의 한 예시이다.
    - 예를 들어 feature를 state 갯수만큼 만들고, 첫번째 feture은 s_1을 방문했을 때 1인 feature vector을 만들고, 이 vector을 w와 내적하면 \hat{v}를 만들 수 있다. 즉 w_1 ~ w_n이 table에 있던 각 칸의 값인 것이다.


## Incremental Prediction Algorithms
- 지금까지는 true value function을 신이 알려주었다고 가정하고 식을 전개해나갔다.
- 하지만 강화학습에는 실제 value가 몇이 되는지 알려줄 supervisor가 없고, reward signal만 있을 뿐이다.
- 우리는 v_\pi(s) 자리에 MC와 TD를 끼워 넣을 것이다.
    - MC: true value function 자리에 G_t를 넣는다.
        - 모방 함수가 state s에 대해서 G_t를 output으로 밷어주기를 바라는 것
        - 같은 state s라고 해도 G_t가 10을 받을수도 20을 받을 수도 있을 텐데, 만약 10과 20밖에 없다면 중간인 15가 될 것이다. 그런 식으로 어떤 샘플들을 이용해서 w를 업데이트한다.
        - 신이 알려주는 true value가 없어서, MC가 알고 있는 return을 넣어주었다.
        - 마치 state s에 대해서 G_t가 나와야 한다는 supervised learning이라고도 볼 수 있다. 그럼 그 차이를 이용해서 그 차이를 줄이는 방향으로 업데이트 한다.
    - TD(0): 현재 step에서 내 state의 예측치는 한 step을 더 간다음에 보상과 그 state의 예측치의 합으로 TD target이 구성된다. 이 TD target을 v_\pi에 대신 끼워 넣는다.
    - TD(\lambda): 지금 step부터 마지막 step까지 (1-\lambda)\lambda^n의 가중평균을 사용한다. forward-view TD(\lambda), backward-view TD(\lambda)가 있다. backward-view TD(\lambda)는 eligibility traces라는 책임을 묻는 값을 사용한다. 이렇게 구해진 \lambda-return을 대신 사용한다.
- reward: 우리가 풀고자 하는 문제의 목적을 어떻게 정의할지를 나타낸다.
- MDP가 정의되어야 MDP를 풀 수 있다.
    - MDP의 구성요소
        - state space
        - action space
        - reward
        - \gamma 등


## Monte-Carlo with Value Function Approximation
- Return G_t: reward의 discounted accumulative sum, 지금 시점부터 받는 reward를 discount시켜서 모두 더한 것
    - value function의 unbiased estimation, 왜냐하면 value function의 정의가 return의 기댓값이기 때문이다.(return이 매번 다르지만)
    - policy, environment, 에피소드의 샘플링등 stochastic하기 때문에 어떻게 샘플링되느냐에 따라 return이 매번 다른데, 그렇게 충분히 많이 샘플링하면 평균은 결국 true value으로 갈 것이기 때문이다.
    - 따라서 G_t를 기준으로 업데이트를 해도 된다.
- MC는 local optimum으로 수렴함이 증명되었다. 심지어 수렴조건이 더 까다로운 nonlinear function approximation을 써도 잘 수렴한다.


## TD Learning with Value Function Approximation
- Linear TD(0)는 global optimum에 가깝게 수렴한다.
- 로이드라는 사람이 논문으로 증명했다고 한다.
- MC와 TD는 수렴성이 다른다.
    - MC는 unbiased estimation이고, TD(0)는 우리가 추측하는 방향으로 추측치를 업데이트하는 것이기 때문에 꼭 맞아야 한다는 보장은 없지만 Linear TD(0)를 쓰면 global optimum에 가깝게 수렴한다는 것이 증명됬다.
- Q: TD-target을 쓸때 true value function부분에도 estimate가 들어간다. 그러면 이대로 미분해도 되는건가? target도 w에 대한 함수인데, 이대로 전개해도 괜찮은 것인가?
    - A: 우리가 업데이트를 할 때 시간 축에서 한 방향을 보고 업데이트를 해야 한다. 미래를 보고 한 step을 더 경험한 것이 정확하다는 의미로 한 방향으로 향해야 하는데, target value도 같이 업데이트를 하게 되면, 한 step에서 더 가서 보는 것과 과거에서 미래를 보는 것이 섞이게 된다. 그렇게 해서 업데이트를 하려면 잘 할 수 있는데, 그것을 이해하려면 관련 배경들을 완전히 이해해야 한다. 한 스텝 더 가서 본 것으로 현재를 업데이트하는 것으로 미래를 향해서 업데이트 한다고 이해하도록 하자. target value도 미분하게 되면 시간을 역행하는 개념이 들어가게 된다.


## TD(\lambda) with Value Function Approximation
- TD(\lambda)도 똑같이 넣을 수 있다.
- Forward-view, backward-view를 각각 넣어서 쓸 수 잇다.


## TD(\lambda) with Value Function Approximation
- Forward-view, backward-view의 linear TD(\lambda)는 동등하다.


## Control with Value Function Approximation
- 지금까지는 prediction에 관한 내용이었다. (value function 학습)
- 지금부터는 control에 관한 내용
    - Policy evaluation: Approximation policy evaluation
        - Q도 마찬가지로 V처럼 학습할 수 있다.
    - Policy improvement: \epsilon-greedy policy improvement


## Action-Value Function Approximation
- 실제 value function q가 있다고 할 때, \hat{q}으로 모방하는 방법
    - mean-squared error를 줄이는 방향으로 w를 업데이트한다.


## Linear Action-Value Function Approximation
- 똑같이 feature을 만들면, 그 feature을 바탕으로 linear의 합을 만들 수 있고, 그것을 넣음으로써 계산한다.


## Incremental Control Algorithm


## Linear Sarsa with Coarse Coding in Mountain Car
- Model-free, function approximation을 써야하는 큰 문제에서 최적의 policy를 찾는 방법론을 배웠다.
    - policy iteration을 번갈아가면서 사용
        - evaluation: Q를 이용한 approximator을 넣음(MC, TD(0))
        - improvement: \epsilon-greedy
        - 이런 방법을 linear sarsa라고 한다. linear가 붙는 이유는 linear function approximation을 썼기 때문이다.
- Mountain Car
    - state: position 1개, velocity
    - action: 정해진 세기로 앞으로 힘을 주거나, 뒤로 힘을 주거나, 힘을 주지 않거나 3가지
    - reward: -1
    - 끝에 도달하면 종료
- state-value function을 나타낸 그래프


## Linear Sarsa with Radial Basis Functions in Mountain Car
- 최종 학습되었을 때의 state-valud function 그래프


## Study of \lambda: Should We Bootstrap?
- TD(0), TD(\lambda)가 꼭 필요한가?
- y축:  error
- x축: \lambda
    - \lambda = 1: MC 방법론
    - \lambda = 0: TD(0) 방법론
- MC가 제일 안좋고, TD(0)이 MC보다는 모두 좋다.
- 0과 1사이에 sweet spot이 존재한다.
- 결론: bootstraping이 필요하다.
    - return 값만 구하면, variance가 너무 커서 실제 문제에서는 잘 안되기 때문에 bootstrapping을 쓰는 것이 좋다.


## Baird's Counterexample
- TD(0)가 항상 수렴하는가?
    - TD(0)가 항상 수렴한다고 보장할 수 없었다.


## Parameter Divergence in Baird's Counterexample
- 이런 상황이 되면 value function이 발산한다.
- TD(0)가 항상 수렴하지 않는다는 반례를 보여주는 것이다.
- 즉 bootstrapping이 좋지만 항상 수렴하지는 않는다는 뜻이다.
    - 수렴성이 보장되어 있지는 않아도, 실제로는(practical) 잘 된다.


## Convergence of Prediction Algorithms
- non-linear도 이전에 사용한 linear 방식을 그대로 사용할 수 있다.
- non-linear로 갈 수록 수렴성이 안좋다.
- MC에서 TD(\lambda)로 갈 수록 수렴성이 안좋다.
- Off-policy로 갈 수록 수렴성이 안좋다.
- 이 테이블은 어디까지나 이론적인 이야기고 실제에서는 신기하게도 잘 된다.


## Gradient Temporal-Difference Learning
- Gradient TD: Bellman error의 true gradient를 따라가기 때문에 non-linear에서도 수렴성이 좋고, off-policy에 대해서도 수렴성이 좋다.


## Convergence of Control Algorithms
- control일 때도 수렴성이 좋다.


## Batch Reinforcement Learning
- 지금까지 봤던 incremental method는 gradient descent를 이용해서 샘플 1개를 봤을 때, 샘플 1개로 업데이트를 하고, 그것으로 policy를 업데이트했다.
    - 이 방법은 sample efficient 하지 않다.
        - transition 1번 한 것, 즉, s에서 a를 선택해서 reward r을 받고, s'에 도착한 것이 샘플 1개이다. 이 transition 1번 한 것으로 1번 업데이트하고, 그 경험은 버려지기 때문에 샘플이 효과적으로 쓰여지지 않는다. (1번쓰고 버려지기 때문)
- Batch method: training data처럼 이미 agent가 쌓은 경험들이 있는데, 이 경험들을 반복해서 쓰면서 학습을 하는 것


## Least Squares Prediction
- 주어진 경험들이 있고, \hat{v}가 v_\pi(s)와 같아지기 바란다.
- 앞에서 한 incremental method는 경험을 버리는데, batch는 경험을 쌓아놓고 하기 때문에 off-policy에 쓸 수 있는 이야기를 하는 것 같다.
- experience D: state, value의 쌍들의 묶음
    - 이것을 활용해서 \hat{v}을 v_\pi(s)에 fitting 시키고 싶다.
    - incremental method에서는 expection에 \pi가 있었지만, 여기서는 주어진 데이터에 대해서 수행하기 때문에 D에 대한 기댓값이 들어간다.
- 마찬가지로 제곱 에러를 계산해서 least square 방법론으로 찾으려 한다.


## Stochastic Gradient Descent with Experience Replay
- D가 있으면 여기서 샘플링을 한다.
    - state와 value를 샘플링해서 그것으로 stochastic gradient descent를 적용한다.
    - 데이터가 1000개 있으면 1000개 중 10개를 샘플링해서 업데이트를 하고 그 작업을 계속 반복한다. 그렇게 10만번을 하게 되면 한 샘플이 여러번 쓰일 것이다. 그렇게 좀더 데이터를 효율적으로 쓸 수 있다.
    - 이것을 Experience replay라고 한다. 즉 경험을 재사용한다.
    - off-policy 방법을 할때 강화학습에서 굉장히 많이 사용하는 방법이다.
    - on-policy에서는 쓸 수 없다.
- 이렇게 해서 업데이트를 하면 least square에 수렴한다.
    - least square: 제곱의 차이를 줄이는 것으로 수렴한다.


## Experience Replay in Deep Q-Networks(DQN)
- off-policy에서 non-linear function을 쓰면 수렴성이 보장되지 않았다. 그냥 naive하게 실행시키면 수렴하지 않고 발산하는데, 이것을 잘 수렴하기 위해서 2가지 트릭을 사용한다.
    1. Experience replay
        - \epsilon-greedy policy로 게임을 하는데, s에서 a를 선택하면 reward를 받고 다음 s'로 간다는 transition이 있는데, 이것을 replay memory에 쌓는다.
        - 거기에서 랜덤하게 mini-batch로 뽑아서 그것으로 학습한다.
        - 학습은 least square을 줄이는 것이다.(TD target과 q의 제곱을 줄이는 것)
    2. Fixed Q-targets
        - TD target을 계산할 때, 예전 버전의 parameter을 쓰는 것이다.
        - parameter는 2 set이 있다. 하나는 고정된 예전 버전, 나머지 하나는 현재 계속 업데이트되고 있는 얘가 있다.
        - 왜 이런 트릭이 도입되었냐하면, 업데이트할때마다 parameter의 방향에 계속 바뀌는데, non-linear function에서는 수렴을 하기가 더 어려워진다. 그래서 1000번 정도 parameter를 고정시키고 업데이트를 하다가, 다시 반대로 고정시켜서 1000번 정도 parameter를 업데이트 시키는 방식으로 parameter set 두 개를 관리한다.
        - w_i^-: 예전 버전, w_i: 최신 버전
        - w_i를 업데이트 하는 건데, TD target은 w_i^-가 쓰인다.
        - 즉 target을 고정시킨다.


## DQN in Atari
- 최신 4장의 게임 화면이 들어오고, convolutional layer를 거쳐서 최종 output은 각 action 갯수만큼 Q 함수를 출력
- reward는 score의 변화량
- 모든 게임에 대해서 hyperparameter와 네트워크 아키텍처를 고정시키고 학습만 돌렸다.


## DQN Results in Atari
- Atari의 여러 종류의 게임이 있어서, linear combination을 이용한 예전 방법론과 비교했을 때, 대부분의 게임에서 다 이겼다.
- 가운데 선이 있는데, 왼쪽은 사람보다 더 잘하고, 오른쪽은 사람보다 더 못하다.
- 수렴성이 안좋은 데서 수렴을 시키도록 학습을 잘되는 것을 도입했다.
    - 수렴성이 좋지 않은 이유: neural net이라는 non-linear function이 들어갔고, off-policy로 학습했고, MC가 아닌 TD를 사용했다.


## How much does DQN help?
- Replay memory와 fixed-Q를 사용했다.
- 이 둘을 사용하지 않았을 때와 비교해서 5개의 게임에 대해서 평균 점수를 나타냈다.
    - 아무것도 안했을 때: 3점
    - Fixed-Q: 점수가 향상된다.
    - Replay만 들어갔을 때: 벽돌깨기에서 거의 80배가 늘었다.
    - Replay와 fixed-Q를 썼을때: 가장 좋다.
