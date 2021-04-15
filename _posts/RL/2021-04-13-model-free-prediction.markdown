---
layout: post
title:  "Lecture 4: Model-Free Prediction"
date:   2021-04-13 18:59:05 +0900
category: "Reinforcement Learning"
tags: YouTube
plugins: mathjax
---

David Silver 님의 [Introduction to reinforcement learning](https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) 강의 내용을 [팡요랩 Pang-Yo Lab](https://www.youtube.com/channel/UCwkGvF7xKz2E0Lv-fZ9wv2g)의 [강화학습의 기초 이론](https://youtube.com/playlist?list=PLpRS2w0xWHTcTZyyX8LMmtbcMXpd3s4TU)에서 풀어서 설명하기에, 이를 바탕으로 공부하여 정리한 내용입니다.

- Video: [[강화학습 4강] Model Free Prediction](https://youtu.be/47FyZtBRglI)
- Slide: [Lecture 4: Model-Free Prediction](https://www.davidsilver.uk/wp-content/uploads/2020/03/MC-TD.pdf)


---

## Introduction


### Model-Free Reinforcement Learning

- 지난 강의: Planning by dynamic programming
    - Prediction, control: MDP를 알 때, 푸는 방법론
        Control: DP을 이용한 policy iteration 방법론
- 이번 강의: Model-free prediction
    - MDP를 모를 때, value function을 계산하는 방법
- 다음 강의: Model-free control
    - MDP를 모를 때, value function을 최적화하는 방법


---

## Monte-Carlo Learning


### Monte-Carlo Reinforcement Learning

- *Monte-Carlo*{: style="color: red"}
    - 머신러닝이나 통계학에서 Monte-Carlo라는 단어가 나오면, 직접 계산하기 어려운 문제를 실제로 계속 사건을 실행하면서 나오는 값을 통해 추정하는 것으로 생각하면 된다.*{: style="color: red"} 쉽게 말하면, 그냥 해보고 세어보는 것이라고 생각하면 된다.
    - *Policy는 알고 있으므로, agent는 environment 안에서 policy를 따라 계속 움직인다. 이 때, 게임이 끝날 때 얻을 수 있는 return은 방문한 state마다 매번 달라질 것이다. 이 return을 state에 기록하면서 게임을 계속한 뒤, 각 state들에서 얻은 return의 평균값을 계산하는 방식이 Monte-Carlo이다.*{: style="color: red"}
    - Transition과 reward를 몰라도 게임을 끝까지 하면서, 경험으로부터 직접 계산한다.
    - *Episode가 끝나야 return이 정해진다.*{: style="color: red"}
        - Return: 게임이 끝날 때 까지 얻은 reward들에 대해, discount가 적용된 accumulate sum을 의미한다.
        - *Value: Return들의 평균 값을 말한다.*{: style="color: red"}
        - *Episode가 끝나야만 return을 확정되어 MC 기법을 사용할 수 있다.*{: style="color: red"}


### Monte-Carlo Policy Evaluation(Prediction)

- 목적: Policy $\pi$로 행동해서 얻은 episode들로 $v_\pi$를 배우는 것
    - $S_1, A_1, R_2, \dots, S_k \sim \pi$
- 지난 내용 정리
    - Return: Discounted reward의 총합
        - $G_t = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{T-1} R_T$
    - Value function: Return의 기댓값
        - $v_\pi(s) = \mathbb{E_\pi} [G_t \| S_t = s]$
        - 매 episode를 진행할 때, 같은 state라도 어떤 return을 받을지 다르기 때문에, return은 확률변수이다.
- *Monte-Carlo policy evaluation: Return의 기댓값이 아닌, 실제로 수행해서 얻은 return의 평균값을 사용한다.*{: style="color: red"}


### First-Visit Monte-Carlo Policy Evaluation

- Monte-Carlo에는 first-visit MC와 every-visit MC가 존재한다.
    - First-visit MC
        - State 갯수만큼 빈 테이블이 있고 각 칸마다 return을 누적하기 위한 공간과 방문 count를 기록할 공간을 만들어 놓는다. Agent가 state를 방문할 때마다 해당 state의 테이블 위치에 return과 count를 기록한다.
        - Episode에서 처음 방문할 때만 count와 return을 기록한 뒤, return의 평균을 계산한다.
            - $S(s) \leftarrow S(s) + G_t$
            - $N(s) \leftarrow N(s) + 1$
            - $V(s) = S(s) / N(s)$
        - 큰 수의 법칙에 의해 $N$이 무한으로 가면 $V(s)$는 $v_\pi(s)$로 수렴한다.
            - $V(s) \rightarrow v_\pi(s) \  \text{as} \  N(s) \rightarrow \infty$
    - Every-visit MC
        - Episode에서 한 state를 여러 번 방문할 수도 있는데, 매 방문마다 count와 return을 기록한 뒤, return의 평균을 계산한다.


### Every-Visit Monte-Carlo Policy Evaluation

- Episode에서 한 state를 여러 번 방문할 수도 있는데, 매 방문마다 count와 return을 기록한다.
- First-visit MC와 every-visit MC 중 어느 것을 사용하던 상관없고, 결과는 같다. 서로 다른 구현방법이 있다고 이해하면 된다.
- MC policy evaluation을 사용하기 위한 조건: *Policy는 골고루 움직이는 정책을 가지고 있어서, 모든 state를 방문한다는 가정이 있어야 한다.*{: style="color: red"}
    - $N(s) \rightarrow \infty$ 일 때, 어떤 state를 방문하지 않으면, 이 state는 갱신되지 않는다.
    - 모든 state를 평가하는 것이 목적이기 때문에, 어떤 state를 방문하지 않으면, 이 state의 가치를 구하기 위해 참고하는 return을 얻을 수 없다.


### Blackjack Example


#### Blackjack Example

- 처음 카드 2장을 받은 뒤 카드를 추가로 더 받거나 그만 받을 수 있지만, 최종적으로 카드의 합이 21에 가까운 사람이 이기는 게임이다.
- 딜러는 2장 중 1장만 open하고, 나는 그 카드만 보고 카드를 추가할지 여부를 판단한다.
- 카드의 합이 21을 넘어가면 무조건 패배한다.
- State: 3개
    - 현재 내 카드의 합: 12 ~ 21
        - 12보다 작은 카드를 2장 받으면, 자동으로 한 장을 더 받는다.
    - 딜러가 open한 카드: ace ~ 10
    - 내가 사용할 수 있는 Ace 수: Yes / No
        - Ace는 1 또는 10으로, 유리한 방향으로 셀 수 있다.
- Action: 2개
    - Stick: 카드를 더 받지 않는다
    - Twist: 카드를 한장 더 받는다.
- Reward
    - Stick인 경우
        - +1: 카드 합 > 딜러의 카드 합
        - 0: 카드 합 = 딜러의 카드 합
        - -1: 카드 합 < 딜러의 카드 합
    - Twist인 경우
        - -1: 크다 합이 21보다 클 때
        - 0: 그 외
- Transitions: 카드 합이 12보다 작을 때, 자동으로 twist한다.


#### Blackjack Value Function after Monte-Carlo Learning

![Blackjack Value function after Monte-Carlo Learning](/assets/rl/blackjack_value_function.png)

- 내 카드의 합이 20 이상일 경우는 stick하고, 아니면 twist하는 policy를 수행할 때의 value function 결과이다.
- Player의 카드 합이 21에 가까울 수록, value function이 높다는 것을 알 수 있다.
- 500,000번 에피소드를 진행하고 난 뒤에야, 10,000번 에피소드의 value function이 아직 학습이 덜 되었다는 것을 알 수 있다.


<br>


### Incremental Monte-Carlo


#### Incremental Mean

- 이전 MC에서는 누적된 return 값에 방문 횟수로 나눠서 평균을 계산했지만, 조금 다른 방식으로 계산할 수도 있다.
    - $\begin{aligned}
            \mu_k &= \frac{1}{k} \sum_{j=1}^k x_j   \newline
                  &= \frac{1}{k} \left( x_k + \sum_{j=1}^{k-1} x_j \right)  \newline
                  &= \frac{1}{k} \left( x_k + (k - 1) \mu_{k-1} \right)     \newline
                  &= \mu_{k-1} + \frac{1}{k} \left( x_k - \mu_{k-1} \right)
    \end{aligned}$
    - Agent가 어떤 state를 10,000번 방문했다면, 10,000개의 return값을 갖고 있다가 평균을 낼 수도 있지만, 위와 같이 새로운 return이 들어올 때마다 이미 계산된 평균값과 방문 횟수만으로 업데이트할 수 있다.


#### Incremental Monte-Carlo Update

- $\begin{aligned}
        N(S_t) &\leftarrow N(S_t) + 1   \newline
        V(S_t) &\leftarrow V(S_t) + \frac{1}{N(S_t)} \left( G_t - V(S_t) \right)
\end{aligned}$
    - 이전에 계산된 value와 새로 얻은 return이 있을 때, 이 차이(error term)만큼 value를 return 방향으로 보정해준다고 생각할 수 있다.
- *Non-stationary problem인 경우*{: style="color: red"}
    - $1/N(S_t)$을 $\alpha$로 고정할 수 있다.
        - 오래된 기억들은 잊는다는 의미로, 예전 경험들을 잊게 된다.
        - 즉 과거는 잊고, 최근의 경험들로만 value를 계산할 때 사용한다.
        - *Temporal-Difference Learning에서도 사용하는 개념이다.*{: style="color: red"}
    - Non-stationary problem: Environment의 MDP가 조금씩 바뀌는 상황을 말한다.


---

## Temporal-Difference Learning


### Temporal-Difference Learning

- 경험으로부터 직접 학습하며, Model-free이므로 MDP에 대한 정보가 없어도 된다.
- *TD와 MC의 차이점*{: style="color: red"}
    - TD: Episode가 끝나지 않아도 학습할 수 있다.
    - MC: Episode가 끝나야만 학습할 수 있다.
- TD updates a guess towards a guess
    - 추측을 추측을 업데이트 한다.


### MC and TD
{: style="color: red"}

- Incremental every-visit Monte-Carlo
    - $V(S_t) \leftarrow V(S_t) + \alpha(G_t - V(S_t))$
        - $V(S_t)$가 $G_t$ 방향으로 업데이트 된다.
        - 즉, 실제 실현된 현재의 정확한 값으로 업데이트 한다.
- Simplest temporla-difference learning algorithm: TD(0)
    - $V(S_t) \leftarrow V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$
        - $V(S_t)$: 현재 state $S_t$가 앞으로 얼만큼의 return을 받을지를 추측한 값
        - $V(S_{t+1})$: 1-step 이후, 다음 state $S_{t+1}$의 추측 value
        - 1-step 이후의 예측값이 더 정확하므로, $V(S_t)$를 $R_{t+1} + \gamma V(S_{t+1})$ 방향으로 업데이트 한다.
        - 즉 1-step 이후의 예측값으로 현재의 예측값을 업데이트 한다.
    - *TD target: $R_{t+1} + \gamma V(S_{t+1})$*{: style="color: red"}
        - 1-step만큼 가서 예측하여 현실이 더 반영되기 때문에, 이전 예측값보다 더 정확할 것이다.
    - *TD error: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$*{: style="color: red"}
- Ex: 차량을 운전하는데, 실수로 중앙선을 침범했다고 하자. 맞은편 트럭이 달려오다가, 트럭 운전자의 반응속도가 좋아서 비켜 지나가고, 나는 다시 원래 차선으로 돌아왔다고 하자.
    - MC의 경우: 현재 나는 충돌하지 않았으므로, 방금 상황에 대한 reward -1억을 얻지 못한다.
    - TD의 경우: 중앙선을 침범하면 다음 step의 상황에서 내가 죽을 수도 있다는 것을 예측 할 수 있기 때문에, 그 추측값과 reward를 사용해서 업데이트한 뒤 다시 중앙선으로 돌아온다.
- TD 의미: 순간적인 시간 차이라는 의미이다.


### Driving Home Example


#### Driving Home Example

![Driving Home](/assets/rl/driving_home.png)


#### Driving Home Example: MC vs. TD

![Driving Home: MC vs. TD](/assets/rl/driving_home_mc_td.png)

- 도착할 때까지 걸린 시간이 총 43분일 때,
    - MC: 지나온 state는 43분을 기준으로 업데이트 한다.
    - TD: 매 step마다 걸린 시간 + 다음 step의 추측값을 기준으로 현재 step의 state를 업데이트 한다.

#### Advantages and Disadvantages of MC vs. TD

- TD
    - Final outcome을 알기 전에 학습 할 수 있다.
    - Continuing(non-terminating)한 environment에서 학습할 수 있다.
- MC
    - 에피소드가 끝날 때까지 기다렸다가, 끝난 뒤에 return을 알게 되면 이 return을 기준으로 업데이트 한다.
    - Episodic(terminating)한 environment에서 학습할 수 있다.


#### Bias/Variance Trade-Off

- Return $G_t = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{T-1} R_T$는 $v_\pi(S_t)$의 편향되지 않은 예측값이다.
- True TD target $R_{t+1} + \gamma V_\pi(S_{t+1})$ 또한 $v_\pi(S_t)$의 편향되지 않은 예측값이다.
    - 전지전능한 신이 $V_\pi(S_{t+1})$의 실제 값을 알려주면, $R_{t+1} + \gamma V_\pi(S_{t+1})$는 편향되지 않기 때문이다.(Bellman equation)
- TD target $R_{t+1} + \gamma V_\pi(S_{t+1})$는 $v_\pi(S_t)$의 편향된 예측값이다.
    - 현실에서는 $V_\pi(S_{t+1})$의 실제 값을 알 수 없기 때문이다.
    - 따라서 100만번 반복적으로 수행해서 계산해도, $v_\pi(S_t)$가 되리라는 보장은 없다.
    - *Bias 관점에서 TD는 안좋지만, variance 관점에서 TD target은 return보다 훨씬 낮다.*{: style="color: red"}
        - Variance: 통계에서의 분산을 의미한다. 어떤 확률 변수에서 평균으로부터 얼마나 퍼져있는지를 나타내는 척도이다.
            - Ex: 평균이 0이고 variance가 큰 Gaussian 분포는 매우 튀는 값이 샘플링 된다. 우리는 평균 0을 학습하려고 할 때, 샘플은 1000, -1300과 같이 매우 흔들리는 값이 나오겠지만 이는 unbiased estimate 샘플이기 때문에 1억번 샘플링해서 평균을 계산하면 0이 나온다.
                - 하지만, 샘플 평균이 -150이므로, 답은 0인데 -150을 기준으로 업데이트가 될 것이다.
            - EX: 평균이 0.5이고, variance가 1인 Gaussian 분포는 0.5 주변에 모여있는 어떤 종 모양이 되는데, 이 때, 샘플링을 해서 얻은 값이 3, -1, 2이라고 하자. 우리는 평균 0을 학습하고 싶은데, 샘플 평균 1.3으로 업데이트가 되겠지만, 흔들리지 않고 더 학습 될 것이다.
                - 즉, *bias가 적다고 능사는 아니고, variance도 고려해야 한다.*{: style="color: red"}
- Return: 현재 step부터 게임이 끝날 때 까지 정말 다양한 return을 얻는다. Environment의 랜덤성, state transition probability, stochastic policy 등 여러가지 랜덤성으로 인해, 게임을 하면 할수록 전혀 다른 return을 얻게 되어, variance가 커진다.
- TD target: 1-step 사이의 랜덤성은 게임이 끝날때 까지의 랜덤성보다 작기 때문에, variance가 적지만 biased한 예측값을 얻게 된다.


#### Advantages and Disadvantages of MC vs. TD (2)

- MC: Variance가 크지만, bias는 0이다.
    - 수렴성이 좋다. 테이블에 적을 수 없을 만큼 큰 문제의 경우, function approximation(Neural net)을 활용하게 되는데, function approximation에서도 수렴성이 좋다.
    - 실제값을 이용해서 업데이트하므로, 초기값에 민감하지 않다.
    - 이해하기 쉽고, 사용하기 편하다.
- TD: Variance가 낮지만, bias는 약간 존재한다.
    - 보통 MC보다 더 효과적이다.
    - TD(0)는 $v_\pi(s)$에 수렴한다. 하지만, function approximation에서 항상 수렴성이 좋지는 않다.
    - Initial value를 이용해서 $v_\pi(S)$를 업데이트하기 때문에, 초기값에 민감하다.
- Bias가 있으므로 결국 틀린 값으로 수렴한다는 뜻인데, 그럼 이것은 동작할 수 있는 알고리즘일까?
    - 다행히 동작은 한다.


<br>


### Random Walk Example


#### Random Walk Example

![Random Walk](/assets/rl/random_walk.png)

- 0으로 초기화하였다.
- TD를 사용했을 때, 100번째에서 true value로 수렴한다.


#### Random Walk: MC vs. TD

![Random Walk: MC vs. TD](/assets/rl/random_walk_mc_td.png)

- $\alpha$ 값에 대해, MC와 TD를 비교하였다.
- 에피소드가 진행되면서, 실제값과 추정값의 차이를 RMS 지표로 구했다.
- MC는 완만히 줄어들고 있고, TD는 줄어들다가 다시 늘어나기도 한다.
    - TD가 다시 늘어나는 것은 $\alpha$가 너무 크기 때문이다.


<br>


### Batch MC and TD


#### Batch MC and TD

- 에피소드를 무한 번 샘플링할 수 있다면, MC와 TD의 $V(s)$는 $v_\pi(s)$로 수렴한다.
- 만약 $k$개의 제한된 에피소드가 있을 때, MC와 TD는 같은 곳으로 수렴할까?
    - $\begin{aligned}
            s_1^1, & a_1^1, r_2^1, \dots, s_{T_1}^1   \newline
            & \vdots  \newline
            s_1^k, & a_1^k, r_2^k, \dots, s_{T_1}^k
    \end{aligned}$


#### AB Example

- State A B가 있고, 8개의 에피소드가 아래와 같이 있다고 하자.
    1. A, 0, B, 0
    2. B, 1
    3. B, 1
    4. B, 1
    5. B, 1
    6. B, 1
    7. B, 1
    8. B, 0
- Model-free이기 때문에, 왜 이렇게 되는지는 알 수 없다.
- 이 경험들을 통해 MC와 TD를 할 수 있다.
- $V(B) = 3/4 = 0.75$
- $V(A) = ?$


#### AB Example

![AB Example](/assets/rl/ab_mdp.png)

- 8개의 에피소드를 통해, MDP를 추측해볼 수 있다.
- MC: $V(A) = 0$
    - State A를 한번 방문했고, 그 때의 return은 0이다.
- TD: $V(A) = 0.75$
    - State A에서 reward 0과 $V(B)$의 value로 $V(A)$를 업데이트한다.


#### Advantages and Disadvantages of MC vs. TD (3)

- TD: Markov property를 이용하여 value를 추측한다.
    - Markov environment에서 더 효과적이다.
    - Markov model의 likelihood를 maximize한다.
- MC: Markov property를 버리고, MSE를 minimize한다.


<br>


### Unified View


#### Monte-Carlo Backup

![Monte-Carlo Backup](/assets/rl/monte_carlo_backup.png)

- $V(S_t) \leftarrow V(S_t) + \alpha(G_t - V(S_t))$
- Monte-Carlo backup: 어떤 state $S_t$에서 시작해서 에피소드가 끝날 때 얻은 $G_t$로 $V(S_t)$를 업데이트한다.


#### Temporal-Difference Backup

![Temporal-Difference Backup](/assets/rl/temporal_difference_backup.png)

- $V(S_t) \leftarrow V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$
- Temporal-difference backup: Reward와 1-step 뒤, 다음 state의 예측치를 사용해서 $G_t$를 대체한다. Bootstrapping이라고도 한다.
    - Bootstrapping: 끝까지 확인하지 않고, 추측치로부터 추측하는 것을 말한다.
        - 다른 분야에서도 쓰이는 용어이다.
        - Ex: Random forest 등


#### Dynamic Programming Backup

![Dynamic Programming Backup](/assets/rl/dynamic_programming_backup.png)

- $V(S_t) \leftarrow \mathbb{E_\pi} [R_{t+1} + \gamma V(S_{t+1})]$
- Dynamic programming backup: 샘플링하지 않고, 1-step뒤, 모든 state의 value를 기준으로 업데이트한다. 끝까지 확인하지 않고, full sweep한다.


#### Bootstrapping and Sampling

- Bootstrapping: 추측치로부터 추측치를 업데이트한다.
    - MC X: 에피소드가 완전히 끝난 뒤에 업데이트하기 때문이다.
    - DP, TD O: 깊이 관점에서, 1-step만 확인하고 멈추기 때문이다.
- Sampling: Full sweep하지 않고, 방문한 샘플들을 통해 업데이트한다.
    - MC, TD O: 너비 관점에서, agent가 policy를 따라 움직여서 방문한 샘플들을 갖고 업데이트하기 때문이다.
    - DP X: 한 state에서 가능한 모든 action을 선택하기 때문이다.


#### Unified View of Reinforcement Learning

![Unified View of Reinforcement Learning](/assets/rl/unified_view_of_reinforcement_learning.png)

| | Bootstrapping | Sampling |
|:-:|:-:|:-:|
| MC | X (Deep backups) | O (Sample backups)
| TD | O (Shallow backups) | O (Sample backups)
| DP | O (Shallow backups) | X (Full backups)

- Exhaustive search: 모든 것을 할 수 있는 트리를 만들어서 찾는 방법이다.
    - 학습방법이라고 하기에도 조금 애매하고, 비용이 너무 많이 든다.
- Model-based인 경우, full backup이 가능하므로 DP를 쓸 수 있다.


---

## TD($\lambda$)


### $n$-Step TD


#### $n$-Step Prediction

![$n$-Step Prediction](/assets/rl/n_step_prediction.png)

- TD의 변형들로, 현실을 1-step 또는 2-step만큼 반영하여 bootstrapping을 할 수 있다.


#### $n$-Step Return

- $n$-step TD return: $G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$
    - $n$만큼 reward를 사용하고, $S_{t+n}$부터는 $V(S_{t+n})$을 사용한다.
- $n$-step temporal-difference learning: $V(S_t) \leftarrow V(S_t) + \alpha \left( G_t^{(n)} - V(S_t) \right)$
    - $n$-step TD error: $G_t^{(n)} - V(S_t)$


#### Large Random Walk Example

![Large Random Walk Example](/assets/rl/large_random_walk.png)

- 1-step TD ~ 1000-step TD까지의 성능을 비교한 그래프이다.
    - y축: RMS error
    - x축: $\alpha$
- 그래프가 아래로 갈수록 좋은데, 1-step TD이 제일 좋지도 않고, 1000-step TD이 제일 좋지도 않다.
- 3-step TD 또는 5-step TD가 좋은 것을 확인할 수 있다.
    - 즉, TD(0)와 MC 사이에 sweet spot이 존재하므로, 적당히 잘 정해주어야 한다.
    - 에피소드를 10개로 제한했기 때문에, MC return의 높은 variance로 인해 성능이 좋지 않다.


#### Averaging $n$-Step Returns

![Averaging $n$-Step Returns](/assets/rl/averaging_n_step_returns.png)

- 2-step과 4-step return의 평균을 예측치로 사용할 수 있다.
    - Ex: $\frac{1}{2}G^{(2)} + \frac{1}{2}G^{(4)}$


<br>


### Forward View of TD($\lambda$)


#### $\lambda$-return

![$\lambda$-return](/assets/rl/lambda_return.png)

- $G_t^\lambda = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$
    - Geometric mean: $1 - \lambda, (1 - \lambda)\lambda, (1 - \lambda)\lambda^2, (1 - \lambda)\lambda^3, \dots, (1 - \lambda)\lambda^{n-1}$의 무한급수 결과가 1이다.
- TD($\lambda$): TD(0)에서 MC까지 return의 geometric mean을 예측치로 사용한다.
    - 논문에서도 많이 쓰이는 방법이다.
- Forward-view TD($\lambda$)와 Backward-view TD($\lambda$)가 있다.
    - Forward-view TD($\lambda$): $V(S_t) \leftarrow V(S_t) + \alpha \left( G_t^\lambda - V(S_t) \right)$


#### TD($\lambda$) Weighting Function

![TD($\lambda$) Weighting Function](/assets/rl/td_lambda_weighting_function.png)

- 시간 $t$가 클수록 가중치가 geometric하게 줄어든다.
- Geomatric Mean을 사용하는 이유: 메모리를 확장시키지 않으면서, 계산을 편하게 할 수 있기 때문이다.
    - 즉, TD(0)과 같은 비용으로 TD($\lambda$)를 계산할 수 있다.


#### Forward-view TD($\lambda$)

![Forward-view TD($\lambda$)](/assets/rl/forward_view_td_lambda.png)

- 미래를 보고 $\lambda$-return을 통해 value function을 업데이트 한다.
- MC와 마찬가지로 게임이 끝나야만 $G_t^\lambda$를 계산할 수 있기 때문에, TD(0)의 장점이 사라진다.


#### Forward-view TD($\lambda$) on Large Random Walk

![Forward-view TD($\lambda$) on Large Random Walk](/assets/rl/large_random_walk_forward_view_td_lambda.png)

- $\lambda$ 값에 따라 결과가 다르다.


<br>


### Backward View of TD($\lambda$)


#### Backward View TD($\lambda$)

- 에피소드가 끝나지 않아도, 매 step마다 업데이트할 수 있다.


#### Eligibility Traces
{: style="color: red"}

![Eligibility Traces](/assets/rl/eligibility_traces.png)

- 어떤 사건이 일어났을 때 책임을 물어서, 책임이 큰 state를 많이 업데이트 하는 방법이다.
- Ex: Bell / Bell / Bell / Light / Shock
    - 전기 충격이 발생한 책음을 가장 최근에 일어난 Light에 책임을 물을 수도 있고, 또는 가장 빈번하게 발생한 Bell에 물을 수 있을 것이다.
- 책임을 나누는 기준
    1. Frequency heuristic: 빈번하게 일어난 state에게 책임을 많이 묻는다.
    2. Recency heuristic: 최근이 발생한 state에게 책임을 많이 묻는다.
- Eligibility trace: 책임을 묻기위해 위의 2가지 기준을 합친 것이다.
    - $\begin{aligned}
        E_0(s) &= 0  \newline
        E_t(s) &= \gamma \lambda E_{t-1}(s) + \mathbb{1}(S_t = s)
    \end{aligned}$
        - 모든 state는 각각 eligibility trace 값을 가지고 있다.
        - Frequency heuristic: 매 시점마다 해당 state를 방문할 때, 1을 더한다.
        - Recency heuristic: 매 시점마다 해당 state를 방문하지 않을 때, $\gamma$를 곱한다.
            - Ex: $\gamma = 0.9$
    - 자주 방문하거나 최근에 방문할 수록 값이 커진다.


#### Backward View TD($\lambda$)
{: style="color: red"}

![Backward View TD($\lambda$)](/assets/rl/backward_view_td_lambda.png)

- TD(0)의 장점과 TD($\lambda$)의 장점을 모두 가진다.
- $\begin{aligned}
    \delta_t &= R_{t+1} + \gamma V(S_{t+1}) - V(S_t)    \newline
    V(s) &\leftarrow V(s) + \alpha \delta_t E_t(s)
\end{aligned}$
    - $E_t(s)$: 시점 $t$일 때, state $s$의 eligibility trace 값
    - $\delta_t$: TD(0)의 error 값
    - 해당 state의 책임이 얼마나 큰지 follow up해서, 이 eligibility trace 값을 곱해서 업데이트하면, TD($\lambda$)와 수학적으로 동일한 의미를 가진다.
- 과거의 책임 소재를 기록해오기 때문에, 매 step마다 업데이트할 수 있다.
- TD($\lambda$)로 backward view TD($\lambda$)를 사용한다.


<br>


### Relationship Between Forward and Backward TD


#### TD($\lambda$) and TD(0)

- Eligibility trace 식에서 $\lambda = 0$일 때, TD(0)와 정확히 같다.
    -  $\begin{aligned}
        E_t(s) &= \mathbb{1}(S_t = s)   \newline
        V(s) &\leftarrow V(s) + \alpha \delta_t E_t(s)
    \end{aligned}$
    - TD(0): $V(S_t) \leftarrow V(S_t) + \alpha \delta_t$


#### TD($\lambda$) and MC

- Episodic environment에서 Offline 업데이트할 경우, eligibility trace 식에서 $\lambda = 1$일 때, MC와 동일하다.
- Offline 업데이트할 경우, TD($\lambda$)는 forward-view TD($\lambda$)와 backward-view TD($\lambda$)는 같다.
    - $\sum_{t=1}^T \alpha \delta_t E_t(s) = \sum_{t=1}^T \alpha \left( G_t^\lambda - V(S_t) \right) \mathbb{1}(S_t = s)$
    - 최근에는 online 업데이트에서도 같다는 연구결과가 있다고 한다.
    - Online update: Agent가 학습하면서 움직이는 것을 말한다.
        - 1-step이 끝나면, 이 1-step 경험으로 학습한 뒤 움직인다.
    - Offline update: 에피소드가 끝난 뒤, 학습하는 것을 말한다.
