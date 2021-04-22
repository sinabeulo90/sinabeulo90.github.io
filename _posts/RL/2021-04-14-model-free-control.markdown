---
layout: post
title:  "Lecture 5: Model-Free Control"
date:   2021-04-14 18:25:55 +0900
category: "Reinforcement Learning"
tags: YouTube
plugins: mathjax
---

David Silver 님의 [Introduction to reinforcement learning](https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) 강의 내용을 [팡요랩 Pang-Yo Lab](https://www.youtube.com/channel/UCwkGvF7xKz2E0Lv-fZ9wv2g)의 [강화학습의 기초 이론](https://youtube.com/playlist?list=PLpRS2w0xWHTcTZyyX8LMmtbcMXpd3s4TU)에서 풀어서 설명하기에, 이를 바탕으로 공부하여 정리한 내용입니다.

- Video: [[강화학습 5강] Model Free Control](https://youtu.be/2h-FD3e1YgQ)
- Slide: [Lecture 5: Model-Free Control](https://www.davidsilver.uk/wp-content/uploads/2020/03/control.pdf)


---

## Introduction

### Uses of Model-Free Control

- MDP로 모델링할 수 있는 몇 가지 문제들이 있다.
    - 엘리베이터 알고리즘
    - 로봇 축구
    - 포토폴리오 등
- 이 문제들은 MDP를 몰라서 샘플을 통해서 경험을 얻거나, 알고 있다하더라도 너무 큰 문제이기 때문에 샘플을 사용해야만 하는 문제들이다.
- 이 문제들을 model-free control로 풀 수 있다.


### On and Off-Policy Learning
{: style="color: red"}

- Policy에는 최적화하고자하는 policy와 environment에서 경험으로 쌓는 behavior policy가 있다.
    - On-policy: 위의 두 policy가 같을 경우를 말한다.
    - Off-policy: 위의 두 policy가 다를 경우를 말한다.
- On-policy learning: 최적화하고자 하는 policy $\pi$가 있고, 그 policy $\pi$로부터 어떤 행동을 선택하고 경험을 쌓아서, 그 경험을 통해 배우는 방법이다.
- Off-policy learning: 최적화하고자 하는 policy $\pi$가 아닌 다른 agent가 행동한 behavior policy $\mu$로부터 어떤 행동을 선택하고 경험을 쌓아서, 그 경험으로부터 배우는 방법이다.
    - 전혀 다른 policy를 사용한다.


---

## On-Policy Monte-Carlo Control


### Generalised Policy Iteration


#### [Generalised Policy Iteration (Refresher)]({% post_url /RL/2021-04-12-planning-by-dynamic-programming %}#generalised-policy-iteration)

- Policy iteration: 최적의 policy를 찾는 control의 방법론이다.
- 2단계로 번갈아 가면서 반복된다.
    1. Policy evaluation: Policy가 고정되어 있을 때, 해당 policy를 평가하여 $v_\pi$를 계산한다.
    2. Policy improvement: 평가된 $v_\pi$를 바탕으로 더 좋은 policy로 향상시킨다.
        - Greedy policy improvement: 평가된 $v_\pi$를 바탕으로 greedy하게 움직이는 policy를 만든다.
        

#### Generalised Policy Iteration With Monte-Carlo Evaluation

- Policy evaluation: Monte-Carlo policy evaluation, $V = v_\pi$?
    - MC는 model-free에서도 evaluation할 수 있다.
- Policy improvement: Greedy policy improvement?
- 이렇게 하면 control 문제가 풀린 것일까? 안된다.
    - Policy imrpovement에서 $v_\pi$를 통해 policy를 향상시킨다는 것은, 현재 state가 있고 다음 state가 무엇인지 알 때, 다음 state의 $v_\pi(s')$ 중 가장 큰 값을 갖는 state로 가겠다는 뜻인데, 이는 MDP를 알아야만 가능하기 때문이다.
    - *Model-free일 때는 MDP를 몰라서 agent가 직접 방문하지 않고는 다음 state를 알 수 없고, state transition probability도 모르기 때문에, 지금 알고 있는 $V$만을 가지고 greedy하게 policy improvement 할 수 없다.*{: style="color: red"}


#### Model-Free Policy Iteration Using Action-Value Function

- State-value function에서는 model을 알아야 $V$에 대해 greedy policy improvement 할 수 있다.
    - $\pi'(s) = \arg\max_{a \in \mathcal{A}} \mathcal{R_s^a} + \mathcal{P_{ss'}^a} V(s')$
        - $V$: 각 state의 가치
- *그럼 action-value function에서 policy evaluation 했을 때, $Q$에 대해 greedy policy improvement 할 수 있을까?*{: style="color: red"}
    - $\pi'(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)$
        - $Q$: 각 (state, action)의 가치
    - Action-value function에서 어떤 state에서 action을 선택할 때 return을 구할 수 있고, 이 return의 평균이 $Q$ value이기 때문에 MC를 사용할 수 있다.
    - 어떤 state에서 선택할 수 있는 action 수는 알 수 있으므로, 가장 높은 $Q$ 값을 가지는 action을 선택하는 policy를 새로운 policy로 만들면 된다.


#### Generalised Policy Iteration with Action-Value Function

![Generalised Policy Iteration with Action-Value Function](/assets/rl/generalised_policy_iteration_with_action_value_function.png)

- Policy evaluation: MC를 이용해서 $Q$를 평가한다.
- Policy improvement: Greedy policy improvement?
    - *Greedy policy improvement 하기 위해서는 모든 state를 충분히 방문해야 한다.*{: style="color: red"}
    - 만약 모든 state를 방문하지 않고 greedy하게만 움직이면, 많은 곳을 충분히 방문할 수 없게 되어 어딘가 막힐 수 있다.


<br>


### Exploration


#### Example of Greedy Action Selection

- Greedy action 선택의 문제점
    - 2개의 문이 있다고 하자. 왼쪽 문을 열었을 때 reward 0을 받았고, 오른쪽 문을 열었을 때 reward 1을 받았을 때, greedy action 선택을 한다면 계속 오른쪽 문을 열 것이다.
    - 이 때, 정말 오른쪽 문이 제일 좋은 문이라고 확신할 수 있을까? 왼쪽 문을 한번 더 열었을 때 reward 100을 받았을 수도 있을 것이다.


#### $\epsilon$-Greedy Exploration

- $\pi(a \| s) = \begin{cases}
                    \epsilon / m + 1 - \epsilon \  & \text{if} \  a^\* = argmax_{a \in \mathcal{A}} Q(s, a) \newline
                    \epsilon / m                   & \text{otherwise}
                \end{cases}$
    - $m$: action의 갯수
    - Policy improvement 할 때, $\epsilon$ 확률로 랜덤하게 다른 행동을 선택한다.
        - $\epsilon$ 확률이 5%일 경우, 95% 확률로 가장 좋은 행동을 선택한다.
- 장점
    - 모든 action에 대해 exploration이 보장된다.
    - $1 - \epsilon$의 확률로 가장 좋은 행동을 선택하므로, policy 향상을 보장한다.


#### $\epsilon$-Greedy Policy Improvement

- $\begin{aligned}
        q_\pi(s, \pi'(s)) &= \sum_{a \in \mathcal{a}} \pi'(a \| s) q_\pi(s, a)  \newline
                          &= \epsilon / m \sum_{a \in \mathcal{A}} q_\pi(s, a) + (1 - \epsilon) \max_{a \in \mathcal{A}} q_\pi(s, a)    \newline
                          & \geq \epsilon / m \sum_{a \in \mathcal{A}} q_\pi(s, a) + (1 - \epsilon) \sum_{a \in \mathcal{A}} \frac{\pi(a \| s) - \epsilon / m}{1 - \epsilon} q_\pi(s, a)    \newline
                          &= \sum_{a \in \mathcal{A}} \pi(a \| s) q_\pi(s, a) = v_\pi(s)
\end{aligned}$
    - $\epsilon$-Greedy를 사용하면, policy improvement 된다.
    - $q_\pi(s, \pi'(s))$: 어떤 state에서, 처음에는 $\pi'$에 따라 action을 선택하고, 다음부터는 $\pi$를 따랐을 때의 value


#### Monte-Carlo Policy Iteration
{: style="color: red"}

- Policy evaluation: MC를 이용해서 $Q$를 평가한다.
- Policy improvement: Greedy policy improvement 대신, $\epsilon$-greedy policy improvement 한다.
- *그럼 여기서 조금 더 효율적으로 할 수 있는 방법은 없을까?*{: style="color: red"}


#### Monte-Carlo Control

![Monte-Carlo Control](/assets/rl/monte_carlo_control.png)

- MC는 에피소드 단위로 평가가 진행되기 때문에, 한 에피소드 만으로 policy evaluation 단계와 policy improvement 단계를 거친다.
- 한 에피소드의 경험을 갖고 policy를 개선할 수 있기 때문에, Policy evaluation 단계에서 완전히 수렴하도록 evaluation하지 않고 바로 policy improvement 단계로 넘어가도 되지 않을까?


<br>


### GLIE


#### GLIE

- GLIE property: Policy가 빠르게 수렴할 수 있도록 하는 성질
    1. Exploration: 모든 state-action pair들이 무한히 방문해야 한다.
        - $\lim_{k \rightarrow \infty} N_k(s, a) = \infty$
    2. Exploitation: $\epsilon$이 5% 인 경우, 학습을 무한히 해서 제일 좋은 policy를 찾더라도 항상 5% 확률로 바보같은 행동을 선택할 것이다. 그렇게 되면 이 policy는 최적의 policy가 아니게 되므로, greedy-policy로 수렴하도록 해야한다.
        - $\lim_{k \rightarrow \infty} \pi_k(a \| s) = \mathbb{1} \left( a = \arg\max_{a' \in \mathcal{A}} Q_k(s, a') \right)$
- $\epsilon = 1/k$로 할 경우, GLIE를 만족한다.


#### GLIE Monte-Carlo Control

- Evaluation 단계
    - $\begin{aligned}
        N(S_t, A_t) &\leftarrow N(S_t, A_t) + 1 \newline
        Q(S_t, A_t) &\leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)} \left( G_t - Q(S_t, A_t) \right)
    \end{aligned}$
        - $N(S_t, A_t)$: $S_t$에서 $A_t$를 선택한 갯수
        - $Q(S_t, A_t)$에서 $1/N(S_t, A_t)$ 대신 $\alpha$를 사용할 수 있다.
- Improvement 단계
    - $\begin{aligned}
        \epsilon &\leftarrow 1/k    \newline
        \pi &\leftarrow \epsilon-\text{greedy}(Q)
    \end{aligned}$
- GLIE Monte-Carlo control을 하면, $Q$가 optimal action-value function으로 수렴한다.
    - $Q(s, a) \rightarrow q_\*(s, a)$


<br>


### Blackjack Example


#### Back to the Blackjack Example

- [Blackjack Example]({% post_url /RL/2021-04-13-model-free-prediction %}#blackjack-example)은 [Lecture 4: Model-Free Prediction]({% post_url /RL/2021-04-13-model-free-prediction %})에서 evaluation의 예시로 사용되었다.
- 여기서는 같은 예시로 control을 수행한다.


#### Monte-Carlo Control in Blackjack

![Monte-Carlo Control in Blackjack](/assets/rl/monte_carlo_control_in_blackjack.png)

- GLIE Monte-Carlo control을 통해 optimal policy와 optimal value의 결과이다.
    - Ace 여부에 따라 2개 종류로 나뉘어져 있다.
    - state: (Ace 여부, 딜러가 보여준 카드, 내 카드의 합(11~21))


---

## On-Policy Temporal-Difference Learning


### MC vs. TD Control

- Monte-Carlo를 사용하는 위치에 TD를 쓸 수 있을까? 가능하다.
- TD
    - Variance가 낮다.
    - Online 학습이 가능하다.
    - 에피소드가 끝나지 않아도 학습할 수 있다.


### Sarsa($\lambda$)


#### Updating Action-Value Functions with Sarsa

![Sarsa](/assets/rl/sarsa.png)

- $Q(S, A) \leftarrow Q(S, A) + \alpha \left( R + \gamma Q(S', A') - Q(S, A) \right)$
    - 1-step action을 선택하여 reward를 받으면, 그 값으로 $Q(s, a)$를 업데이트한다.
    - $\alpha$: 얼만큼 업데이트할 것인지에 대한 지표
    - $R + \gamma Q(S', A')$: TD target
        - $\gamma Q(S', A')$: 1-step 앞의 추측치
    - $R + \gamma Q(S', A') - Q(S, A)$: TD error


#### On-Policy Control With Sarsa

- Policy evaluation: Sarsa를 이용해서 $Q$를 평가한다.
- Policy improvement: Greedy policy improvement 대신, $\epsilon$-greedy policy improvement를 사용한다.
- *한 에피소드마다 진행하는 대신, 1-step마다 진행한다. 즉, 더 짧은 간격을 iteration을 수행한다.*{: style="color: red"}


#### Sarsa Algorithm for On-Policy Control

![Sarsa Algorithm](/assets/rl/sarsa_algorithm.png)

1. $Q(s, a)$ lookup table을 임의로 Initialize한다.
2. 첫 state $S$에서 $\epsilon$-greedy policy를 따르는 action을 선택한다.
    - 높은 확률로 가장 $Q(s, a)$를 갖는 action이 선택되거나, 낮은 확률로 무작위로 선택된다.
3. Reward와 다음 state $s'$을 관측한다.
4. $s'$에서 $\epsilon$-greedy policy를 따르는 action을 선택한다.
5. $Q(s', a')$ 추측치를 통해 $Q(s, a)$를 업데이트한다.
6. 다시 2를 반복한다.


#### Convergence of Sarsa

- Sarsa를 사용하면, $Q$는 optimal action-value function으로 수렴한다.
    - $Q(s, a) \rightarrow q_\*(s, a)$
- 조건
    1. GLIE sequence of policies $\pi_t(a \| s)$: GLIE를 만족해야 한다.
        - 모든 state-action pair를 방문하면서, $\epsilon$이 감소하여 greedy policy로 수렴해야 한다.
    2. Robbins-Monro sequence of step-sizes $\alpha_t$
        - step-size $\alpha$가 아래 조건을 만족해야 한다.
            - $\sum_{t=1}^\infty \alpha_t = \infty$
                - $\alpha$를 무한히 더하면, 무한이 되어야한다.
                - $Q$가 매우 큰 값이 될 수 있도록 step-size가 충분히 커야 한다는 의미이다.
                - Ex: 실제 $Q$가 1억일 때, 0으로 초기화되더라도 1억까지 업데이트 할 수 있어야 한다.
            - $\sum_{t=1}^\infty \alpha_t^2 < \infty$
                - $Q$가 업데이트되는 정도가 점점 작아져서, 결국에는 수렴할 수 있을 정도로 작아야 한다는 의미이다.
            - $\alpha$: 얼만큼 업데이트 할지를 나타내는 값
- 실제 사용할 때, 위 2개 조건들을 고민하지 않아도 잘 수렴한다.
- 이론적으로, 수렴성을 만족하기 위해 이 조건들을 만족해야 한다는 정도로 알아두면 좋을 듯 하다.


#### Windy Gridworld Example

![Windy Gridworld](/assets/rl/windy_gridworld.png)

- S에서 G까지 가는 최적의 길찾기 문제이다.
- x축: 바람 세기
    - 0: 바람 x
    - 1: Agent를 위로 한 칸 올려준다.
    - 2: Agent를 위로 두 칸 올려준다.
- 1-step 움직일 때마다 reward는 -1을 얻으며, discount factor은 사용하지 않는다.


#### Sarsa on the Windy Gridworld

![Sarsa on the Windy Gridworld](/assets/rl/sarsa_on_the_windy_gridworld.png)

- x축: 총 업데이트한 횟수
- y축: G에 도달한 횟수
- $Q$ 테이블 수: 4(행동 수) x 10(가로) x 7(세로) = 280
- 약 2000번 움직이고서야 첫 에피소드가 끝났다.
    - 처음에는 랜덤하게 바보같이 움직이기 때문이다.
- 우연히 한 번 도달하기 시작하면, 정보가 전파되면서 다음 번 에피소드부터 훨씬 적은 time-step이 소모된다.
    - 한번 도달하는 순간 reward가 발생하고, 그 정보가 bootstrapping으로 전파되어 목적지를 더 잘 찾아 간다.


#### $n$-Step Sarsa

- $n$-step TD: $n$까지의 reward를 얻고서 bootstrapping을 한다.
    - $\begin{aligned}
        n &= 1 \  \text{(Sarsa)} & q_t^{(1)} &= R_{t+1} + \gamma Q(S_{t+1})                       \newline
        n &= 2                   & q_t^{(2)} &= R_{t+1} + \gamma R_{t+2} + \gamma^2 Q(S_{t+2})    \newline
          &\vdots                &           & \vdots                                             \newline
        n &= \infty \text{(MC)} & q_t^{(\infty)} &= R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{T-1} R_T
    \end{aligned}$
- $n$-step $Q$-return
    - $q_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n Q(S_{t+n})$
- $n$-step Sarsa
    - $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( q_t^{(n)} - Q(S_t, A_t) \right)$
- [Lecture 4: Model-Free Prediction]({% post_url /RL/2021-04-13-model-free-prediction %})에서 배운 [TD(0)]({% post_url /RL/2021-04-13-model-free-prediction %}#mc-and-td), [$n$-step TD]({% post_url /RL/2021-04-13-model-free-prediction %}#n-step-td), [forward-view TD]({% post_url /RL/2021-04-13-model-free-prediction %}#forward-view-tdlambda), [backward-view TD]({% post_url /RL/2021-04-13-model-free-prediction %}#backward-view-tdlambda-1)를 적용할 수 있다.


#### Forward View Sarsa($\lambda$)

- TD($\lambda$): TD(0)에서 MC까지 return의 geometric mean을 예측치로 사용한다.
    - $q_t^\lambda = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} q_t^{(n)}$
        - 1-step의 가중치: $1 - \lambda$
        - $n$-step의 가중치: $(1 - \lambda) * \lambda^{n-1}$
    - $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( q_t^\lambda - Q(S_t, A_t) \right)$


#### Backward View Sarsa($\lambda$)

- TD($\lambda$)와 같이 eligibility trace를 사용한다.
    - Eligibility trace: 어떤 state를 방문할 때마다, 2가지 관점으로 책임 사유를 묻는다.
        1. 가장 최근에 방문했는가?
        2. 여러 번 방문했는가?
    - $\begin{aligned}
        E_0(s, a) &= 0      \newline
        E_t(s, a) &= \gamma \lambda E_{t-1}(s, a) + \mathbb{1}(S_t = s, A_t = a)
    \end{aligned}$
        - 각 state-action pair마다 값을 갖고 있으며, state-action를 방문하면 1씩 증가하고, step이 지날 때마다 일정한 비율로 감소한다.
        - $E_t(s, a)$가 클수록 책임이 크다고 판단하여 더 많이 업데이트되고, 작을수록 더 작게 업데이트 된다.
    - Forward-view TD($\lambda$)와 수학적으로 동일함이 증명되었다.
- $\begin{aligned}
    \delta_t &= R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)    \newline
    Q(s, a) &\leftarrow Q(s, a) + \alpha \delta_t E_t(s, a)
\end{aligned}$


#### Sarsa($\lambda$) Algorithm

![Sarsa($\lambda$) Algorithm](/assets/rl/sarsa_lambda_algorithm.png)

- [Sarsa Algorithm for On-Policy Control](#sarsa-algorithm-for-on-policy-control)과 비슷한데, 중간이 for-loop이 추가되었다.
- Sarsa Algorithm에서는 한 state-action pair만 업데이트 했지만, Sarsa($\lambda$) Algorithm에서는 모든 state-action pair를 업데이트한다.
    - 모든 state-action piar에 eligibility trace 값이 있기 때문에, 한 action을 하면 $Q$와 $E$를 업데이트해야 한다.
    - 계산량은 좀 더 들어가지만, 대신 정보 전파가 빠르다.


#### Sarsa($\lambda$) Gridworld Example

![Sarsa($\lambda$) Gridworld](/assets/rl/sarsa_lambda_gridworld.png)

- 목적지까지 진행하는 동안 reward -1을 받는다.
- 목적지에 도착하는 순간, 다른 reward를 받고 종료한다.
- 2번째 그림: 1-step Sarsa
    - 목적지까지 도착하는 순간의 것만 업데이트한다.
    - 이전 state에서 위로 가는 것이 좋다는 것만 배운다.
- 3번째 그림: Sarsa($\lambda$)
    - 지나온 모든 경로에 대해, eligibility trace 만큼 책임에 비례해서 업데이트 된다.
    - 가장 최근의 state는 많이 업데이트 되고, 과거로 갈수록 점점 작게 업데이트 된다.


---

## Off-Policy Learning


### Off-Policy Learning

- Off-policy learning: 서로 다른 Behavior policy $\mu$와 target policy $\pi$를 사용한다.
    - Target policy $\pi$에 따라 evaluation해서 state-value function 또는 action-value function을 구하는 것이 아닌, behavior policy $\mu$에 따라 agent가 움직이면서, target policy $\pi$를 학습하는 학습하는 방법이다.
        - Behavior policy $\mu$: 실제 action을 선택하는 policy이다.
            - ${S_1, A_1, R_2, \dots, S_T} \sim \mu$
            - Ex: Stochastic policy: 어떤 확률분포를 가지는 주사위가 있을 때, 주사위를 던져서 나온 수로 action을 선택하여 샘플링한다.
    - Exploration과 exploitation의 trade-off를 잘 조율할 수 있다.
        - 탐험적인 행동을 하면서도, 안전하게 최적의 policy를 배울 수 있다.
    - 이전에 수집한 데이터만 남아있고, 시뮬레이션을 하지 못하는 상황일 때 사용할 수 있다.
    - 예전에 했던 경험이나 다른 agent의 경험들을 재사용할 수 있기 때문에, 굉장히 효율적으로 경험을 사용할 수 있다.
        - Ex: 사람이 어떤 행동을 했을 때 그 행동을 하는 policy를 만드는 것이 아니라, 그 행동을 함으로써 나온 결과를 보고 따라한다거나 다른 행동을 해야겠다는 것 등을 배우는 것이다. 일단 사람은 어떤 경험을 제공해주고, 그 안에서 최적 policy를 배우는 것이다.
    - 한 policy를 따르면서 여러 개의 policy를 학습할 수 있다.
- On-policy learning: 경험을 쌓는 policy와 학습하는 policy가 같으므로, 경험을 통해 policy가 업데이트 되면, 새로 업데이트된 policy로 선택된 action으로 얻은 경험을 통해 다시 policy를 업데이트한다. 이 과정에서 이전 경험은 다시 사용하지 않고 버린다.


### Importance Sampling
{: style="color: red"}

#### Importance Sampling

- Estimate the expectation of a different distribution
    - $\begin{aligned}
        \mathbb{E} _{X \sim P} \left[ f(X) \right] &= \sum P(X)f(X)                       \newline
                                     &= \sum Q(X) \frac{P(X)}{Q(X)} f(X)    \newline
                                     &= \mathbb{E} _{X \sim Q} \left[ \frac{P(X)}{Q(X)} f(X) \right]
    \end{aligned}$
        - $X \sim P$: 확률분포 $P$에서 샘플링 된 어떤 $X$
            - Ex: 주사위 눈(1~6): 확률분포 $P = 1/6$인 주사위에서 샘플링되는 $X$
        - $X \sim Q$: 확률분포 $Q$에서 샘플링 된 어떤 $X$
            - Ex: 비뚤어진 주사위: 50% 확률로 6, 10% 확률로 1~5
        - $f(X)$: 어떤 함수
        - $\mathbb{E}_ {X \sim P} \left[ f(X) \right]$: 확률분포 $P$를 따르는 어떤 $X$에 대해, $f(X)$의 기댓값
- 목적: 확률분포 $P$를 따르는 주사위를 이용해서 $f(X)$의 기댓값을 구하고 싶어서 $P$를 따르는 주사위를 사용했는데, $Q$를 따르는 주사위를 이용했을 때의 기댓값을 구하고 싶다.
    -  $\mathbb{E} _{X \sim P} \left[ f(X) \right]$는 확률분포 $Q$에서 샘플링된 어떤 $X$에 대해 $\frac{P(X)}{Q(X)} f(X)$의 기댓값과 같다.
    - 즉 $Q$ 분포에서의 기댓값을 구하고 싶을 때, $P$분포와 $Q$분포의 비율을 곱해주면 된다.
        

#### Importance Sampling for Off-Policy Monte-Carlo

- Behavior policy $\mu$에 따라 게임이 끝나서 return $G_t$를 얻게 되면, 이 $G_t$를 그대로 사용하지 않고, $G_t$를 얻기까지 선택했던 모든 action들에 대해, 두 policy $\mu$와 $\pi$가 해당 action을 선택할 확률의 비율을 곱해주어야 한다. 이렇게 해야 target policy $\pi$를 따랐을 때의 return을 구할 수 있다.
    - $G_t^{\pi/\mu} = \frac{\pi(A_t \| S_t)}{\mu(A_t \| S_t)} \frac{\pi(A_{t+1} \| S_{t+1})}{\mu(A_{t+1} \| S_{t+1})} \dots \frac{\pi(A_T \| S_T)}{\mu(A_T \| S_T)} G_t$
    - $V(S_t) \leftarrow V(S_t) + \alpha \left( G_t^{\pi/\mu} - V(S_t) \right)$
- 주사위를 한 번만 던지면 $P/Q$를 한 번만 곱해진다. 하지만 실제 게임에서는 끝날때 까지 주사위를 계속 던지므로, 던진 수만큼 importance sampling correction이 들어간다.
- 이 방법은 상상속에서나 존재하는 방법론이며 사용할 수 없다.
    - 이 importance sampling의 variance는 매우 크기 때문에, $G_t^{\pi/\mu}$ 값이 폭발하거나 매우 작게 수렴될 것이다.
    - 또한 $\mu$가 0일 경우도 쓸 수 없다.


#### Importance Sampling for Off-Policy TD

- TD는 1-step마다 업데이트하기 때문에, importance sampling correction이 하나만 필요하다.
    - $V(S_t) \leftarrow V(S_t) + \alpha \left( \frac{\pi(A_t \| S_t)}{\mu(A_t \| S_t)} (R_{t+1} + \gamma V(S_{t+1})) - V(S_t) \right)$
    - 선택한 action으로부터 얻은 예측치에, $\pi/\mu$를 곱해준다.
- TD에서는 Monte-Carlo importance sampling보다 훨씬 작기 때문에 사용할 수 있다.


<br>


### Q-Learning


#### Q-Learning
{: style="color: red"}

- Atari 게임을 풀기위해 사용했던 방법론이다.
- *Importance sampling을 쓰지 않고, off-policy로 $Q$를 학습한다.*{: style="color: red"}
- *Behavior policy $\mu$를 따르는 action을 선택해서, $S_{t+1}$로 이동한다.*{: style="color: red"}
- *TD target인 $R_{t+1} + \gamma Q(S_{t_1}, A')$의 $A'$에는 target policy $\pi$를 따르는 action을 사용한다.*{: style="color: red"}
    - 즉, 다음 state $S_{t+1}$로 이동할 때는 behavior policy $\mu$를 따르고, 업데이트를 할 때는 $S_{t+1}$에서 target policy를 따르는 action $A'$를 선택하여, $Q(S_{t+1}, A')$를 사용한다.
    - TD target은 behavior policy $\mu$를 따르지 않으므로, [Sarsa](#sarsalambda)와 식이 동일하기 때문에 굉장히 말이 된다.


#### Off-Policy Control with Q-Learning

- Behavior policy: Target policy처럼 점점 improvement되면서, 탐험적인 행동을 하면 좋을 것 같다.
    - $\epsilon$-greedy w.r.t. $Q(s, a)$
- Target policy: 항상 좋은 선택만 하면 되기 때문에 greedy로 정한다.
    - Greedy $Q$ w.r.t $Q(s, a)$: $\pi(S_{t+1}) = \arg\max_{a'} Q(S_{t+1}, a')$
- 이렇게 정해서 사용하는 Q-learning이 많이 쓰인다.


#### Q-Learning Control Algorithm

![Q-Learning Control](/assets/rl/q_learning_control.png)

- [Sarsa](#sarsalambda)는 policy를 따르는 action에 대해 한 개의 $S_{t+1}$만 사용하기 때문에 밑에 하나의 점이 있었다.
- Q-learning은 할 수 있는 모든 action 중에 가장 좋은 action을 선택하기 때문에 호가 그려져 있다.
    - $Q(S, A) \leftarrow Q(S, A) + \alpha \left( R + \gamma \max_{a'} Q(S', a') - Q(S, A) \right)$
- Q-learning control을 하면, $Q$가 optimal action-value function으로 수렴한다. 
    - $Q(s, a) \rightarrow q_\*(s, a)$
- Sarsa max라고도 불린다.


#### Q-Learning Algorithm for Off-Policy Control

![Q-Learning Algorithm](/assets/rl/q_learning_algorithm.png)


---

## Summary

![Relationship Between DP and TD](/assets/rl/relationship_between_dp_and_td_1.png)

![Relationship Between DP and TD](/assets/rl/relationship_between_dp_and_td_2.png)