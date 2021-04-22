---
layout: post
title:  "Lecture 6: Value Function Approximation"
date:   2021-04-15 20:03:07 +0900
category: "Reinforcement Learning"
tags: YouTube
plugins: mathjax
---

David Silver 님의 [Introduction to reinforcement learning](https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) 강의 내용을 [팡요랩 Pang-Yo Lab](https://www.youtube.com/channel/UCwkGvF7xKz2E0Lv-fZ9wv2g)의 [강화학습의 기초 이론](https://youtube.com/playlist?list=PLpRS2w0xWHTcTZyyX8LMmtbcMXpd3s4TU)에서 풀어서 설명하기에, 이를 바탕으로 공부하여 정리한 내용입니다.

- Video: [[강화학습 6강] Value Function Approximation](https://youtu.be/71nH1BUjhNw)
- Slide: [Lecture 6: Value Function Approximation](https://www.davidsilver.uk/wp-content/uploads/2020/03/FA.pdf)


---

## Introduction

### Large-Scale Reinforcement Learning

- Lookup table 방법은: 각 state 정보를 담는 table을 만들어놓고 초기화해서 사용한다.
    - 하지만, 현실 문제는 정말 많은 state가 존재한다.
        - Ex: Backgammon: $10^{20}$개
        - Ex: 바둑: $10^{170}$개
        - Ex: 헬리콥더: Table로 만들수 없는 무한한 연속적인 상태 공간을 갖는다.
- 강화학습에서는 이런 현실문제를 어떻게 동작시킬 수 있을까?
    - Model-free method에서 어떻게 scale up 할 수 있을까?


### Value Function Approximation

- 지금까지는 lookup table 기반에서 진행되었다.
    - State-value function $V(s)$는 state $s$ 갯수만큼의 빈칸이 있었다.
    - State-action function $Q(s, a)$는 모든 state-action pair 갯수만큼의 빈칸이 있었다.
    - 단점:
        - 정말 많은 state와 action을 메모리에 담을 수 없다.
        - 담을 수 있다고 하더라도, 각각의 state의 value를 계산하는데 너무 느리다.
- 이제부터는 큰 MDP를 다루기 위해, function approximation을 사용한다.
    - Value function $v_\pi(s), q_\pi(s, a)$를 모방하는 $\hat{v}(s, \mathbf{w}), \hat{q}(s, a, \mathbf{w})$를 만든다.
        - $\begin{aligned}
            \hat{v}(s, \mathbf{w}) &\approx v_\pi(s)     \newline
            \hat{q}(s, a, \mathbf{w}) &\approx q_\pi(s, a)
        \end{aligned}$
            - $\mathbf{w}$: $\hat{v}(s, \mathbf{w})$ 함수 안에 들어있는 파라미터
        - Ex: Neural network 등
    - *내가 방문한 state부터 방문하지 않은 state까지 generalized하게, 실제 $V$와 $Q$를 하습하는 것이 목적이다.*{: style="color: red"}
        - Generalized: 방문하지 않은 state에 대해서도 올바른 value를 도출하도록 한다.
    - *$\hat{v}(s, \mathbf{w})$는 $\mathbf{w}$에 의해 다른 함수가 되므로, 학습한다는 것은 결국 $\mathbf{w}$를 업데이트한다는 의미이다.*{: style="color: red"}
        - $w$를 업데이트하기 위해, MC 또는 TD를 사용한다.


### Type of Value Function Approximation

![Type of Value Function Approximation](/assets/rl/type_of_value_function_approximation.png)

- 사각형을 internal parameter $\mathbf{w}$를 담고있는 blackbox라고 생각한다.
- Value function 모방 방법
    - 첫 번째 사각형: 이 blackbox로 $s$를 쿼리로 던지면, blackbox 내부에서 $\mathbf{w}$를 통해 계산된 $\hat{v}$가 출력된다.
- Q value function 모방 방법
    1. 두 번째 사각형: $s, a$를 쿼리로 던지면, $\hat{q}$가 출력된다.
    2. *세 번째 사각형: $s$만 쿼리로 던지면, $s$에서 할 수 있는 모든 action들에 대해 여러 개의 $\hat{q}$가 출력된다.*{: style="color: red"}


### Which Function Approximator?

- 모방하는 함수로 어떤 것을 사용할 수 있을까?
    - Linear combination of features: 각 feature들마다 가중치 $\mathbf{w}$를 부여해서, 이 가중치 합을 구하는 선형 방법
    - Neural network: 비선형 방법
    - Decision tree
    - Nearest neighbour
    - Fourier / wavelet bases
    - ...
- 꼭 미분 가능한 방법을 선택할 필요는 없지만, 여기서는 미분 가능한 function approximator을 통해 gradient를 계산하여 $\mathbf{w}$를 업데이트 할 것이다.
    - Linear combination of features
    - Neural network
- 그리고 non-stationary, non-iid data에도 적용할 수 있는 학습 방법을 알아볼 것이다.
    - Non-stationary, non-iid: 모분포가 계속 바뀌고, 이전 데이터와 이후 데이터가 서로 연관된 데이터를 의미한다.


---

## Incremental Method

### Gradient Descent

#### Gradient Descent

- Parameter vector $\mathbf{w}$의 미분가능한 함수 $J(\mathbf{w})$가 있다고 하자.
    - $\mathbf{w}$: 일반적인 $n$차원 벡터이고, 꼭 1차원일 필요는 없다.
- 함수 $J$가 최솟값을 갖은 $\mathbf{w}$을 찾기 위해, gradient를 사용한다.
    - $\nabla_\mathbf{w} J(\mathbf{w}) = \begin{pmatrix}
                                            \frac{\partial J(\mathbf{w})}{\partial \mathbf{w}_1}    \newline
                                            \vdots                                                  \newline
                                            \frac{\partial J(\mathbf{w})}{\partial \mathbf{w}_n}
                                        \end{pmatrix}$
- 함수 $J$에서 최솟값을 가지는 $\mathbf{w}$를 구하기 위해 gradient descent를 사용한다.
    - Gradient descent(경사 하강법)
        - $J$를 $\mathbf{w}$에 대해 gradient(편미분)을 구하면 $\mathbf{w}$의 갯수 $n$개 성분만큼 값이 계산되어지는데, 이 값들이 벡터를 이루기 때문에 크기와 방향을 나타낸다.
        - 이 벡터의 방향은 $J$의 $\mathbf{w}$에서 가장 빠르게 변화하는 방향이 되므로, -방향으로 *step-size $\alpha$만큼 아주 작게 움직이고*{: style="color: red"}, 다시 그 위치를 기준으로 gradient를 구해서 다시 -방향으로 움직이고, 이를 반복하면 $\mathbf{w}$ 부근의 최솟값에 도착하게 된다. 이 과정을 gradient descent라고 한다.
            - $\Delta\mathbf{w} = -\frac{1}{2} \alpha \nabla_\mathbf{w} J(\mathbf{w})$
                - $1/2$: 편미분할 때, 수학적 편의를 위해 넣어둔 것이다.
    - 최솟값은 global minimum이 아닌 *$\mathbf{w}$ 부근의 local minimum*{: style="color: red"}이다.
        - 만약 함수 $J$가 convex하면,  local minimum은 global minimum이다.


#### Value Function Approx. By Stochastic Gradient Descent

- 목적: $\hat{v}(s, \mathbf{w})$가 $v_\pi(s)$를 잘 모방하도록 $\mathbf{w}$를 잘 학습하는 것이 목적이다.
- 전지전능한 신이 $v_\pi$를 알려준다고 가정할 때, $v_\pi(s)$와 $\hat{v}(s, \mathbf{w})$의 차이를 계산하기 위해, 이 둘의 차이를 제곱한 기댓값을 구한다.
    - $J(\mathbf{w}) = \mathbb{E_\pi} [ (v_\pi(S) - \hat{v}(S, \mathbf{w}))^2]$
        - $J$: Loss
        - $V_\pi(S)$: 상수값
        - 제곱: +/-의 차이를 같은 차이라고 계산하기 위해 사용
        - 기댓값: policy $\pi$를 따를 때, $S$에 대한 차이를 알기 위해 사용
    - $\hat{v}(s, \mathbf{w})$가 $v_\pi(s)$를 잘 모방하고 있다면 $J$가 작을 것이므로, $J$가 작아지도록 $\mathbf{w}$를 업데이트해야 한다.
        - $\begin{aligned}
            \Delta \mathbf{w} &= \frac{1}{2} \alpha \nabla_w J(\mathbf{w})   \newline
                              &= \alpha \mathbb{E_\pi} \left[ (v_\pi(S) - \hat{v}(S, \mathbf{w}))\nabla_w \hat{v}(S, \mathbf{w}) \right]
        \end{aligned}$
        - 기댓값 갱신은 전체 gradient 갱신과 같다.
- Stochastic gradient descent: policy $\pi$를 따라 state를 방문하면서 샘플을 얻게 되면, 이 샘플들을 통해 업데이트한다.
    - $\Delta\mathbf{w} = \alpha (v_\pi(S) - \hat{v}(S, \mathbf{w}))\nabla_w \hat{v}(S, \mathbf{w})$
    - Policy $\pi$가 1번 state를 자주 방문한다면 1번 샘플이 많아질 것이고, 7번 state를 덜 방문하면 7번 샘플이 적어질 것이므로, 여러번 시행하면 자연스럽게 expectation과 같아진다.


<br>


### Linear Function Approximation


#### Feature Vectors

- State를 $n$개의 feature vector로 표현할 수 있다.
    - $\mathbf{x}(S) = \begin{pmatrix}
                            \mathbf{x}_1(S) \newline
                            \vdots          \newline
                            \mathbf{x}_n(S)
                        \end{pmatrix}$
        - Ex: 주식에서 어떤 종목의 현재 state를 넣으면, 이동평균선, 전고가, 볼린져밴드 등의 feature를 만든다.
    - 실제 문제에서의 state는 추상적인 개념인데, 이를 사람이 숫자 몇 개로 표현한 것이다. 따라서, state를 잘 표현할 수 있도록 feature를 잘 만들어야 한다.


#### Linear Value Function Approximation

- Value function을 학습하기 위해 모방함수를 사용하려고 하고, 이 모방함수로 linear combination of feature을 사용할 때의 예시이다.
- $\hat{v}(S, \mathbf{w}) = \mathbf{x}(S)^T \mathbf{w} = \sum_{j=1}^n \mathbf{x}_j(S)\mathbf{w}_j$
    - $\hat{v}$: Feature들의 각 성분에 $\mathbf{w}$를 곱해서 더한다(내적).
    - $\hat{v}$는 feature가 갖고 있는 어떤 성분의 제곱에 비례할 수도 있는데, 여기서는 선형합으로 구성되어 있기 때문에 충분히 flexible한 함수는 아니다.
    - flexibility가 부족해서 정확히 fitting되지 않을 순 있겠지만, 우리는 이 $\hat{v}$를 갖고 최선을 다하면 되기 때문에 상관없다.
- $J(\mathbf{w}) = \mathbb{E_\pi} \left[ (v_\pi(S) - \mathbf{x}(S)^T \mathbf{w})^2 \right]$
- Stochastic gradient descent는 global optimum으로 수렴한다.
    - 모방함수가 선형함수이기 때문에, 최저점이 하나만 존재한다.
- Update 규칙
    - $\begin{aligned}
        \nabla_\mathbf{w} \hat{v}(S, \mathbf{w}) &= \mathbf{x}(S)    \newline
        \nabla_\mathbf{w}                        &= \alpha (v_\pi(S) - \hat{v}(S, \mathbf{w})) \mathbf{x}(S)
    \end{aligned}$
    - *Update = step-size X prediction error X feature value*{: style="color: red"}
- 처음에는 이상한 값들이 나오겠지만, 업데이트가 진행될수록 실제 value에 근사하도록 $\mathbf{w}$가 수정된다.


#### Table Lookup Features

- *Table lookup과 linear value function approximation의 전혀 다른 2개의 방법이라고 생각하지 말자.*{: style="color: red"}
    - 즉, table lookup은 linear value function approximation의 한 예시일 뿐이다.
    - Ex: Feature를 state의 갯수만큼 만들고, 첫번째 feature은 $s_1$을 방문했을 때 해당 성분이 1인 feature vector를 만든다. 이 feature vector와 $\mathbf{w}$를 내적하면 $\hat{v}$를 만들 수있기 때문에, $\mathbf{w_1}, \dots, \mathbf{w_n}$은 lookup table의 각 칸의 값이 된다.
        - Table lookup features: $\mathbf{x}^{\text{table}}(S) = \begin{pmatrix}
                                                                    \mathbb{1}(S = s_1)     \newline
                                                                    \vdots                  \newline
                                                                    \mathbb{1}(S = s_n)
                                                                \end{pmatrix}$
        - $\hat{v}(S, \mathbf{w}) = \begin{pmatrix}
                                        \mathbb{1}(S = s_1)     \newline
                                        \vdots                  \newline
                                        \mathbb{1}(S = s_n)
                                    \end{pmatrix}
                                    \cdot
                                    \begin{pmatrix}
                                        \mathbf{w_1}            \newline
                                        \vdots                  \newline
                                        \mathbf{w_n}            \newline
                                    \end{pmatrix}$


---

## Incremental Methods

### Incremental Prediction Algorithms

#### Incremental Prediction Algorithms

- 지금까지는 전지전능한 신이 true value function을 알려준다고 가정했다.
- 하지만, RL에서는 true value function을 알려줄 신은 없고, 오직 reward signal만 있다.
- $v_\pi(s)$에 MC와 TD에서 얻은 target을 넣는다.
    - MC: True value functin 자리에 $G_t$를 넣는다.
        - $\Delta\mathbf{w} = \alpha (G_t - \hat{v}(S_t, \mathbf{w})) \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w})$
            - 모방함수가 state $s$에 대해 $G_t$를 출력해주기를 바라는 식이다.
            - 같은 $s$라도 $G_t$는 10 또는 20 등 될 수 있을 텐데, 만약 10과 20만 있다면 그 중간인 15를 향하는 방향으로 $\mathbf{w}$가 업데이트 될 것이다.
            - 즉, 어떤 샘플들을 이용해서 $\mathbf{w}$를 업데이트한다.
    - TD(0): TD target은 현재 state로부터 1-step 나아갈 때 얻는 보상과 다음 state의 추측치의 합이다. True value function 자리에 이 TD target을 넣는다.
        - $\Delta\mathbf{w} = \alpha (R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})) \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w})$
    - TD($\lambda$): True value function 자리에, 현재 step부터 마지막 step까지의 예측값에 $(1 - \lambda)\lambda^n$를 곱한 $\lambda$-return $G_t^\lambda$를 넣는다.
        - $\Delta\mathbf{w} = \alpha (G_t^\lambda + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})) \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w})$
        - [Forward-view TD]({% post_url /RL/2021-04-13-model-free-prediction %}#forward-view-tdlambda), 와 [backward-view TD]({% post_url /RL/2021-04-13-model-free-prediction %}#backward-view-tdlambda-1) 방식에 맞게 사용한다.
- MDP가 정의되어 있어야 하고, 문제의 목적에 따라 reward가 다르게 정의된다.


#### Monte-Carlo with Value Function Approximation

- Return $G_t$
    - Value function으로 return의 기댓값을 사용하기 때문에, true value $v_\pi(S_t)$의 unbiased estimation이다.
    - Policy, environment, 에피소드별 샘플링 등 stochastic 하기 때문에, return은 매번 달라지지만, 충분히 많은 샘플링을 하면 평균은 결국 true value가 된다.
    - 학습데이터 $\langle S_1, G_1 \rangle, \langle S_2, G_2 \rangle, \dots, \langle S_T, G_T \rangle$를 supervised learning 하는 것으로도 생각할 수 있다.
    - Linear Monte-Carlo policy evaluation
        - $\begin{aligned}
            \Delta\mathbf{w} &= \alpha (G_t - \hat{v}(S_t, \mathbf{w})) \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w})   \newline
                             &= \alpha (G_t - \hat{v}(S_t, \mathbf{w})) \mathbf{x}(S_t)
        \end{aligned}$
- Monte-Carlo는 optimum으로 수렴함이 증명되었고, 수렴조건이 더 까다로운 non-linear value function에서도 잘 수렴한다.


#### TD Learning with Value Function Approximation

- TD-target $R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})$
    - True value $v_\pi(S_t)$의 biased estimation이다.
    - 학습데이터 $\langle S_1, R_2 + \gamma \hat{v}(S_2, \mathbf{w}) \rangle, \langle S_2, R_3 + \gamma \hat{v}(S_3, \mathbf{w}) \rangle, \dots, \langle S_{T-1}, R_T + \gamma \hat{v}(S_T, \mathbf{w}) \rangle$를 supervised learning 하는 것으로도 생각할 수 있다.
    - Linear TD(0) policy evaluation
        - $\begin{aligned}
            \Delta\mathbf{w} &= \alpha (R + \gamma \hat{v}(S', \mathbf{w}) - \hat{v}(S_t, \mathbf{w})) \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w})   \newline
                             &= \alpha \delta \mathbf{x}(S_t)
        \end{aligned}$
- MC는 unbiased estimation이고, TD(0)는 우리가 1-step 뒤에 예측하는 방향으로 현재 예측치를 업데이트하는 것이기 때문에 일치한다는 보장은 없지만, linear TD(0)를 사용하면 global optimum에 가깝게 한다는 것이 증명되었다.
- *그럼 TD-target을 쓸 때 true function 부분에도 estimation이 들어가므로 $\mathbf{w}$에 대한 함수인데, TD-target을 편미분한 것과 위의 식은 다르다. 이대로 계산해도 되는 걸까?*{: style="color: red"}
    - $\mathbf{w}$를 업데이트할 때, 시간 축에서 한 방향을 보고 업데이트해야 한다. 즉, 1-step 후의 경험이 더 정확하다는 의미를 갖고 업데이트를 해나가는데, target에 대해서도 같이 업데이트하게 되면 1-step 이후 미래를 보는 것과 1-step 이전 과거를 보는 것이 섞이게 된다.
    - 이렇게 해도 업데이트를 하려면 할 수 있겠지만, 이것을 이해하려면 관련 배경들을 완전히 이해해야 한다. 지금은 위에서 사용하는 식이, 1-step 이후 미래를 향해서 업데이트 하는 것으로 이해하도록 하자.


#### TD($\lambda$) with Value Function Approximation

- $\lambda$-return $G_t^\lambda$
    - True value $v_\pi(S_t)$의 biased estimation이다.
    - 학습데이터 $\langle S_1, G_1^\lambda \rangle, \langle S_2, G_2^\lambda \rangle, \dots, \langle S_{T-1}, G_{T-1}^\lambda \rangle$를 supervised learning 하는 것으로도 생각할 수 있다.
    - Forward-view linear TD($\lambda$)
        - $\begin{aligned}
            \Delta\mathbf{w} &= \alpha (G_t^\lambda - \hat{v}(S_t, \mathbf{w})) \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w})   \newline
                             &= \alpha (G_t^\lambda - \hat{v}(S_t, \mathbf{w})) \mathbf{x}(S_t)
        \end{aligned}$
    - Backward-view linear TD($\lambda$)
        - $\begin{aligned}
            \delta_t &= R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})    \newline
            E_t &= \gamma\lambda E_{t-1} + \mathbf{x}(S_t)   \newline
            \nabla\mathbf{w} &= \alpha\delta_t E_t
        \end{aligned}$
- Forward-view TD($\lambda$)와 backward-view TD($\lambda$)는 동일하다.


<br>


### Incremental Control Algorithms


#### Control with Value Function Approximation

![Control with Value Function Approximation](/assets/rl/control_with_value_function_apprixomation.png)

- Policy evaluation: Approximation policy evaluation
    - $\hat{q}(\cdot, \cdot, \mathbf{w}) \approx q_\pi$
- Policy improvement $\epsilon$-greedy policy improvement


#### Action-Value Function Approximation

- True value function $q$를 알고 있을 때, $\hat{q}$를 모방하는 방법을 설명한다.
    - $\hat{q}(S, A, \mathbf{w}) \approx q_\pi(S, A)$
- Approximate action-value function $\hat{q}(S, A, \mathbf{w})$와 true action-value function $q_\pi(S, A)$의 mean-squared error를 최소화하는 방향으로 $\mathbf{w}$를 업데이트한다.
    - $J(\mathbf{w}) = \mathbb{E_\pi} \left[ (q_\pi(S, A) - \hat{q}(S, A, \mathbf{w}))^2 \right]$
    - Stochastic gradient descent
        - $\begin{aligned}
            -\frac{1}{2}\nabla_\mathbf{w} J(\mathbf{w}) &= (q_\pi(S, A) - \hat{q}(S, A, \mathbf{w})) \nabla_\mathbf{w} \hat{q}(S, A, \mathbf{w})    \newline
            \Delta\mathbf{w}                            &= \alpha (q_\pi(S, A) - \hat{q}(S, A, \mathbf{w})) \nabla_\mathbf{w} \hat{q}(S, A, \mathbf{w})
        \end{aligned}$


#### Linear Action-Value Function Approximation

- State와 action을 $n$개의 feature vector로 표현할 수 있다.
    - $\mathbf{x}(S, A) = \begin{pmatrix}
                            \mathbf{x}_1(S, A)  \newline
                            \vdots              \newline
                            \mathbf{x}_n(S, A)
                        \end{pmatrix}$
- Action-value function을 linear combination of features로 구성할 수 있다.
    - $\hat{q}(S, A, \mathbf{w}) = \mathbf{x}(S, A)^T\mathbf{w} = \sum_{j=1}^n \mathbf{x_j}(S, A)\mathbf{w_j}$
- Stochastic gradient descent update
    - $\begin{aligned}
        \nabla_\mathbf{w} \hat{q}(S, A,\mathbf{w}) &= \mathbf{x}(S, A)  \newline
        \Delta\mathbf{w}                            &= \alpha (q_\pi(S, A) - \hat{q}(S, A, \mathbf{w})) \mathbf{x}(S, A)
    \end{aligned}$


#### Incremental Control Algorithms

- Prediction과 마찬가지로, target에 $q_\pi(S, A)$를 넣어준다.
    - MC: $\Delta\mathbf{w} = \alpha(G_t - \hat{q}(S_t, A_t, \mathbf{w})) \nabla_\mathbf{w} \hat{q}(S_t, A_t, \mathbf{w})$
    - TD(0): $\Delta\mathbf{w} = \alpha(R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}) - \hat{q}(S_t, A_t, \mathbf{w})) \nabla_\mathbf{w} \hat{q}(S_t, A_t, \mathbf{w})$
    - TD($\lambda$)
        - Forward-view TD($\lambda$): $\Delta\mathbf{w} = \alpha(q_t^\lambda - \hat{q}(S_t, A_t, \mathbf{w})) \nabla_\mathbf{w} \hat{q}(S_t, A_t, \mathbf{w})$
        - Backward-view TD($\lambda$)
            - $\begin{aligned}
                \delta_t &= R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}) - \hat{q}(S_t, A_t, \mathbf{w})  \newline
                E_t &= \gamma\lambda E_{t-1} + \nabla_\mathbf{w} \hat{q}(S_t, A_t, \mathbf{w})  \newline
                \nabla\mathbf{w} &= \alpha\delta_t E_t
            \end{aligned}$


<br>


### Mountain Car


#### Linear Sarsa with Coarse Coding in Mountain Car

![Mountain Car](/assets/rl/mountain_car.png)

- State-value function을 나타낸 그래프이다.
- 지금까지 큰 문제에서 최적의 policy를 찾기 위한 model-free function approximation 방법을 배웠다.
    - Policy iteration
        1. Evaluation: Linear function approximation으로 $q$를 평가한다.
            - Ex: MC, TD(0)
        2. Improvement: $\epsilon$-greedy policy improvement
    - Linear function approximation을 사용하므로, linear Sarsa라고 말한다.
- Mountain car
    - State: Position, velocity
    - Action
        - 정해진 세기의 힘을 앞으로 준다.
        - 정해진 세기의 힘을 뒤로 준다.
        - 힘을 주지 않는다.
    - Reward: -1
    - 정상에 도달하면 종료한다.


#### Linear Sarsa with Radial Basis Functions in Mountain Car

![Linear Sarsa with Radial Basis Functions in Mountain Car](/assets/rl/mountain_car_linear_sarsa.png)

- 최종 학습되었을 때, state-value function 그래프이다.


#### Study of $\lambda$: Should We Bootstrap?

![Should We Bootstrap?](/assets/rl/bootstrap.png)

- x축: $\lambda$
    - $\lambda = 1$: Monte-Carlo 방법론
    - $\lambda = 0$: TD(0) 방법론
- y축: RMS error
- TD(0), TD($\lambda$)가 꼭 필요할까?
    - MC가 가장 성능이 안좋고, TD(0)는 MC보다 항상 좋다.
    - $\lambda$가 0과 1사이에 sweet spot이 존재한다.
    - Return의 variance가 매우 크기 때문에, 실제 문제에서 학습이 잘 안되므로 bootstrapping이 필요하다.


<br>


### Convergence


#### Baird's Counterexample

![Baird's Counterexample](/assets/rl/baird_counterexample.png)

- TD(0)는 항상 수렴할까?
    - 항상 수렴한다고 보장할 수 없다.


#### Parameter Divergence in Baird's Counterexample

![Parameter Divergence in Baird's Counterexample](/assets/rl/baird_counterexample_parameter_divergence.png)

- 위의 환경에서 value function은 발산하므로, 항상 TD(0)가 항상 수렴하지 않는다는 반례가 된다.
- 즉, bootstrapping이 좋지만, 항상 수렴하지는 않는다는 뜻이다.
    - 수렴성이 보장되어 있지는 않아도, 실제로는 잘 수렴한다.


#### Convergence of Prediction Algorithms

![Convergence of Prediction Algorithms](/assets/rl/convergence_of_prediction_algorithms.png)

- Linear를 적용할 수 있는 알고리즘에 non-linear를 적용할 수 있다.
- Linear에서 non-linear로 갈수록, MC에서 TD($\lambda$)로 갈수록, on-policy에서 off-policy로 갈수록 수렴성이 안좋다.
- 하지만 이 테이블은 어디까지나 이론적인 것일 뿐, 실제로는 신기하게 잘 수렴한다.


#### Gradient Temporal-Difference Learning

![Gradient Temporal-Difference Learning](/assets/rl/gradient_temporal_difference_learning.png)

- TD does not follow the gradient of any objective function.
- Gradient TD: Bellman error의 true gradient를 계산하기 때문에 non-linear에 대해서도 수렴성이 좋고, off-policy에 대해서도 수렴성이 좋다.


#### Convergence of Control Algorithm

![Convergence of Control Algorithm](/assets/rl/gradient_q_learning.png)

- Gradient Q-learning: Control일 때도 수렴성이 좋다.


---

## Batch Methods


### Batch Reinforcement Learning

- 지금까지 본 incremental method는 gradient descent를 이용해서 한 개의 샘플로 evaluation과 improvement를 했다.
- 하지만 이것은 sample efficient하지 않다.
    - State $s$에서 action을 선택해서 reward를 받고, 다음 state $s'$에 도착할 때, 샘플 하나를 얻는다.
    - 이 transition 한 번으로 1번 policy iteration한 뒤에, 그 경험은 버려지기 때문에 샘플을 효과적으로 쓰지 않고 있다.
- Batch method: Training data처럼 agent가 쌓은 경험들이 있고, 이 경험들을 반복적으로 사용해서 학습하는 방법을 말한다.


### Least Squares Prediction


#### Least Squares Prediction

- 주어진 경험들 $\mathcal{D}$를 통해 $\hat{v}(s, \mathbf{w})$가 $v_\pi(s)$를 모방하는 것이 목적이다.
    - Expereice $\mathcal{D}$: state-value pair의 묶음
        - $\mathcal{D} = \\{ \langle s_1, v_1^\pi \rangle, \langle s_2, v_2^\pi \rangle, \dots, \langle s_T, v_T^\pi \rangle \\}$
- Least squares algorithm: $\hat{v}(s_t, \mathbf{w})$와 $v_t^\pi$ 사이의 sum-squared error을 최소화하는 $\mathbf{w}$를 찾는 알고리즘이다.
    - $\begin{aligned}
        \text{LS}(\mathbf{w}) &= \sum_{t=1}^T(v_t^\pi - \hat{v}(s_t, \mathbf{w}))^2     \newline
                              &= \mathbb{E_\mathcal{D}} \left[ (v^\pi - \hat{v}(s, \mathbf{w}))^2 \right]
    \end{aligned}$
        - Incremental method에서는 expectation에 $\pi$가 있었지만, 여기서는 주어진 경험들 통해 계산하기 때문에 expectation에 $\mathcal{D}$가 들어간다.


#### Stochastic Gradient Descent with Experience Replay

- 아래를 반복한다.
    1. Experience $\mathcal{D}$에서 state-value pair를 샘플링한다.
        - $\mathcal{D} = \\{ \langle s_1, v_1^\pi \rangle, \langle s_2, v_2^\pi \rangle, \dots, \langle s_T, v_T^\pi \rangle \\}$
        - $\langle s, v^\pi \rangle \sim \mathcal{D}$
    2. Stochastic gradient descent update를 적용한다.
        - $\Delta \mathbf{w} = \alpha (v^\pi - \hat{v}(s, \mathbf{w})) \nabla_\mathbf{w} \hat{v}(s, \mathbf{w})$
- 이렇게 업데이트를 진행하면 least squares solution에 수렴한다.
    - $\mathbf{w}^\pi = \arg\min_\mathbf{w} \text{LS}(\mathbf{w})$
- Experience replay
    - 데이터가 1000개 있으면, 그 중 10개를 샘플링해서 업데이트 하면서, 이 작업을 계속 반복적으로 한다.
    - 10만번 반복하게 되면 한 샘플을 여러번 사용하게 되므로, 좀 더 경험들을 효율적으로 쓸 수 있다.
    - Off-policy RL에서 굉장히 많이 사용하는 방법이다.
    - On-policy RL에서는 사용할 수 없다.


#### Experience Replay in Deep Q-Networks (DQN)

- Off-policy에서 non-linear function을 쓰면 수렴성이 보장되지 않았다. 즉, Naive하게 실행시키면 수렴하지 않고 발산하게 된다.
- 이 방법을 사용하면서도 잘 수렴하도록 하기위해, 아래 2가지 트릭을 사용한다.
    1. Experience replay
        - $\epsilon$-greedy policy로 게임을 진행하는데, transition $(s_t, a_t, r_{t+1}, s_{t+1})$를 replay memory $\mathcal{D}$에 저장한다.
        - 그리고 replay memory $\mathcal{D}$에서 랜덤하게 mini-batch를 샘플링해서 이를 통해 학습한다.
    2. Fixed Q-targets
        - 업데이트하지 않는 고정된 파라미터 $\mathbf{w^-}$와 업데이트하는 파라미터 $\mathbf{w}$를 준비한다.
        - TD target을 계산할 때는 고정된 파라미터 $\mathbf{w^-}$를 사용하고, $\mathbf{w}$를 업데이트한다.
        - 파라미터를 하나만 두고 학습하게 되면, 업데이트를 할 때마다 파라미터가 향하는 방향이 계속 바뀌기 때문에 non-linear function에서 수렴하기가 더 어려워진다. 그래서 2개의 파라미터를 만들고, 1000번 정도 한 파라미터를 고정시킨 채 TD target을 계산하면서 다른 파라미터를 업데이트 시키고, 다시 반대로 고정시켜서 1000번 정도 파라미터를 고정시킨 채 TD target을 계산하면서 다른 파라미터를 업데이트 시키는 방식으로 두 개의 파라미터를 관리한다.
- Q-network와 Q-learning target 사이의 MSE를 최적화시킨다.
    - $\mathcal{L}(\mathbf{w_i}) = \mathbb{E_{s, a, r, s' \sim \mathcal{D_i}}} \left [ \left ( r + \gamma \max_{a'}Q(s', a'; w_i^-) - Q(s, a; w_i) \right)^2 \right]$
        - $w_i^-$: 업데이트하지 않는 고정된 파라미터
        - $w_i$: 업데이트하는 파라미터
        - $w_i$를 업데이트하는데, TD target은 $w_i^-$를 사용한다.


#### DQN in Atari

![DQN](/assets/rl/dqn.png)

- 최신 4장의 게임 화면이 들어오면, convolutional layer을 거쳐서 최종 layer에서 action의 갯수만큼 Q-value를 출력한다.
- Reward는 게임 score의 변화량을 사용했다.
- 모든 게임에 대해, hyperparameter와 네트워크 구조를 고정시키고 학습만 진행했다.


#### DQN Results in Atari

![DQN Results in Atari](/assets/rl/dqn_results.png)

- Atari에는 여러 종류의 게임이 있다.
- Linear combination of features을 사용하는 예전 방법론과 비교했을 때, DQN이 대부분의 게임을 이겼다.
- 그림에서 가운데 선이 있는데, 선을 기준으로 왼쪽은 사람보다 더 잘하고, 오른쪽은 사람보다 못한다.


#### How much does DQN help?

![How much does DQN help?](/assets/rl/dqn_compare.png)

- Replay memory와 fixed-Q를 사용했을 때와 사용하지 않았을 때를 비교했고, 둘 모두를 사용했을 때 가장 좋았다.
