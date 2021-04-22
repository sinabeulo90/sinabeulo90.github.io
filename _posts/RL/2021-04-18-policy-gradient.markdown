---
layout: post
title:  "Lecture 7: Policy Gradient"
date:   2021-04-18 14:35:09 +0900
category: "Reinforcement Learning"
tags: YouTube
plugins: mathjax
---

David Silver 님의 [Introduction to reinforcement learning](https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) 강의 내용을 [팡요랩 Pang-Yo Lab](https://www.youtube.com/channel/UCwkGvF7xKz2E0Lv-fZ9wv2g)의 [강화학습의 기초 이론](https://youtube.com/playlist?list=PLpRS2w0xWHTcTZyyX8LMmtbcMXpd3s4TU)에서 풀어서 설명하기에, 이를 바탕으로 공부하여 정리한 내용입니다.

- Video: [[강화학습 7강] Policy Gradient](https://youtu.be/2YFBordM1fA)
- Slide: [Lecture 7: Policy Gradient](https://www.davidsilver.uk/wp-content/uploads/2020/03/pg.pdf)


---

## Introduction

### Policy-Based Reinforcement Learning

- 지금까지는 gradient descent 방법으로 파라미터 $\theta$를 통해 approximated value function 또는 approximated action-value function을 구했다.
    - $V_\theta(s) \approx V^\pi(s)$
    - $Q_\theta(s, a) \approx Q^\pi(s, a)$
    - 전지전능한 신이 $v_\pi$를 알려줄 수 없으므로, Monte-Carlo 또는 TD를 통해 $V$와 $Q$를 업데이트 했다.
    - 즉, *true value function을 모방하기 위해 approximated value function을 업데이트한 뒤, 이를 이용해서 policy를 만들었다.*{: style="color: red"}
- 지금부터는 policy를 직접적으로 parameterize한다.
    - $\pi_\theta(s, a) = \mathbb{P} [a \| s, \theta]$
    - 파라미터를 이용해서 policy를 어떤 함수로 표현한다.
        - Ex: Neural net
- model-free RL로 진행한다.
    - Model-free: MDP에 대해 완전한 정보가 없는 상황일 때, agent가 environment에서 policy를 따라 어떤 행동을 선택하고, 어떤 다른 state로 이동하면서 얻는 경험들을 통해서 학습하는 방법이다.


### Value-Based and Policy-Based RL

- Value based
    - Value function에 기반한 방법론이다.
    - Value function을 어떤 파라미터에 대한 함수로 표현하고, 이 value function이 정확한 value을 출력하도록 파라미터드를 업데이트한다.
    - Implicit policy: Value function을 통해서 만들어진 policy이다.
        - Ex: $\epsilon$-greedy
- Policy based
    - Value function을 학습하지 않고, 직접 policy를 학습한다.
    - Policy gradient
- Actor-Critic
    - Value based와 policy based 모두 이용해서 학습한다.
    - Actor: Policy를 통해 움직인다.
    - Critic: Value function을 평가한다.


### Advantages of Policy-Based RL

- 장점
    - 수렴하는 성질이 더 좋다.
        - Value-based RL에서는 function approximator를 사용하기 때문에, variance가 크면 수렴이 잘 안된다.
    - 매우 많은 action space 또는 continuous action space에서 효과적이다.
        - $Q(s, a)$를 학습한다고 가정할 때, greedy하게 움직이기 위해 해당 state에서 가장 큰 $Q$를 가지는 action을 선택해야 하는데, 이 자체가 하나의 최적화 문제이다.
        - 또한, 어떤 임의의 함수 $Q$가 있고 입력으로 0 ~ 1 사이의 모든 실수에 대해 가능할 때, 가장 큰 $Q$를 가지는 실수 입력을 찾는 것 또한 하나의 최적화 문제이다.
        - Discrete action의 경우, 모든 가능한 행동들을 시도해서 가장 큰 $Q$를 찾으면 된다.
            - Descrete action space: 1, 2, ... 등을 행동으로 정의한다.
        - Continuous action의 경우, 0 ~ 1 사이의 모든 실수를 넣어볼 수 없기 때문에, 최적화 문제를 풀어야 한다.
            - Continuous action space: 0 ~ 1 사이의 실수를 행동으로 정의한다.
                - 무한 개의 action이 가능하며, 굉장히 자주 있는 문제이다.
                - Ex: 로봇 팔을 제어하는 문제일 경우, 1°, 17°, 17.3° 회전이 가능하다.
        - *Policy를 직접 학습하면, state가 주어졌을 때 직접 action이 출력된다.*{: style="color: red"}
        - 지금까지의 value-based RL에서는 모두 deterministic policy만 학습했고, stochastic policy는 학습하지 않았다.
            - Deterministic policy: 어떤 state에서 어떤 action을 하는 것이 결정론적이다.
                - Ex: greedy한 업데이트
            - Stochastic policy: 어떤 state에서 어떤 action을 하는 것이 확률론적이다.
                - 같은 state에서 선택되는 행동이 매번 달라진다.
                - Ex: 1번 action: 50%, 2번 action: 50%
- 단점
    - Global optimum보다 local optimum에 수렴하기 쉽다.
    - Variance가 크고, 비효율적인 면이 있다.
        - Value-based RL은 매우 aggressive한 방법론이다. $Q$를 한번 업데이트한 뒤, agent가 선택할 수 있는 action 중 가장 큰 $Q$를 가지는 action을 선택하므로 policy가 급격하게 바뀐다.
        - Policy Gradient: Gradient만큼 조금씩 업데이트 되므로, 1번 action이 좋은 선택인 경우 한 번 업데이트 될 때, 1번 action을 선택할 확률이 60%에서 62%로 향상된다. 즉, value-based보다 좀 더 smooth하고 stable하지만, 조금 효율성이 떨어진다.


### Rock-Paper-Scissors Example


#### Example: Rock-Paper-Scissors

- Stochastic policy가 필요한 상황에 대한 예시이다.
- Deterministic policy의 경우, agent A가 항상 주먹을 낼 경우, 몇 번 게임을 진행하면 상대방 agent B는 agent A를 이기는 policy를 금방 찾아낼 것이다.
- Uniform random policy의 경우, 가위/바위/보를 낼 확률이 1/3이므로 상대방 agent B는 agent A를 완전히 이길 수 없을 것이다.
    - 따라서 두 agent는 1/3확률로 가위/바위/보를 내는 것이 최적이다. 이를 게임이론에서 *[내시 균형(Nash equilibrium)](https://ko.wikipedia.org/wiki/내시_균형)*{: style="color: red"}이라고 한다.
        - Nash equilibrium: A, B가 어떤 같은 전략을 썼을 때, 두 agent 모두 전략을 바꿀 요인이 없어서 그대로 균형을 유지하는 상태를 말한다.
    - *내시 균형에 도달하여 최적이 될 수 있도록 학습하기 위해서는 policy-based RL이 필요하다.*{: style="color: red"}


<br>


### Aliased Gridworld Example


#### Example: Aliased Gridworld (1)

![Aliased Gridworld](/assets/rl/aliased_gridworld_1.png)

- Stochastic policy가 필요한 상황에 대한 예시이다.
- 해골에 빠지면 죽고, 금에 도달하면 성공하는 게임이다.
- MDP의 모든 정보를 알고 있는 상황을 partially observable한 상황으로 바꿔보자.
    - Ex: 동/서/남/북 방향에 벽이 있는지 여부를 확인하는 방식으로 state를 구분하는 feature를 만들면, fully known MDP가 깨지게 된다. 이렇게 되면, 회색 칸은 이 feature만으로 구분되지 않는 칸이 된다.
- 이 가정은 굉장히 합리적인데, feature는 얼마든지 완전하지 않을 수 있고 또한 feature가 완전한지 완전하지 않은지도 모르는 복잡한 상황이 있을 수 있기 때문이다.


#### Example: Aliased Gridworld (2)

![Aliased Gridworld](/assets/rl/aliased_gridworld_2.png)

- Deterministic policy인 경우
    - 2개의 회색 칸은 완전히 같은 state이므로, 같은 value를 출력한다.
    - 만약 policy가 한 회색 칸에서 왼쪽으로 가야한다고 정의되었다면, 나머지 회색 칸도 같은 policy를 따라 왼쪽으로 가는 행동을 선택할 것이다. 이렇게 되면, 왼쪽 상단의 회색 칸에 있는 agent는 계속 좌우를 이동하면서 평생 금으로 도달하지 못하게 된다.


#### Example: Aliased Gridworld (3)

![Aliased Gridworld](/assets/rl/aliased_gridworld_3.png)

- Stochastic policy인 경우
    - 회색 칸이 왼쪽 방향을 선택할 확률이 50%, 오른쪽 방향을 선택할 확률이 50%라면, 최적으로 항상 금에 도달할 수 있다.
    - 즉 state를 완전히 구분할 수 없는 partially observable한 상황에서 사용할 수 있다.
- [Policy Improvement]({% post_url /RL/2021-04-12-planning-by-dynamic-programming %}#policy-improvement-1)에서 항상 optimal deterministic policy가 존재한다는 것을 확인하고 학습을 진행해왔는데, 위의 예시는 반례가 되는 것이 아닐까?
    - Markov property(fully observable MDP)가 성립하면 optimal deterministic policy가 존재한다.
    - 위의 예시는 feature가 다른 state를 같은 state로 바라보는 partially observable MDP이기 때문에 반례가 되지는 않는다.


<br>


### Policy Search


#### Policy Objective Functions
{: style="color: red"}

- $\pi_\theta(s, a)$: 파라미터 $\theta$에 대해 어떤 state $s$에서 action $a$를 선택할 확률을 출력하는 function approximator이다.
- 그럼 어떤 policy를 좋은 policy라고 할 수 있을까?
    - Policy objective function: Maximize하고자 하는 목적함수이다.
        - 전지전능한 신이 value function을 알려줬을 때, 가장 큰 value를 가지는 행동을 선택하는 것이 좋은 policy인 것처럼, 어떤 policy를 따랐을 때 다른 policy보다 return이 더 크면 더 좋은 policy가 된다.
- Policy를 비교하기 위한 3가지 방법
    1. Episodic environments인 경우
        - $J_1(\theta) = V^{\pi_\theta}(s_1) = \mathbb{E_{\pi_\theta}} [v_1]$
        - 게임이 끝나는 environment인 경우, start value를 사용할 수 있다.
            - Start value: 처음 시작하는 state $s_1$의 value를 의미한다.
        - Ex: 철권에서 항상 두 캐릭터는 어느정도 거리를 두고 서로 마주보는 state $s_1$에서 시작한다. 이 때, 어떤 policy $\pi$로 게임을 끝까지 했을 때, 얼마의 return을 얻었는지에 따라, return의 기댓값으로 policy $\pi$의 value가 정해진다.
        - State $s_1$는 하나가 될 수도 있고, 고정된 분포일 수도 있다.
            - Ex: 철권에서 두 캐릭터가 50% 확률로 1m 떨어져서 시작하고, 50% 확률로 2m 떨어져서 시작해도 된다.
            - 즉, 고정된 start state 분포가 있을 때 정해지는 start state의 value를 maximize하는 것이 목표이다.
    2. Continuing environments인 경우
        - $J_{\text{av}V}(\theta) = \sum_s d^{\pi_\theta}(s)V^{\pi_\theta}(s)$
            - Stationary distribution $d^{\pi_\theta}(s)$
                - [Markov chain]({% post_url /RL/2021-04-09-markov-decision-process %}#markov-chains) 환경에서 agent가 policy $\pi$를 따라 계속 움직이면, agent가 각 state에 위치하는 확률을 구할 수 있다. 각 state별로 어느 정도 수렴한 뒤에 얻은 stationary distribution이 $d^{\pi_\theta}(s)$이다.
            - Average value: (Agent가 어떤 state에 있을 확률 X 해당 state의 value)의 총합을 말하며, 이를 목적함수로 사용한다.
    3. Average reward per time-step
        - $J_{\text{av}R}(\theta) = \sum_s d^{\pi_\theta}(s) \sum_{a} \pi_\theta(s, a) \mathcal{R_s^a}$
            - 어떤 state에서 policy $\pi_\theta$를 1번 따랐을 때, reward의 기댓값을 계산한다. 그리고 agent가 해당 state에 있을 확률 $d^{\pi_\theta}(s)$와 return을 곱한다. 이를 모든 state에 대해 계산한 뒤 모두 더한다.
            - 모든 state에 대해 policy $\pi_\theta$는 1-step action을 선택하고, 이때의 모든 state의 value 합을 목적함수로 사용한다.
- 위 3가지 서로 다른 목적함수에 대해 같은 방법론이 작동한다. 즉, 하나의 방법론이 어떤 한 개의 목적함수에만 맞는 방법론이 아니라, 3가지 목적합수에 적용될 수 있다.


#### Policy Optimisation

- Policy-based RL은 optimization 문제이다.
    - Optimization problem: 어떤 임의의 함수가 주어졌을 때, 값을 최대화하는 입력을 찾는 문제이다.
    - Policy는 $\theta$로 parameterized 되어있기 때문에, $\theta$가 바뀌면 policy도 함께 바뀐다. 그러면 게임을 하면서 얻는 reward도 바뀌면서 목적함수 $J(\theta)$가 바뀌게 되다. 여기서 $\theta$를 조정해서 $J(\theta)$가 maximize하는 $\theta$를 찾는 것이 목표이다.
- 최적화 문제를 풀기위한 다양한 방법론들이 존재한다.
    - Gradient를 사용하지 않는 방법
        - Hill climbing
        - Simplex / amoeba / Nelder Mead
        - Genetic algorithms
    - Gradient를 사용하는 방법
        - Gradient descent
        - Conjugate gradient
        - Quasi-newton
            - Newton method(2차 편미분)을 통해 좀 더 빠른 속도로 문제를 푸는 방법
            - 일반적인 방법으로 2차 편미분을 구하려면 계산이 너무 어려워져서, Quasi-newton이 나왔다.
- 여기서는 gradient descent를 사용할 것이다.


---

## Finite Difference Policy Gradient

### Policy Gradient

- 어떤 목적함수 $J(\theta)$가 $\theta$에 대한 gradient를 구할 수 있을 때, 이 gradient를 통해 step-size $\alpha$만큼 $\theta$를 업데이트 하는 방식이다.
    - $\Delta\theta = \alpha\nabla_\theta J(\theta)$
    - Policy $\pi$는 우리가 정하는 함수이기 때문에, gradient를 구할 수 있는 함수로 정의하면 된다.
    - Policy $\pi$로 neural net을 쓰는 이유도 gradient를 잘 계산할 수 있는 함수이기 때문이다.
- Value function approximation에서는 function approximator가 true value function을 잘 모방하기 위해 error를 minimize하는 방법으로 gradient descent를 적용했지만, 여기서는 목적함수 $J(\theta)$를 maximize하기 때문에 gradient ascent를 적용한다.
- $\nabla_\theta J(\theta)$: Policy gradient
    - $\nabla_\theta J(\theta) = \begin{pmatrix}
                                    \frac{\partial J(\theta)}{\partial\theta_1}     \newline
                                    \vdots                                          \newline
                                    \frac{\partial J(\theta)}{\partial\theta_n}     \newline
                                \end{pmatrix}$
    - Policy gradient는 목적함수 $J(\theta)$에 대한 각 $\theta$ 성분의 편미분이기 때문에, 성분이 $n$개 있으면 $\theta_1$로 편미분한 값, $\theta_2$로 편미분한 값, ..., $\theta_n$으로 편미분한 값이 모여서 성분 $n$개의 벡터가 만들어진다. 벡터는 크기와 방향이 있기 때문에, 이 방향으로 step-size $\alpha$를 곱한 값 만큼 $\theta$를 업데이트한다.
    - *Policy gradient는 목적함수 $J(\theta)$에 대한 gradient로 업데이트하는 것이지, $\pi_\theta$에 대한 gradient로 업데이트하는 것이 아니다. 즉, $J(\theta)$를 최대화하는 것이 목적이다.*{: style="color: red"}


### Computing Gradients By Finite Differences

- 목적함수 $J(\theta)$의 gradient를 구하는 것은 쉽지 않다. $\pi_\theta$에 대한 gradient를 알아도 구하기가 쉽지 않다.
- 제일 무식하면서 쉬운 방법: $\theta_1 \sim \theta_n$까지 차례를 돌아가면서, 하나씩 값을 조금씩 바꿔보면서 각 항에 대한 기울기를 계산한다.
    - Ex: $\theta_1$이 1이라면, 1.01을 넣을 때의 $J(\theta)$ 값을 확인한다. 이 때, $\theta_1$을 0.01만큼 바꿨을 때 $J(\theta)$의 변화량을 알 수 있으므로, $\theta_1$의 편미분을 구할 수 구할 수 있다. 마찬가지로 $\theta_2$를 조금씩 바꿔가면서 $J(\theta)$의 변화량을 통해 $\theta_2$의 기울기를 계산하면서 총 $n$번 반복한다.
    - $n$-dimension gradient를 한 번 구하기 위해, $n$번 evaluation 해야한다.
        - 만약 $\theta$의 성분이 백만 개가 있다면(Neural net의 경우 1억), 한 번 업데이트하기 위해 백만 개를 evaluation 해야한다.
    - 굉장히 단순하지만 noisy하고 비효율적이다.
    - 미분 가능하지 않는 어떠한 policy에 대해서도 gradient를 계산할 수 있다.


### AIBO example


#### Training AIBO to Walk by Finite Difference Policy Gradient

![AIBO](/assets/rl/aibo.png)

- 12개의 파라미터로 구성된 policy에 대해 finite difference policy gradient를 적용해서 학습해도 잘 되었다.


---

## Monte-Carlo Policy Gradient


### Likelihood Ratios


#### Score Function

- Policy gradient를 수학적으로 접근해보자.
- 미분가능한 policy $\pi$가 있다고 하자.
- *Likelihood ratio*{: style="color: red"}: 나중에 무언가를 편하게 사용하기 위해 이렇게 바꾸는 일종의 트릭이다.
    - $\begin{aligned}
        \nabla_\theta \pi_\theta(s, a) &= \pi_\theta(s, a) \frac{\nabla_\theta \pi_\theta(s, a)}{\pi_\theta(s, a)}  \newline
                                       &= \pi_\theta(s, a) \nabla_\theta \log\pi_\theta(s, a)
    \end{aligned}$
        - $\pi_\theta$의 gradient에 분자/분모에 $\pi$를 곱해서 정리하면, $\log\pi_\theta$의 gradient가 된다.
- Score function: $\nabla_\theta \log\pi_\theta(s, a)$


<details markdown="1">
<summary markdown="1">
#### Softmax Policy
</summary>

![Softmax Policy](/assets/rl/softmax_policy.png)

- Neural net에서 굉장히 많이 쓰이는 형태로, 각 action들을 확률 형태로 표현시켜주는 operator이다.
- Softmax policy인 경우, score function을 쉽게 구할 수 있다.
</details>


<details markdown="1">
<summary markdown="1">
#### Gaussian Policy
</summary>

![Gaussian Policy](/assets/rl/gaussian_policy.png)

- Gaussian 분포를 이용한 policy로, 어떤 state에서 대체로 어떤 action을 선택하는데, 약간의 variance를 줘서 가장 많이 선택되는 action으로부터 조금 떨어진 다른 action을 선택하도록 하는 policy이다.
</details>


<br>


### Policy Gradient Theorem


#### One-Step MDPs

- Stationary distribution $d(s)$로 정해지는 state $s$에서부터, 1-step 진행한 뒤에 reward $r = \mathcal{R_{s, a}}$를 얻고 끝나는 MDP가 있다고 하자.
    - Ex: Bandit machine: 어떤 slot machine을 돌릴 것인지에 대한 확률분포가 있다고 가정하고, 선택된 slot machine을 당겨서 결과를 확인한다.
    - $\begin{aligned}
        J(\theta) &= \mathbb{E_{\pi_\theta}} [r]   \newline
                &= \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_\theta(s, a)R_{s, a}
        \end{aligned}$
        - $J(\theta)$: [목적함수 3개](#policy-objective-functions) 중에서 3번째 정의를 가져온다.
        - $d(s)$: 해당 state에 위치할 확률
        - $\pi_\theta(s, a)$: 해당 state에서 어떤 action을 선택할 확률
    - $\begin{aligned}
        \nabla_\theta J(\theta) &= \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_\theta(s, a) \nabla_\theta \log\pi_\theta(s, a) R_{s, a}    \newline
                                &= \mathbb{E_{\pi_\theta}}[\nabla_\theta \log\pi_\theta(s, a) r]
    \end{aligned}$
        - $r$: 어떤 state $s$에서 선택한 action $a$의 reward이다. 같은 state에서 같은 action을 해도 매번 다를 수 있다. 실제 샘플에서 얻은 값이다.
        - $R_{s, a}$: 어떤 state $s$에서 action $a$를 선택했을 때, 실제 reward를 받을지 알 수 없는 확률변수이다.
        - Likelihood ratio trick에 의해 $\nabla_\theta \pi_\theta(s, a)$ 대신, $\pi_\theta \nabla_\theta \log\pi_\theta(s, a)$를 넣어준다.
        - *이렇게 하면, $\nabla_\theta J(\theta)$를 기댓값으로 표현할 수 있다. 즉, policy $\pi_\theta$를 따라 행동을 선택할 때, 해당 $\langle s, a \rangle$에 대한 $\log\pi_\theta(s, a)$의 gradient와 $r$를 곱하면 $\nabla_\theta J(\theta)$가 된다는 의미이다.*{: style="color: red"}
        - *$\pi_\theta$를 따르는 action을 선택하면, 각 transition의 샘플은 $\nabla_\theta J(\theta)$의 샘플들이 되며, 이를 통해 unbiased한 $\nabla_\theta J(\theta)$을 구할 수 있다.*{: style="color: red"} 각 샘플들은 확률변수이기 때문에, $\nabla_\theta J(\theta)$와 같지 않을 수 있지만, 반복적으로 시행하면 대수의 법칙에 의해 기댓값은 결국 $\nabla_\theta J(\theta)$와 같아진다.
        - $\sum\sum$의 항에 $\pi_\theta$가 없었다면 expectation 형태를 만들 수 없었을 것이다. 실제 문제에서는 agent가 policy를 따라 얻은 경험만을 활용해야하기 때문에, 해당 policy에 대한 expection을 만들어야 우리가 원하는 값을 구하기가 편해진다.
- Likelihood ratio trick을 사용하지 않았다면, 직접 $\nabla_\theta \pi_\theta(s, a) r$을 구해야 했을 것이다. 이 값은 environment에서 움직이는 agent가 얻는 샘플과 별로 관련이 없기 때문에 구하기가 어려웠을 것이다.
- 개발자의 관점
    1. Policy $\pi_\theta$를 정한다.
        - Ex: Softmax Policy
    2. Feature를 정하고, feature에 weight를 곱한 뒤, $\pi_\theta$에 넣는다.
    3. $\pi_\theta$가 출력한 값에 $\log$를 취한다.
    4. Gradient를 계산한다.
        - Ex: TensorFlow의 tf.GradientTape()
    5. Reward를 곱한다.
    6. Environment에서 agent가 policy $\pi_\theta$를 통해 게임을 하면서 얻은 샘플들로 2부터 5를 반복하면서 $\theta$를 업데이트한다.


#### Policy Gradient Theorem

- 이전까지는 1-step MDP에 대해서 이야기했지만, 일반적인 multi-step MDP에서는 어떨까?
    - Policy gradient theorem: Likelihood ratio trick을 이용한 방법론을 multi-step MDP까지 일반화시킨 정리이다.
        - *미분 가능한 policy $\pi_\theta(s, a)$와 어떠한 목적함수 $J = J_1, J_{\text{av}R}, \frac{1}{1 - \lambda}J_{\text{av}V}$에 대해, policy gradient $\nabla_\theta J(\theta) = \mathbb{E_{\pi_\theta}} [\nabla_\theta \log\pi_\theta(s, a) Q^{\pi_\theta}(s, a)]$ 이다.*{: style="color: red"}
            - 어떻게 $r$을 $Q^\pi(s, a)$로 대체할 수 있었을까?
                - 직관적으로 접근해보면, 1-step MDP는 reward $r$을 한 번 받고 끝나므로 $r$이 곧 accumulate reward이다. 반면, multi-step MDP는 각 step에서 action을 선택했을 때 받는 reward를 모두 더해줘야하는데, $Q^{\pi_\theta}(s, a)$가 각 state에서 action을 선택했을 때 받는 reward의 총합을 의미하므로, $r$ 자리에 $Q^{\pi_\theta}(s, a)$를 넣어주면 multi-step MDP에서도 똑같이 동작한다.
        - 목적함수 3개에 적용 가능하다.
        - Score function X $Q^\pi(s, a)$


#### Monte-Carlo Policy Gradient (REINFORCE)

![Monte-Carlo Policy Gradient (REINFORCE)](/assets/rl/reinforce.png)

- $Q^{\pi_\theta}(s, a)$ 자리에 return을 사용한다.
    - Return $v_t$는 $Q^{\pi_\theta}(s, a)$의 unbiased sample이다. 즉, 모분포는 $Q$이고 $r$은 샘플이므로, 계속 샘플링한 $r$의 기댓값은 $Q$가 된다.
        - $\Delta\theta_t = \alpha \nabla_\theta \log\pi_\theta(s_t, a_t)v_t$
            - $v_t$: Accumulate discounted reward
            - step-size $\alpha$를 곱한 만큼 $\theta$를 업데이트한다.
- Pseudo code
    1. $\theta$를 임의로 초기화한다.
    2. $\theta$로 구성된 policy $\pi$로 게임을 진행한다.
    3. 한 에피소드가 끝나면, 첫 번째 state부터 마지막 state를 사용하여 $\theta$를 업데이트한다.
        - 초기화 되었던 $\theta$에 $\alpha \nabla_\theta \log\pi_\theta r$을 더해준다.
    4. 점점 더 좋은 policy $\pi$가 된다.
- Value function을 사용하지 않고 직접 policy를 업데이트하므로, policy-based RL이다.
- $Q$ 자리에 return을 쓰는 Monte-Carlo 방법을 사용하므로, Monte-Carlo policy gradient라고 부르고, REINFORCE 알고리즘이라고도 한다.


#### Puck World Example

![Puck World](/assets/rl/puck_world.png)

- 아이스하키의 puck에 힘을 줘서 움직이는 게임으로, continuous action space 문제이다.
- Target은 30초마다 위치가 바뀌고, target에 가까울수록 reward를 받는다.
- 학습곡선에서 점점 reward가 올라가는 것을 확인할 수 있다.
    - Value-based RL처럼 지그재그하게 흔들리면서 올라가지 않고, stable하게 올라간다.
        - Value-based RL: 가장 높은 가치를 얻는 action을 선택하므로, policy가 급격하게 바뀐다.
        - Policy-based RL: $\theta$를 조금씩 업데이트한다.
    - *x축을 보면 $10^7$부터 서서히 reward가 올라가는 것을 보면 알 수 있듯이, return의 variance가 매우 크기 때문에, 매우 느리게 학습된다는 것을 알 수 있다.*{: style="color: red"}
        - Monte-Carlo에서 return을 사용할 때의 단점이 여기서도 영향을 미친다.
        - TD(0)에서 biased는 있지만, variance가 작다는 성질도 똑같이 영향을 미친다.
        - 그럼 variance는 어떻게 줄일까?
            - *Actor-Critic을 사용한다.*{: style="color: red"}


---

## Actor-Critic Policy Gradient


### Reducing Variance Using a Critic

- Monte-Carlo policy gradient(REINFORCE)는 variance가 높다는 문제가 있었다.
    - REINFORCE: $v_t$ 자리에 return이 들어간다.
- Variance를 줄이기 위해 critic이라는 개념이 들어오게 된다.
    - Critic은 action-value function을 계산한다.
        - $Q_\mathbf{w}(s, a) \approx Q^{\pi_\theta}(s, a)$
            - 전지전능한 신이 $Q^{\pi_\theta}$ 알려주지 않기 때문에, $Q^{\pi_\theta}$를 모방하도록 function approximator $Q_\mathbf{w}(s, a)$를 만들어서, $v_t$ 자리에 넣는다.
    - *Actor-critic 알고리즘*{: style="color: red"}
        - $\begin{aligned}
            \nabla_\theta J(\theta) &\approx \mathbb{E_{\pi_\theta}} \left[ \nabla_\theta \log\pi_\theta(s, a) Q_\mathbf{w}(s, a) \right]    \newline
            \Delta\theta            &= \alpha \nabla_\theta \log\pi_\theta(s, a)Q_\mathbf{w}(s, a)
        \end{aligned}$
            - $Q$는 파라미터 $\mathbf{w}$로 구성되어 있다.
            - $\pi$는 파라미터 $\theta$로 구성되어 있다.
            - *학습 대상: $Q_\mathbf{w}$와 $\pi_\theta$*{: style="color: red"}
                - $Q_\mathbf{w}(s, a)$는 사실 $Q^{\pi_\theta}$를 모방하기 위해 학습하는 중에, actor에서는 $\theta$가 업데이트 되면서 $\pi_\theta$와 $Q^{\pi_\theta}$가 같이 향상된다.
                - Critic에서 향상된 $Q^{\pi_\theta}$를 모방하기 위해 $\mathbf{w}$를 업데이트 해서 $Q_\mathbf{w}(s, a)$를 향상시킨다.
                - 향상된 $Q_\mathbf{w}(s, a)$를 통해 $\theta$가 업데이트 되고, 이것을 계속 반복한다.
                - 이와 같이 $Q_\mathbf{w}$와 $\pi_\theta$가 학습되며, policy iteration과 유사하게 학습된다.


### Estimating the Action-Value Function

- Critic은 policy evaluation 문제를 다루고 있고, 이 문제는 이전 장에서 다루었다.
    - [Monte-Carlo policy evaluation]({% post_url /RL/2021-04-13-model-free-prediction %}#monte-carlo-policy-evaluationprediction)
    - [Temporal-Difference learning]({% post_url /RL/2021-04-13-model-free-prediction %}#temporal-difference-learning)
    - [TD($\lambda$)]({% post_url /RL/2021-04-13-model-free-prediction %}#tdlambda)
- [Least-squares policy evaluation]({% post_url /RL/2021-04-15-value-function-approximation %}#least-squares-prediction)도 사용할 수 있다.
- 자신이 사용하고 싶은 방법으로 $Q_\mathbf{w}$를 학습한다.


### Action-Value Actor-Critic

![Action-Value Actor-Critic](/assets/rl/action_value_actor_critic.png)

- QAC: Q Actor-Critic
    - Critic: Linear TD(0)로 $\mathbf{w}$를 업데이트한다.
        - $Q_\mathbf{w}(s, a) = \phi(s, a)^T \mathbf{w}$
        - Function approximator로 linear combination of features를 사용했다.
    - Actor: Policy gradient로 $\theta$를 업데이트한다.
    - 가장 간단한 Actor-Critic이다.
- Pseudo code
    1. State $s$와 $\theta$를 초기화한다.
    2. Policy $\pi_\theta$를 따르는 action $a$을 샘플링한다.
    3. 매 step마다 reward를 얻고 다음 state $s'$을 방문하면, 다음 state $s'$에서 policy $\pi_\theta$를 따르는 action $a'$을 샘플링한다.
        1. $\mathbf{w}$를 업데이트하기 위해 TD error를 계산한다.
            - $\delta = r + \gamma Q_\mathbf{w}(s', a') + Q_\mathbf{w}(s, a)$
        2. $\theta$를 업데이트한다.
            - $\theta = \theta + \alpha \nabla_\theta \log\pi_\theta(s, a) Q_\mathbf{w}(s, a)$
        3. $\mathbf{w}$를 업데이트한다.
            - $\mathbf{w} \leftarrow \mathbf{w} + \beta\delta\phi(s, a)$
                - $\beta$: Learning rate
                - $\phi(s, a)$: $Q_\mathbf{w}(s, a)$의 features
                    - Neural net을 사용할 경우, [gradient Q]({% post_url /RL/2021-04-15-value-function-approximation %}#convergence-of-control-algorithm)를 적용한다.
- 한 에피소드 안에서 매 step마다 evaluation과 improvement가 동시에 발생하므로, generalized policy iteration의 다른 형태라고 할 수 있다.
    - $\theta$를 어떻게 업데이트하는지를 보면, $\nabla_\theta \log\pi_\theta$를 통해 $\pi_\theta$가 action을 선택할 확률을 업데이트한다. 만약 $Q_\mathbf{w}(s, a_1)$가 좋았으면 $a_1$가 더 좋은 방향으로 업데이트하고, $Q_\mathbf{w}(s, a_2)$가 안좋았으면 $a_2$가 더 안좋은 방향으로 업데이트한다. 즉 action $a$가 좋은 선택이었더면 더 자주 선택하도록 업데이트하고, 좋지 않은 선택이었다면 피하는 선택을 하도록 업데이트한다. 이 때, 좋은지, 좋지 않은지의 지표는 $Q_\mathbf{w}$가 제공한다.


<br>

### Advantage Function Critic

#### Reducing Variance Using a Baseline

- Monte-Carlo로 인한 높은 variance를 줄이기 위해 Critic 개념을 사용했는데, 여기서 variance를 더 줄이는 방법이 있다.
- $\begin{aligned}
    \mathbb{E_{\pi_\theta}} [ \nabla_\theta \log\pi_\theta(s, a)B(s) ] &= \sum_{s \in \mathcal{S}} d^{\pi_\theta}(s) \sum_a \nabla_\theta\pi_\theta(s, a)B(s)       \newline
                                                                       &= \sum_{s \in \mathcal{S}} d^{\pi_\theta}(s) B(s) \nabla_\theta \sum_{a \in \mathcal{A}} \pi_\theta(s, a)     \newline
                                                                       &= 0
\end{aligned}$
    - Baseline $B(s)$: State $s$에 대해 정할 수 있는 임의의 함수
    - Likelihood ratio trick을 역으로 사용해서 원래의 식으로 돌아갔다.
    - Expection이 없어지면서, $\sum_a$에 $\pi$가 추가되었다.
    - $B(s)$는 $\sum_a$의 $a$와 관련이 없으므로, $\sum_a$ 밖으로 빼놓을 수 있다.
    - $\sum_{a \in \mathcal{A}} \pi_\theta(s, a) = 1$이므로 $\nabla_\theta \sum_{a \in \mathcal{A}} \pi_\theta(s, a) = 0$이 된다.
    - Policy gradient에 baseline function $B(s)$를 빼서 계산한다.
        - Policy gradient는 action $a$를 선택할 때 $Q^{\pi_\theta}(s, a)$가 좋으면 더 자주 선택하도록 하고, 좋지 않으면 덜 선택하도록 policy를 바꾼다는 의미이다. 이 때, $a_1$을 선택했을 때 $Q^{\pi_\theta}(s, a_1)$가 1,000,000이고, $a_2$를 선택했을 때 $Q^{\pi_\theta}(s, a_2)$가 999,000이라고 하면, $a_1$을 선택하도록 policy를 업데이트하고 싶을 것이다. 그런데 두 action의 $Q^{\pi_\theta}$가 너무 크기 때문에, 우선 처음에는 두 action에 대해 학습을 한 뒤 $Q^{\pi_\theta}$가 어느정도 수렴하고 상대적인 차이를 알게 될 때, policy가 발전하게 된다.
        - 결국 상대적인 차이가 중요하므로, 수렴할 때까지 기다린 뒤 policy를 발전시키는 것은 비효율적이기 때문에, 똑같이 999,500을 빼주면 $a_1$을 선택할 때의 $Q^{\pi_\theta}(s, a_1)$는 500이 되고 $a_2$를 선택할 때의 $Q^{\pi_\theta}(s, a_2)$는 -500이 되므로 훨씬 학습이 쉽게 이루어진다. 즉 baseline을 빼주어 variance를 줄이는 방법이다.
        - 더 나은 policy란 어떤 state에서 어떤 action의 상대적 우열을 가리는 것이기 때문에, 상대적 차이가 중요하다. $Q^{\pi_\theta}$가 너무 크거나 너무 작을 때, 그것을 학습하기 위해 너무 많은 것을 희생하게 된다.
        - *즉, policy gradient의 $Q^{\pi_\theta}(s, a)$ 자리에 $B(s)$를 넣으면, 이 때의 기댓값이 0이기 때문에, $Q^{\pi_\theta}(s, a) - B(s)$를 넣어도 기댓값이 바뀌지 않으면서 variance를 줄일 수 있다.*{: style="color: red"}
    - *그럼, $B(s)$로 어떤 함수를 사용해야 할까?*{: style="color: red"}
        - $B(s)$는 $s$에 대한 일반적인 함수이기 때문에, value function $V^{\pi_\theta}(s)$을 사용하면 $Q^{\pi_\theta}(s, a)$ 자리에 $Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)$를 넣어도 $\nabla_\theta J(\theta)$와 동일하다.
            - $Q^{\pi_\theta}(s, a)$가 1,000,000과 999,000이고 이들의 상대적인 차이를 1 또는 -1로 계산하고 싶을 때, value $V^{\pi_\theta}(s)$와 $Q^{\pi_\theta}(s, a)$의 차이로 상대적인 차이를 비교할 수 있다. 따라서 $V^{\pi_\theta}(s)$는 좋은 baseline이 된다.
        - Advantage function $A^{\pi_\theta}(s, a)$
            - $\begin{aligned}
                A^{\pi_\theta}(s, a) &= Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)    \newline
                \nabla_\theta J(\theta) &= \mathbb{E_{\pi_\theta}} [\nabla_\theta \log\pi_\theta(s, a) A^{\pi_\theta}(s, a)]
            \end{aligned}$


#### Estimating the Advantage Function (1)

- Advantage function은 policy gradient의 variance를 눈에 띄게 줄일 수 있기 때문에, critic은 advantage function을 estimation해야 할 것이다.
- Policy $\pi$를 학습하기 위해 neural net weight $\theta$ 1개와 $Q^{\pi_\theta}$를 학습하기 위한 neural net weight $\mathbf{w}$가 기본으로 필요하고, advantage function을 계산하기 위해 필요한 $V^{\pi_\theta}$를 학습하기 위한 neural net weight $\mathbf{v}$가 필요하다.
    - $\begin{aligned}
        V_\mathbf{v}(s) &\approx V^{\pi_\theta}(s)  \newline
        Q_\mathbf{w}(s, a) &\approx Q^{\pi_\theta}(s, a)    \newline
        A(s, a) &= Q_\mathbf{w}(s, a) - V_\mathbf{v}(s)
    \end{aligned}$
        - Advantage function을 계산하기 위해 2개의 function approximator와 2개의 파라미터 벡터가 필요하다.
        - Ex: [QAC](#action-value-actor-critic)의 pseudo code에서 $Q$만 업데이트하는 것이 아니라, TD를 사용해서 $V$도 같이 업데이트하면 된다.


#### Estimating the Advantage Function (2)

- *$Q_\mathbf{w}(s, a)$를 없애고, $V_\mathbf{v}(s)$만 사용해서 advantage function을 계산할 수 있다.*{: style="color: red"}
    - 전지전능한 신이 true value function $V^{\pi_\theta}(s)$를 알려준다고 하면, 이 때의 TD error $\delta^{\pi_\theta}$는 unbiased estimation이 되고 advantage function의 샘플들로 계산할 수 있다.
        - $\delta^{\pi_\theta} = r + \gamma V^{\pi_\theta}(s') - V^{\pi_\theta}(s)$
        - $\begin{aligned}
            \mathbb{E_{\pi_\theta}} [\delta^{\pi_\theta} \| s, a ] &= \mathbb{E_{\pi_\theta}} [r + \gamma V^{\pi_\theta}(s') \| s, a] - V^{\pi_\theta}(s)   \newline
                                                                   &= Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)      \newline
                                                                   &= A^{\pi_\theta}(s, a)
        \end{aligned}$
            - TD error $\delta^{\pi_\theta}$의 기댓값은 $r + \gamma V^{\pi_\theta}(s')$의 기댓값에 $V^{\pi_\theta}(s)$를 빼는데, $\mathbb{E_{\pi_\theta}} [r + \gamma V^{\pi_\theta}(s') \| s, a]$ 항은 $s, a$와 관련된 수식이지만 $V^{\pi_\theta}(s)$는 $a$와 관련이 없기 때문에 따로 계산된다.
            - $\mathbb{E_{\pi_\theta}} [r + \gamma V^{\pi_\theta}(s') \| s, a]$ 항은 어떤 state에서 같은 action을 선택해도 transition probability에 의해 다양한 값을 가질 수 있기 때문에 기댓값으로 표현되었다. 이는 어떤 state $s$에서 어떤 action $a$를 선택할 때 받는 reward와 1-step 이후의 value의 기댓값을 더한 것이기 때문에, $Q^{\pi_\theta}(s, a)$로 나타낼 수 있다.
            - TD error $\delta^{\pi_\theta}$의 기댓값은 advantage function $A^{\pi_\theta}$이기 때문에, TD error $\delta^{\pi_\theta}$의 샘플들은 advantage function $A^{\pi_\theta}$의 unbiased 샘플이 된다. $\delta^{\pi_\theta}$의 샘플 한 개는 $A^{\pi_\theta}$와 같지는 않지만, 반복적으로 시행하면 대수의 법칙에 의해 샘플들의 평균은 결국 $A^{\pi_\theta}$와 같아진다.
            - *이렇게 되면, true value function에 대해 advantage function $A^{\pi_\theta}$ 자리에 TD error $\delta^{\pi_\theta}$를 넣을 수 있다.*{: style="color: red"}
                - $\nabla_\theta J(\theta) = \mathbb{E_{\pi_\theta}} [\nabla_\theta \log_\theta \log\pi_\theta(s, a) \delta^{\pi_\theta}]$
                    - *이를 실제로 사용할 때는 true value function을 모방하는 $V_\mathbf{v}$를 사용할 수 있다. 즉 학습을 통해 모방하고 있는 TD error $\delta_\mathbf{v}$를 사용하면, $Q_\mathbf{w}$를 학습할 필요가 없어진다.*{: style="color: red"}
                        - $\delta_\mathbf{v} = r + \gamma V_\mathbf{v}(s') - V_\mathbf{v}(s)$
    - 위와 같은 방법을 따르면, critic 파라미터 $\mathbf{v}$만 사용하게 된다.


<br>


### Eligibility Traces


#### Critics at Different Time-Scales

- MC: $\Delta\theta = \alpha(v_t - V_\theta(s))\phi(s)$
- TD(0): $\Delta\theta = \alpha(r + \gamma V(s') - V_\theta(s))\phi(s)$
- Forward-view TD($\lambda$): $\Delta\theta = \alpha(v_t^\lambda - V_\theta(s))\phi(s)$
- Backward-view TD($\lambda$)
    - $\begin{aligned}
        \delta_t &= r_{t+1} + \gamma V(s_{t+1}) - V(s_t)    \newline
        e_t      &= \gamma \lambda e_{t-1} + \phi(s_t)      \newline
        \Delta\theta &= \alpha\delta_t e_t
    \end{aligned}$


#### Actors at Different Time-Scales

- Critic과 actor을 따로 학습해야 한다.
- $Q$ 또는 $V$를 학습한 뒤, actor을 학습할 때도 다른 시간 크기에 대해 학습할 수 있다.
- Policy gradient: $\nabla_\theta J(\theta) = \mathbb{E_{\pi_\theta}} [\nabla_\theta \log\pi_\theta(s, a) A^{\pi_\theta}(s, a)]$
    - Monte-Carlo policy gradient: $\Delta\theta = \alpha(v_t - V_\mathbf{v}(s_t)) \nabla_\theta \log\pi_\theta(s_t, a_t)$
    - Actor-critic policy gradient: $\Delta\theta = \alpha(r + \gamma V_\mathbf{v}(s_{t+1}) - V_\mathbf{v}(s_t)) \nabla_\theta \log\pi_\theta(s_t, a_t)$


#### Policy Gradient with Eligibility Traces

- Forward-vew TD($\lambda$)와 같이, actor도 time-scale을 섞을 수 있다.
    - $\Delta\theta = \alpha(v_t^\lambda - V_\mathbf{v}(s_t)) \nabla_\theta \log\pi_\theta(s_t, a_t)$
        - $v_t^\lambda - V_\mathbf{v}(s_t)$: Advantage function의 biased estimate이다.
- Backward-view TD($\lambda$)는 책임을 묻는 eligibility trace를 사용한다. 여기서는 score function에 대해 책임 여부를 묻는다는 점에서 table lookup과 차이가 있다.
    - $\begin{aligned}
        \delta &= r_{t+1} + \gamma V_\mathbf{v}(s_{t+1}) - V_\mathbf{v}(s_t)    \newline
        e_{t+1} &= \gamma \lambda e_t + \nabla_\theta \log\pi_\theta(s, a)      \newline
        \Delta\theta &= \alpha\delta_t e_t
    \end{aligned}$


<br>


### Summary of Policy Gradient Algoritms
{: style="color: red"}

![Summary of Policy Gradient Algoritms](/assets/rl/policy_gradient_algorithms.png)

- Policy gradient는 여러가지 형태들이 있고, 각 형태별로 알고리즘 이름을 붙였다.
    - REINFORCE return: $v_t$
        - Critic X
        - Actor O
    - Q Actor-Critic return: $Q^\mathbf{w}(s, a)$
        - Policy gradient theorem
        - REINFORCE의 return은 Monte-Carlo 방법으로 구한 값이기 때문에 variance가 높으므로, variance를 줄이기 위해 $Q^\mathbf{w}(s, a)$를 학습시킨다.
    - Advantage Actor-Critic return: $A^\mathbf{w}(s, a)$
        - $Q^\mathbf{w}(s, a)$의 상대적인 차이가 중요하기 때문에, variance를 더 줄이기 위해 baseline $V_\mathbf{v}(s)$를 통해 advantage function $A^\mathbf{w}(s, a)$을 구한다.
        - $A^\mathbf{w}(s, a) = Q^\mathbf{w}(s, a) - V_\mathbf{v}(s)$
    - TD Actor-Critic return: $\delta$
        - Advantage function을 그대로 사용하려면 $Q^\mathbf{w}$와 $V_\mathbf{v}$를 학습해야 하는데 너무 많은 파라미터가 필요하고 학습도 복잡해지기 때문에, TD error $\delta$를 advantage function 자리에 사용한다.
        - TD error $\delta$는 advantage function의 샘플이다.
    - TD(\lambda) Actor-Critic return: $\delta e$
        - TD Actor-critic은 1-step 이후의 경험을 통해 계산하는데, time-scale을 고려하기 위해 eligibility trace를 적용하여 TD($\lambda$)를 사용한다.
        - Eligibility trace는 score function의 책임을 계산한다.
- 위의 policy gradient는 stochastic gradient ascent 알고리즘을 사용한다.
- Critic은 policy evaluation을 하기 때문에, MC 또는 TD를 통해 학습하면 된다.
