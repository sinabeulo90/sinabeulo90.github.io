---
layout: post
title:  "Lecture 8: Integrating Learning and Planning"
date:   2021-04-20 11:04:49 +0900
category: "Reinforcement Learning"
tags: YouTube
plugins: mathjax
---

David Silver 님의 [Introduction to reinforcement learning](https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) 강의 내용을 [팡요랩 Pang-Yo Lab](https://www.youtube.com/channel/UCwkGvF7xKz2E0Lv-fZ9wv2g)의 [강화학습의 기초 이론](https://youtube.com/playlist?list=PLpRS2w0xWHTcTZyyX8LMmtbcMXpd3s4TU)에서 풀어서 설명하기에, 이를 바탕으로 공부하여 정리한 내용입니다.

- Video: [[강화학습 8강] Integrating Learning and Planning](https://youtu.be/S216ZLuCdM0)
- Slide: [Lecture 8: Integrating Learning and Planning](https://www.davidsilver.uk/wp-content/uploads/2020/03/dyna.pdf)


---

## Introduction


### Model-Based Reinforcement Learning

- 지금까지는 agent가 경험을 쌓으면, 쌓인 경험을 바탕으로 직접 policy를 업데이트 했다.
    - Q-learning
    - Policy gradient
- 여기서는 경험으로부터 model을 직접 학습하고, 학습한 model로 planning 할 것이다. 그리고 learning과 planning을 합칠 것이다.
    - Planning: Model이 있을 때, value function과 policy를 구하는 과정을 의미한다.


### Model-Based and Model-Free RL

- Model-free RL: Model 없이 경험들로부터 직접적으로 value function 또는 policy를 learning 하는 방법이다.
- Model-based RL: 먼저 경험으로부터 model을 학습하고, 학습된 model을 바탕으로 value function 또는 policy를 planning 하는 방법이다.


### Model-Free RL

![Model-Free RL](/assets/rl/model_free_rl.png)

- Action을 선택하면, environment에서 다음 state와 reward를 반환한다. 그리고 이 정보를 바탕으로 action을 선택한다. 이 과정에서 쌓인 경험들로부터 학습한다.


### Model-Based RL

![Model-Based RL](/assets/rl/model_based_rl1.png)

- 실제 environment를 모방한 model과 상호작용하면서 학습한다. Model은 실제 environment를 정확히 모방할수록 좋지만, 조금 다른 부분이 있을 수 있다.


---

## Model-Based Reinforcement Learning


### Model-Based RL

![Model-Based RL](/assets/rl/model_based_rl2.png)

- Policy를 따라 action을 선택해서 경험이 생기면, 이 경험을 바탕으로 직접 value function 또는 policy를 업데이트 하는 것을 direct RL이라고 한다.
- 경험으로부터 model을 learning하고, 학습된 model로부터 planning하여 value function 또는 policy를 업데이트 하는 것을 model-based RL이라고 한다.
    - Planning: 여기서 MDP를 푸는 것은 model을 알기 때문에 planning이라고 한다.
    - Model learning: 경험으로부터 MDP를 학습하는 것이다.
    - Fully known MDP가 아니지만, MDP를 learnning하고, 학습된 MDP를 planning하여 MDP를 푼다.


### Advantages of Model-Based RL

- 장점
    - 최적의 policy를 찾기 매우 어렵지만, model 자체가 매우 간단한 경우가 있다.
        - Ex: 체스의 model은 게임의 규칙을 의미한다. 어떤 state에서 어떤 action을 선택할 때 어떻게 움직이는지, 상대편 말을 잡을 수 있는지 등 model이 간단하기 때문에, supervised learning 방법으로 쉽게 학습시킬 수 있다.
    - Model의 불확실성을 활용할 수 있다.
        - Model을 학습시킨다는 것은 environment가 어떻게 작동하는지 이해하는 것이다. 즉, model을 만들 때, environment에 대해 확실히 아는 부분과 모르는 부분이 있을 것이다. 확실히 안다는 것은 해당 state에 대해 방문 횟수가 많아서 다음 state가 어떨지 잘 알고 있다는 것이고, 잘 모른다는 것은 해당 state에 대해 방문 횟수가 적어서 다음 state가 어떨지 잘 모른다는 것이다. 이 때, 높은 정확도를 갖고 예측할 때와 낮은 정확도를 갖고 예측할 때 학습에 미치는 영향도 달라져야 할 텐데, 이 정확도에 따라 가중치를 가변적으로 부여해서 학습시킬 수 있다.
- 단점
    - 2단계에 거쳐서 학습이 진행되기 때문에, 추정 오차가 발생할 원인이 2개로 늘어난다.
        - 경험으로부터 바로 배우는 것이 아니라, 먼저 model을 학습시키고, 학습된 model로부터 value function과 policy의 최적을 찾는다.
        - Model learning 또는 planning이 잘못되면, 최적의 value function/policy에서 멀어지게 된다.


### Learning a Model


#### What is a Model?

- Model $\mathcal{M}$: MDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R} \rangle$을 파라미터 $\eta$로 표현한 것이다.
    - $\mathcal{S}$: State의 집합
    - $\mathcal{A}$: Action의 집합
    - $\mathcal{P}$: Transition probability
        - 어떤 state에서 어떤 action을 선택했을 때, 이동할 다음 state에 대한 확률분포
        - 같은 state에서 같은 action을 선택하더라도, 이 확률분포에 따라 다른 state로 이동할 수 있다.
    - $\mathcal{R}$: Reward의 집합
    - $\eta$
        - Ex: Neural net의 경우, weight는 $\eta$가 된다.
- State space $\mathcal{S}$와 action space $\mathcal{A}$를 안다고 가정할 때, model $\mathcal{M}$은 $\langle \mathcal{P_\eta}, \mathcal{R_\eta} \rangle$로 표현된다.
    - $\begin{aligned}
        \mathcal{P_\eta} &\approx \mathcal{P}   \newline
        \mathcal{R_\eta} &\approx \mathcal{R}
    \end{aligned}$
    - $\begin{aligned}
        S_{t+1} &\sim \mathcal{P_\eta}(S_{t+1} \| S_t, A_t)     \newline
        R_{t+1} &= \mathcal{R_\eta}(R_{t+1} \| S_t, A_t)
    \end{aligned}$
        - $S_{t+1}$: 확률분포 $\mathcal{P_\eta}$에 의해 선택된 다음 state
        - $R_{t+1}$: 확률분포 $\mathcal{R_\eta}$에 의해 선택된 reward
        - $S, R$은 서로 관계가 없으므로, 독립적으로 구할 수 있다.
            - $\mathbb{P}[S_{t+1}, R_{t+1} \| S_t, A_t] = \mathbb{P}[S_{t+1} \| S_t, A_t]\mathbb{P}[R_{t+1} \| S_t, A_t]$
    

#### Model learning
{: style="color: red"}

- 경험 $\\{ S_1, A_1, R_2, \dots, S_T \\}$로 어떻게 model $\mathcal{M}$을 학습시킬까?
    - 우선 transition $\langle s, a, r, s' \rangle$을 쌓아두고, supervised learning으로 푼다.
        - Supervised learning: Input과 output이 주어졌을 때, 이 둘의 관계를 학습시키는 방법이다. 여기서는 $r, s'$라는 답이 있기 때문에 가능한 학습 방법이다.
        - $s, a \rightarrow r$은 regression problem이다.
        - $s, a \rightarrow s'$은 density estimation problem이다.
    - Loss function을 정의하여, 이 loss를 줄이는 방향으로 파라미터 $\eta$를 업데이트한다.
        - Loss function: 추정값과 기댓값의 차이를 정의한다.
            - $s_1, a_1$을 입력으로 주어졌을 때, $r'$이 출력되었다고 하자. 경험에서는 $r_2$가 출력되어야 하는데 $r'$이 출력되었으므로, 이 둘 차이를 계산하는 함수를 정의한다.
            - Ex: Mean-squared error, KL divergence(확률 분포의 차이를 표현)
        - Gradient descent 방법론 등을 사용해서 가장 작은 loss를 출력하는 파라미터 $\eta$를 찾는다.
    

#### Examples of Models
- Table Lookup Model
    - Reward와 state transition probability를 테이블에 채워나가는 방식이다.
- Linear Expectation Model
- Linear Gaussian Model
- Gaussian Process Model
- Deep Belief Network Model
- Neural net Model


#### Table Lookup Model

- Model은 $\hat{\mathcal{P}}$와 $\hat{\mathcal{R}}$을 구해야 한다.
    - $\begin{aligned}
        \hat{\mathcal{P}}^a_{ss'} &= \frac{1}{N(s, a)} \sum_{t=1}^T \mathbb{1}(S_t, A_t, S_{t+1} = s, a, s')    \newline
        \hat{\mathcal{R}}^a_s &= \frac{1}{N(s, a)} \sum_{t=1}^T \mathbb{1}(S_t, A_t = s, a)\mathcal{R_t}
    \end{aligned}$
        - $\frac{1}{N(s, a)}$: 해당 state-action pair를 몇 번 선택했는지를 저장한다.
        - $\hat{\mathcal{P}}^a_{ss'}$: 해당 state $s$에서 action $a$를 선택했을 때, 다음 state $s'$으로 이동할 평균 확률 계산
        - $\hat{\mathcal{R}}^a_s$: 해당 state $s$에서 action $a$를 선택했을 때의 평균 reward 계산
- 또는, experience tuple $\langle s, a, r, s' \rangle$를 계속 기록해 두고, 어떤 state $s$에서 action $a$를 선택했을 때, 기록되어있는 데이터 중 $s, a$로 시작하는 tuple을 uniform random하게 샘플링할 수도 있다. 이렇게 해도, 위에서 직접 계산하는 것과 거의 일치히다.
- 단점: Agent가 직접 경험하지 못한 state transition probability는 알 수 없다.
    - 이 경우, neural net을 사용하게 되면, 경험한 transition을 바탕으로 경험하지 못한 transition에 대해서도 interpolation하면서 예측할 수 있다. 물론 정확도의 문제는 남아있다.


#### AB Example

![AB Example](/assets/rl/ab_mdp.png)

- State A B가 있고, 8개의 에피소드가 아래와 같이 있다고 하자.
    1. A, 0, B, 0
    2. B, 1
    3. B, 1
    4. B, 1
    5. B, 1
    6. B, 1
    7. B, 1
    8. B, 0
- 이 때, model은 어떻게 될까?
    - State A에서 state B로 가는 샘플이 1개만 있기 때문에, 항상 A에서 B로 이동한다.
    - State B에서 terminal state로 이동할 때, 샘플 8개 중 6개는 reward 1을 얻고, 나머지 2개는 reward 0을 얻기 때문에, 75% 확률로 reward 1을 얻고, 25% 확률로 reward 0을 얻으면서 종료된다.
    - 위 방법으로 경험을 통해 table lookup model을 만들 수 있다.


<br>


### Planning with a Model


#### Planning with a Model

- Model $\mathcal{M} = \langle \mathcal{P_\eta}, \mathcal{R_\eta} \rangle$이 만들어지면, 아래의 마음에 드는 planning algorithm을 사용해서 MDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{P_\eta}, \mathcal{R_\eta} \rangle$을 풀 수 있다.
    - Value iteration, Policy iteration
        - Dynamic programming MDP를 알 때, 사용할 수 있는 알고리즘이다.
        - MDP를 모델링한 가상의 MDP를 알기 때문에, 이 방법론을 그대로 적용할 수 있다.
    - Tree search
    - ...


#### Sample-Based Planning

- Model을 통해 MDP를 추측할 수 있는데, MDP를 모른다고 가정하고 model-free RL 방법론을 사용하는 것이다.
    - $\mathcal{P_\eta}, \mathcal{R_\eta}$를 통해 샘플을 만들면, 이 샘플들을 environment에서 얻은 경험으로 간주하고 model-free RL 방법론을 사용한다.
- Dynamic programming 방법론과 model-free 방법론을 모두 사용할 수 있기 때문에, 굉장히 간단하지만 매우 강력한 planning 방법이다.
    - Dynamic programming 방법론: Agent가 이동할 수 있는 모든 state를 고려해서 value를 업데이트하는, naive full-width backup 방법이다.
        - Sample-based planning에서 사용할 경우, 자주 겪는 경험일수록 더 많이 샘플링되고, 해당 경험들을 더 집중적으로 업데이트 하므로 curse of dimensionality를 깰 수 있다.
            - Curse of dimensionality: 차원의 저주
                - Full-width backup을 할 경우, 어떤 문제에 대해 1차원씩 더 복잡해질수록 계산복잡도가 exponential하게 늘어난다.
        - Naive full-width backup은 모든 state에 대해 brute-force로 계산하므로, 자주 겪는 경험들과 자주 겪지 않는 경험들을 대등하게 계산한다. 반면, 1만 번 샘플링을 통해 어떤 state에서 다음 state로 8,000번 이동하고, 나머지 2,000번은 나머지 state로 골고루 이동한 경험을 모델링 했을 때, 자주 겪은 transition에 대해 더 정확하게 업데이트 되면서 차원의 저주를 해결하게 된다.
        - Ex: 바둑에서 현재 70수이고, 게임이 끝나기까지 200수가 더 필요하다고 하자. 그러면 다음 한 수로 둘 수 있는 경우의 수는 200이고, 다음 한 수로 둘 수 있는 경우의 수는 199, 이렇게 계속 exponential하게 늘어난다. 따라서 이 모든 경우를 시도할 수 없을 것이다. 이 때, 게임을 10만 번 진행한 뒤, 각 수에 대해서 평균을 냈을 때, 해당 수에서 이길 수 있는 확률을 알게 되어 차원의 저주를 해결할 수 있다.


#### Back to the AB Example

- 실제 경험을 통해 table lookup model을 만들고, 해당 model에 대해서 샘플링한 경험이 아래와 같이 있다고 하자.
    1. B, 1
    2. B, 0
    3. B, 1
    4. A, 0, B, 1
    5. B, 1
    6. A, 0, B, 1
    7. B, 1
    8. B, 0
- 샘플링한 경험으로 model-free RL을 한다.
    - Ex: Monte-Carlo learning: $V(A) = 1, V(B) = 0.75$
- 실제 경험은 8개 밖에 없어도, 이 경험을 통해 model을 갖게 되는 순간, 새로운 경험들을 계속 생성할 수 있다.


#### Planning with an Inaccurate Model

- Model을 통해서 학습을 진행하는데, model이 environment와 조금씩은 다를 것이다.
- Model이 완전히 같지 않다면, model-based RL의 성능은 approximate MDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{P_\eta}, \mathcal{R_\eta} \rangle$의 optimal policy에 국한될 것이다.
    - 즉, model이 좋으면 좋을수록 planning은 그만큼 더 좋은 policy를 계산하고, model이 부정확하다면 suboptimal policy를 계산할 것이다.
- Model이 틀릴 때는 어떻게 학습해야 할까?
    1. Model을 버리고, model-free RL을 사용한다.
    2. Model을 학습할 때, 얼마나 신뢰할 수 있는지 정도를 같이 학습한다.
        - Ex: Model에 어떤 구간을 출력한다고 가정하면, 구간분포가 넓을수록 그만큼 불확실하다는 것을 의미할 것이다. 실제 reward는 30인데, model에서 reward가 0 ~ 100 구간에 있다고 말하는 것과 29.5 ~ 30.5 구간에 있다고 말할 때, 불확실한 정도를 알 수 있을 것이다.
        - Ex: Bayesian model-based RL: Bayesian 방법론을 통해 prior 분포를 두고, 더 많은 경험을 통해 이 분포를 점점 정교하도록 업데이트하면서 불확실성(variance)을 이용해 학습한다.


---


## Integrated Architectures

### Dyna

#### Real and Simulated Experience

- 경험은 2가지로 얻을 수 있다.
    1. Real experience: Environment(true MDP)에서 샘플링된 경험
        - $\begin{aligned}
            S' &\sim \mathcal{P_{ss'}^a} \newline
            R &= \mathcal{R_s^a}
        \end{aligned}$
    2. Simulated experience: Model(approximate MDP)에서 샘플링된 경험
        - $\begin{aligned}
            S' &\sim \mathcal{P_\eta}(S' \| S, A) \newline
            R &= \mathcal{R_\eta}(R \| S, A)
        \end{aligned}$


#### Integrating Learning and Planning

- Model-Free RL: Model 없이 실제 경험으로부터 value function을 학습한다.
- Model-Based RL: 실제 경험으로부터 model을 학습하고, 학습된 model에서 얻은 경험으로부터 value function을 계획한다.
- Dyna: 실제 경험으로부터 model을 학습하고, 두 종류의 경험으로부터 value function을 학습과 계획을 한번에 한다.


#### Dyna Architecture


![Dyna Architecture](/assets/rl/dyna_architecture.png)

- [Model-Based RL](#model-based-rl-1)에서는 direct RL 화살표가 없었다. Experience는 model을 만들기 위해서만 사용되고, value function과 policy를 학습시키기 위해 사용되지 않았다.
- Dyna에는 direct RL과 model을 학습시킨 뒤의 planning도 함께 존재한다.


#### Dyna-Q Algorithm

![Dyna-Q Algorithm](/assets/rl/dyna_q_algorithm.png)

- Table lookup 방법이다.
    - Function approximator를 사용할 수 있다.
- 실제 경험으로부터 1번 learning하고, model에서 시뮬레이션된 경험으로부터 $n$번 planning 한다.
    - Ex: $n = 50$
    - (a) ~ (e): Real experience로 업데이트한다.
    - (f): Simulated experience로 업데이트한다.
- Environment과 쉽게 상호작용할 수 있다면, 실제 경험을 통해 학습하면 된다. 반면, environment와 상호작용이 어렵고 비용이 많이 들 경우, 한번 경험을 얻고, 얻은 경험으로부터 model을 만들어서 시뮬레이션된 경험을 통해 학습하면 된다.
- Pseudo code
    1. 모든 state와 action에 대해, $Q$와 model을 random 값으로 초기화한다.
    2. 현재 state $S$에서 $\epsilon$-greedy를 통해 action $A$를 선택한 뒤, reward $R$를 얻고 다음 state $S'$으로 이동한다.
    3. $S, A, R, S'$을 통해 $Q$를 업데이트한다.
        - $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \arg\max_a Q'(S', a) - Q(S, A)]$
            - Model-free RL의 일반적인 Q-learning 방법론을 사용했다.
    4. $S, A, R, S'$을 통해 Model을 학습한다.
        - $Model(S, A) \leftarrow R, S'$
            - 쉽게 학습시키기 위해, deterministic environment라고 가정한다.
    5. Planning을 통해 $Q$를 $n$번 업데이트한다.
        - $\begin{aligned}
            S &\leftarrow \text{random previously observed state}    \newline
            A &\leftarrow \text{random action previously taken in } S    \newline
            R, S' &\leftarrow Model(S, A)    \newline
            Q(S, A) &\leftarrow Q(S, A) + \alpha [R + \gamma\max_a Q(S', a) - Q(S, A)]
        \end{aligned}$
            - 한번 방문했던 state를 뽑아서, 해당 state에서 선택했던 action을 뽑는다.
            - 그리고, model을 통해 샘플링된 $S, A$에 대한 reward $R$과 다음 state $S'$를 뽑는다.
            - 샘플링된 transition $\langle S, A, R, S' \rangle$을 통해 model-free RL과 같은 방식으로 업데이트한다.


#### Dyna-Q on a Simple Maze

![Dyna-Q on a Simple Maze](/assets/rl/dyna_q_on_a_simple_maze.png)

- S에서 시작해서 G에 도착해야 끝나는 게임이다.
    - Planning을 사용하지 않을 경우, 많은 step을 거쳐야 한 에피소드가 끝난다.
    - Planning을 사용할 경우, 몇 번 step을 거치면 한 에피소드가 끝난다.
        - 50 planning steps인 경우, 실제 경험의 60배나 효율적으로 학습에 사용했다. 즉 아주 적은 경험을 짜내서(squeeze) model을 만들고, model에서 시뮬레이션된 경험으로부터 학습을 진행했다.
- 0 planning steps(direct RL)과 5 planning steps의 차이가 극심하고, direct RL이 학습을 진행하는 동안 5 planning steps과 50 planning steps은 이미 최적의 value function에 도달했다.
- x축이 1일 때, 3가지 RL의 step per episode는 어땠을까?
    - 이 문제의 경우, Direct RL과 Dyna-Q는 차이가 없다.
    - 만약 벽에 부딪힐 때 reward가 -값을 얻고 다시 S로 되돌아온다면, Dyna-Q의 경우, G에 도착할 때까지 벽에 부딪히면 안된다는 것을 planning으로 배우기 때문에, 첫 에피소드에서 더 빨리 도착할 것이다.


#### Dyna-Q with an Inaccurate Model

![Dyna-Q with an Inaccurate Model](/assets/rl/dyna_q_with_an_inaccurate_model_1.png)

- Dyna를 사용할 때, model이 틀릴 경우의 예시이다.
    - 1000번째 step에서 environment가 S에서 G로 가는 경로를 더 어렵게 바꿨다.
- Dyna-Q+가 더 빨리 배우는 이유는, action을 선택할 때 exploration을 더 잘 하도록 설정했기 때문이다.


#### Dyna-Q with an Inaccurate Model (2)

![Dyna-Q with an Inaccurate Model](/assets/rl/dyna_q_with_an_inaccurate_model_2.png)

- Environment가 S에서 G로 가는 경로를 더 쉽게 바꿨다.


---

## Simulation-Based Search

- Planning을 좀 더 효과적으로 하는 방법에 대해 다룬다.


### Forward Search

![Forward Search](/assets/rl/forward_search_tree.png)

- 위의 search tree는 모든 state에서 2가지 action만 선택할 수 있는 상황이다.
    - 흰색 원: State
    - 검은색 원: Action
- State space가 매우 큰 상황에서 planning을 할때는 사실 현재 위치한 state에 대해서만 관심이 있지, 먼 미래의 다른 state에 대해서는 관심이 없다. 
    - Ex: 바둑에서 170수까지 진행된 상황에서, 처음 30수는 어디에 두어야할 지는 170수인 상황과 관련이 없다.
- 현재 state의 상황이 더 중요하기 때문에, 현재부터 미래만 보겠다는 것이 forward search의 철학이다.
    - Lookahead를 통해 best action을 선택한다.
    - 현재 state $s_t$를 root로 가지는 search tree를 만든다. Planning을 할 수 있으므로, MDP model을 이용해서 이용해서 lookahead search tree를 만든다.
    - MDP전체를 풀 필요 없이, 현재 state $s_t$부터 이어지는 sub-MDP만 푼다.


#### Simulation-Based Search

![Simulation-Based Search](/assets/rl/simulation_based_search.png)

- Simulation-based search: Forward search + model-free RL
    - Simulation-based search는 결국 forward search이다.
- Sample-based planning을 통해, 현재 state $s_t$를 기준으로 미래의 상황들을 forward search 한다.
    1. Model을 통해 현재 state $s_t$에서 시작하는 여러 에피소드들을 생성한다.
        - $\\{ s_t^k, A_t^k, R_{t+1}^k, \dots, S_T^k \\}^K_{k=1} \sim \mathcal{M_\mathbf{v}}$
    2. 생성된 에피소드들에 대해 model-free RL을 사용한다.
        - Monte-Carlo sarch: Monte-Carlo 방법론을 사용한 simulation-based search
        - TD sarch: Sarsa 방법론을 사용한 simulation-based search


<br>


### Monte-Carlo Search

#### Simple Monte-Carlo Search

- Simulation-based search + Monte-Carlo control
- Model과 simulation policy $\pi$가 주어졌다고 가정하자.
    1. 모든 action $a$에 대해, 실제 environment에서 state $s_t$에서 시작하는 시뮬레이션된 에피소드를 $K$개 만든다.
        - $\\{ s_t, a, R_{t+1}^k, \dots, S_T^k \\}^K_{k=1} \sim \mathcal{M_\mathbf{v}}, \pi$
        - Ex: 만약 action space가 10이고, $K$가 100일 경우, 총 1,000개의 에피소드를 만든다.
    2. 모든 $K$개 에피소드에 대해 평균을 계산한다.(Monte-Carlo evaluation)
        - $Q(s_t, a) = \frac{1}{K} \sum_{k=1}^K G_t \xrightarrow[]{P} q_\pi(s_t, a)$
            - Policy $\pi$로 simulation했기 때문에, 각 action에 대해 return의 평균은 $q_\pi$가 된다.
    3. 가장 큰 $Q(s_t, a)$를 가지는 action을 실제 environment에서 선택한다.
        - $a_t = \arg\max_{a \in \mathcal{A}} Q(s_t, a)$


#### Monte-Carlo Tree Search (Evaluation)

- MCTS: RL의 planning에 속하면서, 그 중 forward search 에 속하고, 그 중 simulation-based search에 속하고, 그 중 Monte-Carlo search에 속한다.
    1. 같은 모델이 주어졌을 때, [Simple Monte-Carlo Search](#simple-monte-carlo-search)처럼 모든 action에 대해서 $K$개의 에피소드를 만드는 것이 아니라, 현재 policy $\pi$를 통해 $K$개의 에피소드를 만든다.
        - $\\{ s_t, R_t^k, R_{t+1}^k, \dots, S_T^k \\}^K_{k=1} \sim \mathcal{M_\mathbf{v}}, \pi$
        - Policy $\pi$에 따라 특정 action을 더 많이 선택하게 되므로, 해당 action을 포함한 경험들이 더 많이 뽑히게 된다.
        - 에피소드를 만들 때, root 노드를 가지는 현재 state $s$에 대한 tree만 만들어지는 것이 아니라, 어떤 action을 선택했을 때 방문한 다른 state $s'$들도 search tree에 포함된다.
    2. 현재 state $s$에서 선택할 수 있는 각 action $a$에 대해 mean return으로 $Q(s, a)$를 구한다.
        - $Q(s, a) = \frac{1}{N(s, a)} \sum_{k=1}^K \sum_{u=t}^T \mathbb{1}(S_u, A_u = s, a) G_u \xrightarrow[]{P} q_\pi(s, a)$
    3. Search가 끝나면, 가장 큰 $Q(s, a)$를 가지는 action을 실제 environment에서 선택한다.
        - $a_t = \arg\max_{a \in \mathcal{A}} Q(s_t, a)$


#### Monte-Carlo Tree Search (Simulation)

- [Monte-Carlo Search](#simple-monte-carlo-search)에서 simulation policy $\pi$가 고정되어 있는데, simulation policy $\pi$도 같이 향상되면 더 좋을 것이다.
    -  모델에서 생성되는 경험을 통해 planning하고 있는데, 이 경험을 이용해서 simulation policy $\pi$를 향상시킬 수 있다.
- 시뮬레이션은 아래 2단계로 구성된다. (in-tree, out-of-tree)
    1. Tree policy(improves): 가장 큰 $Q(S, A)$를 가지는 action을 선택한다.
    2. Default policy(fixed): Random하거나 고정된 policy를 따르는 행동을 선택한다.
        - Tree를 생각해보면, model을 통해 생성한 영역과 생성되지 않은 바깥 영역이 있을 것이다. 그 중 model이 생성한 영역은 tree에 대한 정보를 알기 때문에 $Q$를 계산할 수 있지만, 바깥 영역은 그에 대한 정보를 모르기 때문에 default policy를 통해 영역을 확장한다.
- 매 시뮬레이션마다 아래 과정을 반복한다.
    1. Evaluate: Monte-Carlo 방법으로 해당 state의 $Q(S, A)$ 평가한다.
    2. Improve: $\epsilon$-greedy($Q$)를 통해 tree policy를 향상시킨다.
        - 매 시뮬레이션마다 $Q$를 평가하므로 점점 더 정확해지고, 이 과정에서 가장 큰 $Q$를 가지는 action을 선택하므로 policy도 점점 향상된다.
        - Policy iteration과 유사하게 동작한다.
- Simulated experience에 Monte-Carlo control을 적용한 것으로, 이는 optimal search tree로 수렴한다.
    - $Q(s, a) \rightarrow q_\*(s, a)$


<br>


### MCTS in Go


#### Case Study: the Game of Go

- 바둑은 2500년 된 가장 어려운 보드게임이고, 당시 grand challenge였다.
- 전통적인 game-tree search로 풀려는 시도가 있었지만 실패했다.


#### Rules of Go

- 바둑의 규칙은 19x19 보드에서 진행되며, 규칙은 간단하지만 전략이 복잡하다.
- 검은 색과 흰 색의 영역 싸움이며, 더 많은 영역을 차지하는 플레이어가 이기는 게임이다.


#### Position Evaluation in Go

- MCTS를 이용해서 어떤 position이 얼마나 좋은지를 평가한다.
- Reward function(undiscounted)
    - $\begin{aligned}
        R_t &= 0 \ \text{for all non-terminal steps } t < T  \newline
        R_T &= \begin{cases}
                    1 \ \text{if Black wins}    \newline
                    0 \ \text{if White wins}
                \end{cases}
    \end{aligned}$
        - 1: 게임이 끝나고 검정 색이 이길 경우
        - 0: Terminal state에 도착하지 않거나, 게임이 끝나고 흰 색이 이길 경우
- Policy $\pi = \langle \pi_\text{B}, \pi_\text{W} \rangle$
- 양 쪽 플레이어가 번갈아가면서 게임을 진행하고, 어떤 position $s$가 얼마나 좋은지 나타내는 value function은 $s$에서 black이 이길 확률로 정의했다.
    - $v_\pi(s) = \mathbb{E_\pi} [R_T \| S = s] = \mathbb{P} [\text{Black wins} \| S = s]$
- Optimal value function에서 흰 색 policy $\pi_w$의 $w$는 value function을 최대한 낮추도록 학습하고, 검은 색 policy $\pi_b$의 $b$는 최대한 높이도록 학습한다. 즉, 각 플레이어의 policy가 서로 싸우는 형태로 학습한다.
    - $v_\*(s) = \max_{\pi_\text{B}}\min_{\pi_\text{W}} v_\pi(s)$
        - Min-max, Mini-max 게임


#### Monte-Carlo Evaluation in Go

![Monte-Carlo Evaluation in Go](/assets/rl/monte_carlo_evaluation_in_go.png)

- 현재 state에서 각 플레이어가 각자의 policy를 통해 번갈아 게임을 진행하면서 4개의 에피소드를 관찰했을 때, 결과가 1, 1, 0, 0이 나왔다고 하자.
    - 이 때 현재 state에서 검은 돌이 이길 확를은 0.5라고 평가한다.
- Policy가 좋을수록 더 좋지 않을까?
    - 당연히 좋을수록 좋다.
    - MCTS의 놀라운 점은, random policy를 사용해도 꽤 정확하다.


#### Applying Monte-Carlo Tree Search (1)

![Monte-Carlo Tree Search](/assets/rl/monte_carlo_tree_search_1.png)

- Tree policy: 현재 state부터 tree를 만들때 사용하는 policy
- Default policy: 현재 state부터 만들어진 tree 밖의 부분들에서 사용하는 policy


#### Applying Monte-Carlo Tree Search (2)

![Monte-Carlo Tree Search](/assets/rl/monte_carlo_tree_search_2.png)

- 2번째 에피소드에서 흰 색이 이겼으므로, current state는 전체 에피소드 2개 중 검은색이 이긴 에피소드는 1개이므로, value는 0.5가 된다.
- current state의 다음 노드는 전체 에피소드 1개 중 검은색이 이긴 에피소드는 0개이므로, value는 0이 된다.


#### Applying Monte-Carlo Tree Search (3)

![Monte-Carlo Tree Search](/assets/rl/monte_carlo_tree_search_3.png)

- 3번째 에피소드에서 다른 행동을 선택하였으므로, current state에서 다른 노드가 추가된다.
- 이와 같은 방식으로 트리가 점점 확장된다.


#### Applying Monte-Carlo Tree Search (4)

![Monte-Carlo Tree Search](/assets/rl/monte_carlo_tree_search_4.png)


#### Applying Monte-Carlo Tree Search (5)

![Monte-Carlo Tree Search](/assets/rl/monte_carlo_tree_search_5.png)

- Current state를 거치는 에피소드 수는 계속 늘어날 것이고, sub-tree의 노드들의 갯수도 늘어나면서 tree가 점점 정확해진다.
    - Tree policy: 생성된 tree안에서 행동을 선택할 때, 가장 큰 $Q(s, a)$를 가지는 action을 선택한다.
        - Ex: Current state의 왼쪽 노드 value는 2/3이고, 오른쪽 노드 value는 0/1이므로, 왼쪽 노드로 이동하는 행동을 선택한다.
        - Exploration이 포함된 $\epsilon$-greedy policy로 인해, tree policy가 점점 향상된다.
    - Default policy: 에피소드가 끝날때 까지 random하거나 고정된 policy를 따르는 행동을 선택한다.


#### Advantages of MC Tree Search

- Current state를 거치는 에피소드를 반복적으로 만들면서 계산되므로, current state는 매우 정확하다.
- State를 dynamically하게 계산한다.
    - Full-width backup을 하는 Dynamic programming과는 다른 의미이다.
    - Dynamically하게 계산한다는 것은 매번 해당 state에 방문할 때마다 계산한다는 것을 의미한다.
        - Ex: 10개의 state가 있다고 할 때, $s_3$에서 $s_1$을 평가하는 것과 $s_8$에서 $s_1$을 평가하는 것은 달라질 수 있다. 왜냐하면 모든 state를 한번에 같이 평가하지 않기 때문이다. 매번 모든 state를 고려해서 평가하면 동일한 값을 계산하겠지만, MCTS는 현재 처한 상황을 기준으로 집중적으로 더 많이 평가하기 때문에, 위치한 state에 따라 평가가 달라진다.
- 샘플링을 통해 학습하므로, 차원의 저주를 깨뜨린다.
- 모델이 어떻게 학습되었는지는 상관없이, query만 날릴 수 있으면 된다. 즉 샘플링을 위해 $s, a$를 입력하면 $r, s'$을 출력하기만 하면 된다.
- 계산이 효율적이며, 병렬성도 좋다.


#### Example: MC Tree Search in Computer Go

![MC Tree Search in Computer Go](/assets/rl/mc_tree_search_in_computer_go.png)


<br>


### Temporal-Difference Search


#### Temporal-Difference Search

- MC를 사용하냐, TD를 사용하냐는 중요하지 않다. 중요한 것은 현재 상황에 국한지어서 forward search에 더 특화된 policy를 만들고, 이 때 효과가 있는지가 중요한 것이다.
- TD search: Simulation-based search의 MC자리에 TD를 넣은 것이다.
    - MCTS는 현재 상황으로부터 생성되는 sub-MDP에 MC control을 적용한 것이다.
    - TD search는 sub-MDP에 Sarsa를 적용한다.


#### MC vs. TD search

- Model-free RL에서의 TD learning(bootstrapping)
    - Variance가 감소되지만, bias가 증가한다.
    - 일반적으로 MC보다 더 효과적이고, TD($\lambda$)는 훨씬 더 효과적일 수 있다.
- Simulation-based search에서의 TD learning(bootstrapping)
    - Planning을 통해 생성된 샘플에 같은 방법을 적용하므로, model-free RL에서와 같은 효과를 볼 수 있다.


#### TD Search

- Model로부터 현재 state부터 시작하는 에피소드들을 생성하고, 시뮬레이션된 에피소드들의 매 step마다 Sarsa를 활용해서 action-value function을 업데이트한다.
    - $\Delta Q(s, a) = \alpha (R + \gamma Q(S', A') - Q(S, A))$
        - $\alpha$: Learning rate
    - $\epsilon$-greedy를 통해 action을 선택한다.
    - $Q$를 function approximator로 표현할 수 있다.


#### Dyna-2

- Dyna의 또 다른 버전이다.
- Feature weight를 2개 사용한다.
    1. Long-term memory: TD learning을 통해, real experience로부터 학습한다.
        - 모든 에피소드에 적용되는 일반적인 도메인 정보를 활용한다.
    2. Short-term (working) memory: TD search를 통해, simulated experience로부터 학습한다.
        - 현재 상황에 대한 특정 local 정보를 활용한다.
- Value function은 long-term memory와 short-term memory의 합으로 나타낸다.


#### Results of TD search in Go

![Results of TD search in Go](/assets/rl/results_of_td_search_in_go.png)

- Dyna-2: TD learning + TD search
    - TD learning에는 search를 사용하지 않고 real experience만으로 학습하므로, planning에 속하지 않는다.
    - 그래프를 통해, 실제 경험 만으로 학습하는 것은 좋지 않다는 것을 확인할 수 있었다. 그리고 그만큼 search도 중요하다는 것을 알려주고 있다.
- UCT: Upper Confidence bounds applied to Trees
