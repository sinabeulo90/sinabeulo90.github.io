---
layout: post
title:  "Integrating Learning and Planning"
date:   2021-03-17 01:00:00 +0900
category: "Reinforcement Learning"
tags: YouTube
---

## 참고

- video
    - [[강화학습 8강] Integrating Learning and Planning](https://youtu.be/S216ZLuCdM0)
- slide
	- [Lecture 8: Integrating Learning and Planning](https://www.davidsilver.uk/wp-content/uploads/2020/03/dyna.pdf)



## Model-Based Reinforcement Learning
- 이전 강의 까지는 경험들로부터 직접적으로 policy를 학습했다. agent가 경험을 쌓고, 쌓인 경험을 바탕으로 직접적으로 policy를 업데이트했다. policy gradient, Q-learning 등 경험으로부터 학습했다.
    - policy, value function을 학습
- 이번 강의에서는 경험으로부터 모델을 직접적으로 학습해본다. 모델을 학습해서, planning 하는 것을 배운다.
- planning: 모델이 있을 때, value function과 policy를 구하는 과정을 말한다.
- 마지막엔 learning과 planning을 합치는 것을 배운다.



## Model-Based and Model-Free RL
- Model-Free RL
    - 모델이 없고, 경험들로부터 직접적으로 value function이나 policy를 학습
- Model-Based RL
    - 먼저 경험으로부터 모델을 배우고, 그 모델을 바탕으로 value function이나 policy를 계획



## Model-Free RL
- action을 하면 환경에서 state와 reward를 반환한다. 이를 바탕으로 또 action을 한다. 여기서 쌓인 경험들로부터 학습을 한다.



## Model-Based RL
- 실제 환경을 모방한 모델과 상호작용을 하면서, 학습을 한다. 모델은 실제 환경을 정확히 반영할 수록 좋겠지만, 실제 환경과 조금 다른 부분이 있을 수 있다.



## Model-Based RL
- policy가 action을 하면 경험이 생기고, 이 경험을 바탕으로 바로 policy와 value를 수정하는 것을 direct RL이라 한다. 경험으로부터 model을 먼저 만들고(learning) 만들어진 모델로부터 planning을 통해 value와 policy를 만드는 것을 Model-based RL의 개요도라고 볼 수 있다.
- MDP를 푸는 것을 여기서는 모델을 알기 때문에 planning이라고 표현한다.
- model learning은 MDP를 배우는 부분이고, fully known MDP가 아니지만, MDP를 learn하고 planning하여 MDP를 푼다.



## Advantages of Model-Based RL
- 그럼 왜 model-based RL을 할까?
- 장점
    - 실제 최적의 policy는 찾기 매우 어려운데, 모델 자체가 매우 간단한 경우가 있다.
        - ex. 체스
        - 체스에서 모델은 게임의 규칙을 의미한다. 어떤 state에서 어떤 action을 하면 어떤 칸으로 움직이는지, 말 잡고, 어떻게 움직이는지는 간단하기 때문에, supervised learning 같은 방법으로 쉽게 효과적으로 배울 수 있다.
    - 모델의 불확실성에 대해서도 이용할 수 있다. 모델을 만든다는 것은 환경이 어떻게 작동하는 지를 이해하는 것이다. 우리가 모델을 만들 때, 환경이 어떻게 작동하는지, 확실히 아는 부분도 있고, 잘 모르는 부분도 있을 수 있다. 만약 방문 횟수가 적어서 다음 state가 어떻게 될지 확률 분포가 확실하지 않아서, 높은 불확실성을 가지고 예측할 때도 있고, 반대로 방문 횟수가 많아서 다음 state가 어떻게 될지를 잘 알 수 있을 것이다. 이렇게 높은 정확도를 가지고 예측할 때와 낲은 정확도로 예측할 때, 이 둘이 학습에 영향을 미치는 것이 달라야 할 것이다. 이렇게 어느정도의 정확도에 따라 가중치를 가변적으로 적용해서 학습시킬 수 있다.
- 단점
    - 바로 배우는 것이 아니라, 먼저 모델을 배우고 그 모델로부터 value와 policy의 최적을 찾아나간다. 즉 2단계를 거쳐서 학습이 진행되기 떄문에 틀릴 수 있는 군데가 2군데나 생긴다.
        - model learning을 학습하는데, 모델이 잘못 학습되면 오류가 발생하여 최적으로부터 멀어지게 될 수 있다.
        - 그 모델로부터 value, policy를 planning할 때 또 오류가 있을 수 있다.



## What is a Model?
- 모델은 무엇일까?
    - 모델은 MDP의 표현형 S, A, P, R 이다.
        - S: state의 집합
        - A: action의 집합
        - P: transition probability, 어떤 state에서 어떤 action을 하면 다음 state가 무엇이 될지에 대한 확률분포
            - 같은 state에서 같은 action을 하더라도 확률에 따라 다른 state로 갈 수 있다.
        - R: reward
    - \eta로 parameterized해서 표현한다.
        - ex. neural net일 경우 neural net의 weight가 \eta가 된다.
    - 여기서 state space와 action space는 안다고 가정한다.
        - 모른다고도 할 수 있지만, 그렇게 되면 좀더 복잡해지므로 여기는 intro 강의이기 때문에 안다고 가정하고 진행한다.
    - 모델은 P와 R을 표현하는 것이다.
        - P: s에서 a를 할 때, 다음 state가 뭐가 될지에 대한 확률분포
        - S_{t+1}: 확률분포 P_\eta에 의해 뽑힌다.
        - R_{t+1}: 확률분포 R_\eta에 의해 뽑힌다.
    - S와 R을 서로 관계가 없이 독립적으로 구할 수 있다.



## Model learning
- 모델을 어떻게 학습할 것인가?
    - 그냥 experience들 S, A, R, S이 어떻게 됬는지, 모든 데이터를 쌓아둔다.
    - s, a ---> r은 regression problem
    - s, a ---> s은 density estimation problem(확률 측정 문제)
    - supervised learning 방법을 사용
        - input과 output들이 주어졌을 때, 그것을 바탕으로 input, output의 관계를 학습하는 것
        - r, s라는 답이 있기 때문에, supervised라고 한다.
    - 틀린 정도 error를 정의한다. 예를 들어 s1, a1를 넣었을 때, output r'이라고 할 경우, r'은 r2가 되어야 할 텐데, 이 차이를 계산한다. ex. mean-squared error
        - ex. 확률 분포의 차이를 표현한 KL divergence
    - gradient descent 방법론 등을 사용해서, 실제 데이터에 대해서 이 차이를 줄이는 방향으로 파라미터 \eta를 업데이트 한다.



## Examples of Models
- Table Lookup Model
    - reward, state transition probability 테이블을 채워나가는 방식
- Gaussian Model
- Neural net model



## Table Lookup Model
- MDP의 P와 R을 구해야 한다.
- 각 state-action pair를 방문할 때마다 어떤 counter를 두어서 계속 더한 뒤, 평균을 낸다.
    - reward: 어떤 state에서 action a를 했을 때 받았던 reward의 평균을 계산
    - 확률분포: 어떤 state에서 action a를 했을 때 어느 state에 도착했는지의 평균을 계산
    - 한번 방문할 때마다 계속 업데이트 하면서 테이블에 저장하는 식으로 모델을 학습
- 또는 S, A, R, S'에 대한 tuple들을 기록해서 s에서 a를 선택하면, 미리 기록되어 있는 데이터들 중에서 s, a로 시작하는 tuple들 중에서 uniform random하게 샘플링을 하는 방법도 있다. 이렇게 하면 위에서 하는 것과 거의 대등하다.
- 단점: 내가 경험하지 못한 state transition은 알 수 없다.
- scale up하기 위해 neural net을 사용하게 되면, 우리가 가보지 못한 transition들도 interpolation하면서, 그 사이에 어떻게 될지 미리 가본 점들을 바탕으로 예측할 수 있다. 물론 정확도의 문제는 남아있게 될 테지만..



## AB Example
- state A, B 2개가 있고 왼쪽과 같은 어떤 경험들이 있을 때, 이 경험을 바탕으로 모델이 어떻게 될까?
    - 오른쪽을 보면 A에서 항상 B로 가게 된다. 왜냐하면 그에 대한 샘플이 1개밖에 없기 때문이다.
    - B는 8번중 6번은 1로, 2번은 0으로 가기 떄문에 각각 75%, 25% 확률로 terminal state에 도착한다.
    - 이렇게 table lookup model을 experience로부터 만들어 내었다.



## Planning with a Model
- 모델이 만들어졌으니, 이제 모델을 이용해서 풀어야 한다.(planning)
    - MDP S, A, P_\eta, R_\eta
- planning algorithm 들 중 마음에 드는 것을 선택해서 사용한다.
    - value iteration, policy iteration
        - Dynamic programming MDP를 알 때 사용할 수 있다.
        - 지금은 MDP를 모델링한 가상의 MDP를 알기 때문에 이 방법론을 그대로 똑같이 쓰면 된다.
    - tree search



## Sample-Based Planning
- planning 방법론들 중에 dynamic programming 방법론을 쓸 수 있고, 또는 sample-based planning을 사용할 수 있다.
    - dynamic programming 방법론: MDP를 안다고 가정하고 사용, full-width backup
- 모델을 만들어서 MDP를 알고 있는데, MDP를 모른다 가정하고 model-free 방법론을 쓰는 것이다. MDP를 모델을 통해서 알고 있는데, 그것으로부터 샘플들을 만들어 낼 수 있다. 그러면 이 샘플을 만들어내는데에만 모델을 사용하고, 이제 모델은 모른다고 가정하고 그 샘플로부터 model-free 방법론을 적용할 수도 있다.
    - 샘플을 P_\eta, R_\eta로 부터 만들어내서, 이 샘플들이 실제 experience인 것 처럼 이용해서 model-free 방법론으로 update한다.
        - Monte-Carlo control
        - Sarsa
        - Q-learning
- 이것은 굉장히 간단하지만, 강력한 planning 방법론이다.
    - 즉 dynamic programming 방법론을 사용할 수 있고, model-free 방법론도 사용할 수 있다.
    - 왜냐하면, dynamic programming 방법론은 naive full-width backup이다. 즉 순진하게 모든 갈수 있는 가지들에 대해서 모든 값들을 업데이트에 이용한다. 그런데, 이 sample-based planning은 자주 일어나는 사건들에 집중할 수 있다. 왜냐하면 더 자주 일어날 확률이 높을 수록 샘플이 더 많이 뽑힐 것이기 때문이다. 그렇게 자주 일어나는 사건들에 더 집중해서 학습을 할 수 있다. 이를 이용해서 curse of dimensionality를 해결할 수 있다.
        - curse of dimensionality: 차원의 저주
            - 예를 들어 full-width backup을 할 경우, 무언가 문제가 한 차원씩 복잡해질수록 exponential하게 계산복잡도가 늘어난다.
            - 바둑을 두는데 현재 수가 70수인데, 보통 200수 정도이면 게임이 끝난다고 하자. 그럼 200수까지 더 가야 할 텐데, 한수 한수 갈 때마다 가능한 경우의 수가 처음에는 200이 가능하고 다음은 198개 가능하고, 이렇게 exponential하게 늘어난다. 이 모든 경우의 수를 다 시도해볼 수 없을 것이다. 그러면 정해진 10만판 게임을 하고 이를 바탕으로 평균을 내면, 해당 수에서 이길 수 있는 확률을 알 수 있다.
                - 즉 샘플기반 방법론들은 모두 차원의 저주를 해결하는 좋은 방법론들이 된다. navie 또는 full-width backup은 높은 차원의 모든 state에 대해 brute-force로 모든 경우를 생각해야 한다. 자주가는 state나 자주가지 않는 state나 대등하게 처리될 것이다. 반면 만번 샘플링을 한다면 그중 자주 가는 state가 8000번, 자주 가지 않는 state들은 골고루 섞여서 2000번 샘플링 되었다면, 자동으로 자주 가는 것에 비례해서 데이터를 얻게 될 것이다. 중요한 데이터가 더 자주 얻게 될 테니까, 자주 가는 쪽에 대해서 더 정확하게 업데이트가 될 것이고, 차원의 저주를 해결할 수 있다.



## Back to the AB Example
- table lookup 모델을 실제 경험으로부터 만들었다.
- model-free RL을 sampled experience에 적용한다.
- 우리가 모델을 갖게 되는 순간, 우리는 무한한 데이터를 얻게 되는 것과 마찬가지가 된다.
    - 데이터는 8개밖에 없는데, 이 8개로 모델을 만드는 순간, 우리는 데이터를 계속 생성해낼 수 있고, 이 생성된 데이터로 model-free 방법론은 적용할 수 있다.
        - ex. Monte-Carlo 방법론: A에 방문한 것을 평균내는 방법론
            - A를 방문했을 때, 최종적으로는 reward를 1 받으므로 V(A) = 1



## Planning with a Inaccurate Model
- 우리는 어디까지나 모델을 생성해서 학습을 진행하는데, 모델이 정확할 일은 거의 없을 것이다. 조금씩 다를 것이다. 모델이 완전히 같지 않으면 model-based RL의 performance는 이 가짜 MDP의 optimal policy로 국한이 될 것이다.
- model-based RL은 model이 좋을 수록 그만큼 학습이 잘 될것이다. model이 부정확하다면 planning이 suboptimal policy로 계산이 될 것이다.
- 그럼 모델이 틀릴때는 어떻게 해야 할까?
    1. 그냥 모델을 사용하지 않고, model-free RL을 사용하면 될 것이다.
    2. 모델을 학습하는데 몇 % 정도 자신있는지를 같이 학습한다. 예를 들면 구간을 output으로 출력한다면, 구간이 넓으면 그만큼 자신이 없다는 뜻일 것이다. reward값이 실제는 30인데, 0 ~ 100사이의 값이라고 말하는 것과 29.5 ~ 30.5라고 말하는 것처럼 uncertainty를 이용한다.
        - ex. bayesian model-based RL: bayesian 방법론을 이용해서 prior 분포를 두고, 모델이 경험할 때마다 그것을 업데이트하면서 분포가 점점 정교해지도록 하여 불확실성(variance)과 같이 이용해서 RL을 할 수 있다.



## Real and Simulated Experience
- 경험은 2가지 출처가 있다.
    1. real experience: true MDP로부터 뽑힌 샘플들
    2. simulated experience: 모델로부터 나온 샘플들
        - \eta로 parameterize된 approximate MDP



## Integrating Learning and Planning
- Model-Free RL
- Model-Based RL
- Dyna
    - 실제 경험으로부터 모델을 배우고, 이 2가지 source의 경험들을 바탕으로 learn과 plan을 한번에 한다.
    - learn: 실제 경험으로부터 배우는 것
    - plan: 모델을 이용해서 배우는 것



## Dyna Architecture
- Model-based RL에서는 direct RL 화살표가 없었다. experience는 순전히 모델을 만들기 위해 사용되고, value/policy를 직접적으로 업데이트(direct RL)하기 위해 사용되지 않았다.
- Dyna는 direct RL도 있고, 모델을 학습한 뒤의 planning도 존재한다.



## Dyna-Q Algorithm
- table lookup 방법
    - 당연히 function approximation도 사용할 수 있다.
- pseudo code
    1. 모든 state와 action에 대해서 Q와 모델을 random값으로 초기화
    2. s에서 e-greedy로 action을 하나 뽑고, 그 action을 한뒤, 다음 state s'으로 가고 reward를 받는다.
    3. 그리고 위에서 받은 reward와 다음 state s'으로 Q를 업데이트한다.
        - 일반적인 Q-learning 방법론: 원래 값이 있고, \alpha만큼 업데이트를 한다.
            - TD target이 있고, TD error을 계산해서, TD error만큼 업데이트 한다.
            - Q가 추측이고, r + \gamma \maxQ가 한 step 더 가서의 추측이다. 이 한 step 더 가서의 추측이 더 정확하기 때문에, 그 방향으로 가도록 Q를 업데이트 한다.
            - model-free 방법론
    4. Model을 배운다. s에서 a를 받으면, r을 하고 s'이 된다는 것을 모델 자리에 넣어준다. 여기서는 쉽게 학습하기 위해서 deterministic environment라고 가정한다.
    5. n번 planning을 이용해서 Q를 업데이트한다. 가봤던 state 중 한 개를 뽑고, 그 state에서 했던 action을 한 개 뽑는다. 그 s, a에 대해서 똑같이 Q를 업데이트 한다. 이 때, s, a를 이용해서 다음 state와 reward가 무엇이 될지를 모델을 통해서 뽑고, 이 s, a, r, s'이라는 하나의 가상의 tuple을 이용해서 업데이트 한다.
        - 이 때, 한번 action은 실제 환경과 교류를 해서 Q와 모델을 업데이트 한 뒤에, 이 모델을 이용해서 n번, 예를 들어 50번 정도 Q를 업데이트 한다.
- (a) - (e): real experience로 업데이트
- (f): simulated experience로 업데이트
- 실제 환경과의 교류가 너무 쉽게 이루어진다면, 실제 데이터를 통해서 학습하면 되는데, 실제 환경과의 교류가 너무 어렵고, 비용이 많이 들 경우, 한번 데이터를 얻고 그로부터 모델을 만들어서 손 쉽게 업데이트하는 방식이다.



## Dyna-Q on a simple Maze
- planning을 쓰지 않고 S에서 G로 갈 경우, 한 에피소드를 끝내기 위해 너무 많은 step이 필요하다.
- planning을 사용할 경우, 몇번 step이 지나지 않아서 Goal에 도착하여 에피소드가 끝난다.
- 게다가 50 planning steps을 사용했을 경우, 실제 경험으로 쌓은 데이터를 60배나 효율적으로 사용했다. 이미 쌓인 실제 경험이 있으면, 그 경험만으로도 더 배울 수 있다. 즉 아주 적은 경험을 짜내서(squeeze) 모델을 만들어서 상상속에서 학습을 진행했다.
- 0과 5의 차이가 극심하다. 5와 50은 이미 최적에 도달해 있다.
- 그래프 상 x축이 2부터 표현이 되어있는데, 만약 1일 경우에는 어땠을까?
    - 이 문제의 경우에는 direct RL이나 Dyna나 차이가 없다.
    - 그런데, 만약 벽에 부딪히면 - reward를 가지면서 다시 S로 되돌아가게 된다면, 처음 Goal에 도착할 때까지 벽에 부딪히면 안된다는 것을 배우게 되기 때문에 planning을 통해 더 학습하게 되서 처음 도착할 때 까지도 더 잘 갈 수 있게 된다.



## Dyna-Q with an Inaccurate Model
- Dyna를 사용할 때, 모델이 틀릴 경우
- 딱 1000번째 step에서, environment가 S에서 G로 가는 경로가 돌아가도록, 더 어렵게 바뀌었다.
- Dyna-Q+: Q+가 더 빨리 배우는 이유는, action을 선택할 때 exploraition을 더 하도록 포함되었다.



## Dyna-Q with an Inaccurate Model(2)
- environment가 S에서 G로 가는 경로가 빠르게 가도록, 더 쉽게 바뀌었다.



## Outline - Simulation-Based Search
- planning을 좀 더 효과적으로 하는 방법에 대해 배운다.



## Forward Search
- 전체 state의 공간은 너무 큰 상황에서 planning을 할때, 사실 다른 state는 관심이 없다. 현재 있는 상황에서 일어날 일들에만 관심이 있지, 전혀 멀리 있는 다른 상황은 관심 없다.
- ex. 바둑을 170수까지 진행된 상황에서, 맨 처음 30수 쯤에는 어디에 두어야 하는지는 현재 170수인 상황에서 아무런 관련이 없다.
- 즉 현재 상황이 특별히 더 중요하다. 따라서 지금으로부터 미래만 보겠다는 것이 forward search의 철학이다.
- lookahead(미래를 보는 것)를 통해서 best action을 선택한다.
    - 우선 현재 state s_t가 root가 되는 search tree를 만들고, planning이 있으므로 모델을 아는 상황이므로, MDP의 모델을 이용해서 lookahead search tree를 만든다.
    - 즉, MDP 전체를 풀 필요가 없다. 지금부터 이어지는 sub-MDP만 풀면 된다.
- 흰색 원: state
- 검은 원: action
    - 위 그림은 모든 state에서 2가지 action밖에 못하는 상황인 것 같다.



## Simulation-Based Search
- 미래의 상황들을 sample based planning을 사용해서 Forward search를 한다.
- 현재 s_t에서 시작하는 여러 에피소드들을 생성하고, 생성된 에피소드들에 대해서 model-free RL을 적용한다.
- planning: 모델을 이용해서 더 나은 policy를 만드는 일
- forward search: 현재 상황으로부터만 더 보는 것
- forward search + model-free RL: simulation-based search
    - 결국 simulation-based search는 forward search이다.



## Simulation-Based Search(2)
- 우선 모델로부터 만들어진 경험들을 시뮬레이션을 통해서 만들어낼 때, 시작지점이 현재 s_t로부터 시작하는 경험들만 만들어내기 때문에 forward search에 해당한다.
- 여기서 만들어낸 경험들을 바탕으로, model-free 방법론을 적용한다.
    - Monte-carlo, Sarsa를 쓸 수 있다.
    - MC, TD 방법론을 쓸 수 있다.
    - 이 때, Monte-Carlo 방법론을 쓴 것을 Monte-Carlo search라고 한다.



## Simple Monte-Carlo Search
- simulation 기반의 search + Monte-Carlo control
- 모델이 주어져있고, simulation하는 policy \pi가 주어졌을 때, 모든 각각의 action에 대해서, 어떤 state s에 대해 할 수 있는 action이 10개가 있다고 가정하자. 그럼 10개 각각에 대해서 K만큼 에피소드를 만들어 낸다.(ex. K = 100) 그러면 s_t에서 a를 한 것은 실제 한 행동인데, 그 이후 100개의 에피소드들은 모두 가상의 에피소드가 된다(10개의 행동 x 100개의 에피소드 = 1000개의 에피소드). 그리고 monte-carlo 방식으로 모든 에피소드들에 대해서 평균을 낸다. 즉 각 action들에 대해서 return을 평균낸 것이 q_\pi가 된다. 왜냐하면 policy \pi로 simulation을 했기 떄문이다. 그렇게 10개 action들 중에서 평균이 제일 높았던 action을 선택한다.



## Monte-Carlo Tree Search(Evaluation)
- MCTS: RL의 planning에 속하고, 그중 forward search에 속하면서, simulation-based search에 속하고, 그중 monte-carlo search에 속한다.
- 똑같이 모델이 주어졌을 때, 각 action들에 대해서 K개의 에피소드들을 만드는 것이 아니라, 그냥 현재 policy를 이용해서 K개의 에피소드를 만들어 낸다. 그러면 policy는 어떤 action을 더 많이 할 것이고, 그러면 그런 action을 한 것들이 더 많이 뽑힐 것이다.
- 위에서 에피소드를 만들 때, root노드만 만들어지는 것이 아니라 action을 선택했을 때의 방문하는 다른 state들도 만들어 지는데, MCTS에서 search tree를 만들 때, 이들의 정보도 함께 같이 가지고 있다.
- 그 각각의 Q(s, a)는 똑같이 방문한 state의 mean return으로 평가를 하고, search가 끝나면 그 중 action이 가장 높은 것을 선택하면 된다.



## Monte-Carlo Tree Search(Evaluation)
- monte-carlo search에서는 simulation policy \pi가 고정되어 있었다.
    - 그런데 이 simulation policy도 같이 똑똑해지면 좋을 것이다.
    - 우리는 지금 planning 방법론으로 경험들을 모델을 이용해서 쌓고 있는데, 경험들을 이용해서 simulation policy도 똑똑해질 수 있다.
- MCTS에서는 simulation policy도 개선된다.
- 각각의 시뮬레이션은 2단계로 나뉘어진다.
    - Tree policy(imporves): Q를 maximize하는 행동을 선택한다.
    - Default policy(fixed): random하게 또는 고정된 policy를 통해 행동을 선택한다.
        - tree를 보면, 내가 만든 영역이 있고, 만들어지지 않은 바깥 영역이 있을 텐데, 내가 만든 영역은 tree를 알기 때문에 Q를 계산할 수 있는데, 만들어지지 않은 바깥 영역의 경우 Q를 모르기 때문에 이 경우 default policy를 수행한다.
        - 알파고의 경우 default policy는 random policy가 아닌 rollout policy를 사용한다.
- 각 시뮬레이션마다 반복한다.
    - evaluate: Monte-carlo 방법론으로 evaluate한다.
    - imporve: e-greedy(Q)를 사용해서 tree policy를 향상시킨다.
    - Q를 평가하면서 점점 정확해지고, Q중에 가장 큰 행동을 선택하므로 policy도 점점 좋아진다. 즉 policy iteration과 유사하게 동작한다.
- simulated experience에 monte-carlo가 적용된 것으로 볼 수 있다.
- optimal search tree에 수렴한다.



## Rules of Go
- 2500년된 가장 어려운 보드게임
- 이 강의는 알파고가 나오기 이전이니까, 당시 grand challenge였다.
- 전통적인 game tree search는 실패했다.
- 바둑의 규칙은 19x19에서 진행하고, 규칙은 간단하지만 전략은 복잡하다.
- 검은 색과 흰 색의 영역 싸움이며, 더 많은 영역을 먹은 돌이 이기는 게임이다.



## Monte-Carlo Evaluation in Go
- MCTS를 이용해서, 어떤 position이 있을 때, 그 position이 얼마나 좋은지 평가를 한다.
- 먼저 reward를 정의할 때, terminal state가 아니면 0, 끝날 때, 검정이 이기면 1, 흰색이 이기면 0으로 했다.
- policy는 \pi_b와 \pi_w고 각각 흑/백의 policy가 있다. 이것을 이용해서 양쪽 player를 번갈아가면서 진행한다. 이때 어떤 position s가 얼마나 좋은지를 나타내는 value function은 s에서 black이 이길 확률로 정했다.
- optimal value function은 흰색 w는 V를 최대한 낮추도록 하고, 검정 b는 V를 최대한 크게 하도록 한다. 즉 서로 싸우는 형태로 진행한다.(min-max, mini-max 게임)
- 현재 state에서 policy를 이용해서 쭉 번갈아 두면서 4개의 에피소드를 관찰했을 때, 결과가 1, 1, 0, 0이 나왔다면, 이때 현재 state에서 검은 돌이 이길 확률이 0.5라고 평가한다.
- policy가 좋아야 더 좋지 않을까?
    - 당연히 잘 둘수록 좋고, MCTS의 놀라운 점은 심지어 random policy를 써도 꽤 쓸만하다.



## Applying Monte-Carlo Tree Search(1)
- tree policy: 윗 부분에만 tree가 생성된 상태
    - tree가 생성되어 있을 때, 사용하는 정책
- default policy: 아래 부분은 더 보는 부분
    - tree가 생성되지 않았을 때, 사용하는 정책
- 처음 에피소드가 끝났을 때, 흑이 이겼으므로 current state는 이긴 점수(1)/에피소드 수(1)



## Applying Monte-Carlo Tree Search(2)
- 2번째 에피소드에서 백이 이겼으므로, current state는 이긴 점수(1)/에피소드 수(2)
- 그 다음 노드는 트리에 메달리면서 흰색 별에는 이긴 점수(0)/에피소드 수(1)가 기록된다.



## Applying Monte-Carlo Tree Search(3)
- 3번째 다른 에피소드에서 다른 시도를 하면서 current state에는 다른 노드가 추가된다.
- 점점 트리가 확장이 된다.



## Applying Monte-Carlo Tree Search(4)
- 같은 방식으로 또 다른 노드가 추가된다.



## Applying Monte-Carlo Tree Search(5)
- 현재 state는 계속 숫자가 늘어날 것이고, 그 아래의 노드들의 숫자도 늘어날 것이다.
- tree search와는 다르게, tree가 점점 더 정확하게 된다.
- tree policy: 생성된 tree안에서 행동을 정할 때는, Q를 이용해서 행동을 정한다.
    - 2/3가 0/1보다 크기 때문에, 왼쪽 노드 방향의 행동을 선택한다.
    - exploration이 포함된 e-greedy policy를 통해 tree policy는 점점 향상된다.
- default policy: random이나 다른 고정된 policy를 사용해서 에피소드가 끝날때까지 행동을 선택한다.



## Advantages of MC Tree Search
- 첫 state는 여러번 반복적으로 계산되기 때문에, 처음 선택되는 state가 매우 정확하다.
- state를 dynamically하게 평가한다.
    - DP와는 다르다. dynamically하다는 것은 매번 그 state에 방문할 때마다 계산을 한다는 것이다. 예를 들어 1 ~ 10번 state가 있을 때, 3번 state에서 1번 state를 평가하는 것과 8번 state에서 1번 state를 평가하는 것이 달라질 수 있다. 왜냐하면 모든 state를 한번에 같이 평가하지 않기 때문이다. 매번 모든 state를 고려해서 평가하게 되면 동일한 값을 가지겠지만, 이 방법은 현재 처해진 상태에 좀더 집중적으로 더 많이 평가하고 하기 때문에, 현재 처해진 상태를 기준으로 평가를 진행한다.
    - Dynamic programming은 full-width backup을 하기 때문에 dynamically하게 평가하는 것과 차이가 있다.
- 샘플링을 이용하기 때문에 차원의 저주를 깨뜨린다.
- 모델이 어떻게 생겼는지 몰라도 상관 없이, 단지 query만 날릴수 있으면 된다. 즉 샘플링을 위해 어떤 state에서 어떤 action을 하면 다음 state와 reward가 무엇인지를 알려주기만 하면된다.
- 계산적으로 효과적이고, 병렬성도 좋다.



## Example: MC Tree Search in Computer Go
- x축: 시간
- y축: 바둑을 두는 실력, 위로 갈 수록 더 잘하는 실력
- 상위에 있는 프로그램들이 MCTS 기반 방법론들이고, 아래에 있는 프로그램들은 MCTS를 안쓴 프로그램들이다.
    - MCTS를 사용한 프로그램들이 더 좋았다.



## Temporal Difference Search
-  MC냐 TD냐는 중요하지 않다. 여기서 중요한 것은 현재 상황에 국한지어서 forward search로 더 특화된 policy를 만드는 것이 중요하고, 그것이 힘이 있는 거지, MC를 쓰던, TD를 쓰던 상관 없다.
- TD search도 마찬가지로, simulation-based search인데 MC자리에 TD를 쓰는 것이다.
- MCTS는 현재로부터의 sub-MDP에 MC control을 적용한 것인데, TD search는 sub-MDP에 Sarsa를 적용한다.



## MC vs. TD search
- model-free RL에서 TD learning이 variance를 줄이고 bias를 올린다. 보통 MC보다 더 효과적이고, TD(\lambda)는 더 많이 효과적일 수 있다.
- simulation-based search도 마찬가지로 똑같이 적용된다.
    - 왜냐하면 planning이 만들어준 샘플에 적용하는 것 뿐이기 때문이다.



## TD Search
- 현재 state로부터 시작되는 에피소드들을 뽑아내고, 뽑아낸 에피소드들의 action-value function을 평가하는데, 매 step마다 Sarsa를 활용해서 Q를 업데이트한다.
    - TD target에서 Q를 뺀 것에 learning rate를 곱해서 더해준다.
- 그리고 action은 e-greedy를 사용해서 선택한다.
- 당연히 Q에 function approximation을 쓸 수 있다.



## Dyna-2
- Dyna의 다른 버전
- feature weight를 2개 준비한다.
    1. Long-term memory: real experience로 학습
    2. Short-term (working) memory: simulated experience로 학습



## Results of TD Search in Go
- TD Learning + TD Search = Dyna-2
- TD Learning은 search가 아예 쓰이지 않으므로, planning이 아예 들어가지 않은 것이다.
    - real experience만 사용한 것이다. 즉 실제 경험만 사용하면 이만큼 안좋으므로, 그만큼 search가 중요하다.
