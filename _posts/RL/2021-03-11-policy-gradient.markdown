---
layout: post
title:  "Policy Gradient"
date:   2021-03-11 01:00:00 +0900
category: "Reinforcement Learning"
tags: YouTube
---

## 참고

- video
    - [[강화학습 7강] Policy Gradient](https://youtu.be/2YFBordM1fA)
- slide
	- [Lecture 7: Policy Gradient](https://www.davidsilver.uk/wp-content/uploads/2020/03/pg.pdf)



## Policy-Based Reinforcement Learning
- 지난 시간에 gradient descent라는 방법으로 value function의 \theta를 업데이트하는 방법을 배웠다.
    - 신이 답을 알려줄 수 없으므로, Monte-Carlo나 TD를 값으로 두고, v와 Q를 업데이트 하였다.
    - 이때는 value function만 있고, policy는 없었다.
    - policy는 value function을 이용해서 만든 policy였다.
        - Q에서의 epsilon greedy policy
- 여기서는 직접적으로 policy를 parameterize한다.
    - policy를 파라미터를 이용해서 어떤 함수로 표현
    - ex. neural net
- model-free: MDP에 대한 완전한 정보가 없는 상황에서 환경에 agent가 던져져서 어떤 행동을 하고, policy를 따라서 경험들로부터 바로 배우는 방법론



## Value-Based and Policy-Based RL
- Value Based Method
    - Value function에 기반한 방법론
    - value function을 어떤 함수로 표현했고, 함수의 파라미터들을 업데이트해서 그 함수가 정확한 value를 return하도록 학습
    - implicit policy: value function을 이용해서 policy를 만듬
- Policy Based
    - 아예 value function을 학습하지 않고, 바로 policy를 배움
    - policy gradient
- Actor-Critic
    - value-based와 policy-based를 같이 학습하는 것
    - actor: 움직이는 것, policy
    - critic: 평가하는 것, value function
    - policy 함수와 value 함수를 둘다 학습해서 agent가 학습하게 하는 방법론



## Advantages of Policy-Based RL
- 장점
    - 수렴하는 성질이 더 좋다. value based method는 function approximatior을 사용하기 떄문에, variance가 크면 수렴을 잘 안하면 학습이 어렵다.
    - 할 수 있는 action(dimension)의 가짓수가 많을 때, 심지어 continuous action space를 생각해보면, 이런 행동 가짓수는 value based로 학습하기 어렵다.
        - ex. Q(S, a)를 학습했다고 가정할 때, greedy하게 움직이기 위해, 내가 가능한 action중 제일 큰 Q값을 가지는 action a를 뽑아야 할텐데, 그 자체가 또 하나의 최적화 문제이다.
            - 어떤 임의의 함수 Q가 있고, input이 0~1사이의 모든 실수에 대해서 가능할 때, 이 때 Q를 제일 maximize하는 input을 찾는 문제 자체도 또 하나의 고유의 문제이다.(optimization 문제)
            - discrete action일 경우, 모든 가짓수를 시도하면 된다.
            - continuous action일 경우, 0~1사이의 모든 실수를 모두 넣어볼 수 없으므로 최적화 문제를 풀어야 한다.
        - discrete action space: 1, 2, 3 행동
        - continuous action space: 0~1 사이의 실수를 행동으로 정의
            - action이 무한개가 가능
            - 굉장히 자주 있는 문제이다.
            - ex. 로봇팔을 제어하는 문제일 때, 1도, 17도, 17.3도 회전할 수 있을 것이다.
        - 그냥 policy을 학습하면, policy의 input을 넣으면 바로 action이 나올 수 있다.
    - value based 방법론은 stochastic policy가 없었고, 모두 deterministic policy만 학습했다.
        - deterministic policy: 어떤 state에서 어떤 action을 하는 것이 결정론적이다.
            - 지금까지 항상 greedy하게 업데이트해왔다.
        - stochastic policy: 어떤 state에서 어떤 action을 하는 것이 확률론적이다.
            - 1번 action: 50%, 2번 action: 50%
            - 같은 state에서 매번 하는 행동이 다르다.
- 단점
    - global optimum보다 local optimum에 빠지기 쉽다.
    - variance가 크고, 좀 비효율적인 면이 있다.
        - value based는 매우 aggressive(공격적인)한 방법론이다. Q 한번 업데이트하고 나면, 내가 할수 있는 action 중에 max를 취하므로, policy가 급격하게 바뀐다.
        - policy gradient: 훨씬 stable하다. 왜냐하면, gradient만큼 조금 업데이트하므로 1번 action이 좋으면 확률이 60%가 62%가 되므로, 좀더 smooth하고, stable하게 바뀌어서, stable 하지만 조금 효율성이 떨어진다.
