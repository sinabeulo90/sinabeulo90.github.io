---
layout: post
title:  "Policy Gradient"
date:   2021-03-14 01:00:00 +0900
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
            - 어떤 임의의 함수 Q가 있고, input이 0~1사이의 모든 실수에 대해서 가능할 때, 이 때 Q를 제일 maximize하는 input을 찾는 문제 자체도 또 하나의 고유의 문제이다.(optimization 문제)
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



## Example: Rock-Paper-Scissors
- stochastic policy가 필요한 상황에 대한 예시
- 반복해서 계속하는 가위.바위.보
    - 만약 deterministic policy이기 때문에 내가 항상 주먹만 낸다면, 몇 번만 진해하면 상대가 나를 이기는 policy를 금방 찾아갈 것이다.
    - 먄약 uniform random policy가 있다면(가위.바위.보를 1/3 확률로 내는 policy), 상대방이 나를 완전히 이길 수 없을 것이다.
        - 즉 각각 1/3 확률로 내는 것이 최적이다. 게임 이론 표현으로 내쉬균형이라고 한다. 내가 이 전략을 쓰고, 상대도 같은 전략을 썼을 때, 둘다 바꿀 요인이 없어서 평형을 이루는 평형
        - 이런 내쉬평형에 도달해서 최적이 될 수 있도록 학습하기 위해 policy-based 방법론이 필요하다.



## Example: Aliased Gridworld(1)
- stochastic policy가 필요한 상황에 대한 예시
- 해골에 빠지면 죽고, 금에 도달하면 성공하는 게임이다. MDP의 모든 정보를 알고 있는 상황을 partially observable한 상황으로 바꿔보자.
    - ex. 북쪽에 벽이 있는가, 남쪽에 벽이 있는가, 하는 방식으로 state를 구분하는 feature를 만들어보자. 그럼 이 feature가 완전하지 않아서 fully known MDP인 상황이 깨지게 된다. 즉, 회색 칸의 경우, 이 feature로는 구분이 되지 않는 칸이 된다.
    - feature는 얼마든지 완전하지 않을 수 있다. 또는 이 feature가 완전하지 않은지도 모르는 복잡한 상황이 있기도 한다.
    - deterministic policy인 경우: 회색 칸의 2개 state는 함수 입장에서 같은 state이므로, 같은 output을 내놓게 된다. 만약 회색 칸에서 왼쪽으로 가는 policy라고 한다면, 나머지 한쪽 회색 칸에서도 왼쪽으로 가게 될 것이다. 왼쪽 상단에서 agent가 출발하면 오른쪽으로 가게되고, 다음 회색칸에서는 왼쪽으로 가게 되므로, 절대로 금에 도달하지 못할 것이다.
    - 따라서 어떨 때는 왼쪽으로 보내주어야하고, 어떨 때는 오른쪽으로 보내주어야하는데, feature가 불완전해서 2개 회색 칸이 다른 칸이라는 것을 모른다. 따라서 저 회색칸이 50%는 왼쪽, 50%는 오른쪽으로 가야만 최적으로 항상 금에 도달할 수 있다.



## Example: Aliased Gridworld(3)
- value based RL: deterministic policy를 학습하기 떄문에, 평생 금에 도달하지 못할 것이다.
- 양쪽으로 가는 것이 1/2확률인 stochastic policy가 있다면, 몇 번 step만 밟으면 금에 도착할 것이다.
- 즉 state의 완전한 표현이 불가능한 상황(partially observable한 상황)에서 사용할 수 있다.
- 3장? 4장?에서 항상 최적의 deterministic한 policy가 존재한다는 정리를 통해서, deterministic policy를 안전하게 믿고 학습을 진행해왔는데, 위의 예시는 반대되는 것이 아닐까?
    - deterministic policy의 경우 Markov property(fully observable MDP)가 성립하면 최적의 deterministic policy가 성립한다.
    - 하지만, 위 예시는 partially observable한 솽황이고, 다른 state인데 구분되지 않는 상황이므로 성립된다.



## Policy Objective Functions
- policy 학습할 때, policy는 \pi 라는 function approximation을 사용하는 것이다.
    - 어떤 파라미터 \theta에 대해서 \pi는 action a를 할 확률을 뱉어주는 함수이다.
- 그렇다면 어떤 policy가 좋은 \pi인지 정의되어야 한다.
    - objective functions: maximize 하고자 하는 목적함수
        - 신이 value function을 알려준다면, 그것이 곧 reward의 기댓값이므로, value function이 높으면 그 policy가 좋은 policy인 것 처럼, 이 policy를 따랐을 때, 총 reward를 더 많이 받으면 좋은 policy가 된다.
    - 3가지 방법
        1. episodic environments: 한판, 한판 끝나는 환경
            - start value를 사용할 수 있다.
                - start value: 처음 시작 state가 항상 정해져 있다는 의미
                - ex. 철권에서 항상 같은 state, 어느정도 떨어져서 서로 마주보는 state에서 시작한다. state에서 시작했을 때, 이 policy \pi로 게임을 끝까지 했을 때, 얼마의 reward를 받을 것인가가 목적이 된다. 그리고 이 값이 곧 value이기 때문에 \pi를 따랐을 때의 value의 기댓값으로 볼 수 있다.
            - state s는 하나의 값이어도 되고, 고정된 분포여도 된다. 예를 들어 50% 확률로 1m 떨어져서 시작하고, 나머지 50% 확률로 2m 떨어져서 시작해도 된다. 즉 고정된 start state의 분포가 있을 때, 그런 상황에 대해서 value를 maximize 해본다.
        2. continuing environments
            - d^\pi_\theta(s): stationary distribution
                - Markov chain에서 policy \pi를 따라 계속 움직이다보면, 각 state별로 머무는 확률들을 구할 수 있는데, 각 state별로 어느정도 수렴하고 snapshot을 보면 agent가 몇 % 확률로 어디에 있는지를 알 수 있다. 그 확률 분포가 d^\pi(s)이다.
            - 각 state에 있을 확률 x 해당 state의 value 들의 총합을 목적함수로 삼는다.
            - average value
            - snapshot을 보면 각 state별로 몇 %의 확률로 있는지를 알기 때문에, 모든 state들에 대해서 sum을 하는 것
        3. average reward per time-step
            - stationary distribution d^\pi(s)에 대해서 각 state에서 policy의 action을 한번 하고 그 떄 얻는 reward를 \pi의 확률 가중치를 곱해서 모두 더한 뒤, stationary distribution 가중치를 곱해서 더한다.
            - 한 step에 대해서만 보는 것이다. snapshot을 보면 각 state별로 몇 %의 확률로 있는지를 알기 때문에, 모든 state들에 대해서 sum을 하고, 각 state에서 action을 할 수 있는 확률들이 있고, 그 action을 했을 때 얻는 reward를 더해서 목적함수로 사용한다.
    - 위의 3가지 서로 다른 목적에 대해서 똑같은 방법이 작동한다. 어느 한 방법론이 있으면, 동시에 어느 한개 목적함수를 최적화하는 것이 아니라, 3개를 동시에 최적화시킨다.



## Policy Optimisation
- optimisation problem: 어떤 임의의 함수가 있을 때, 주어진 정의에 관해서 값을 최대화하는 input x를 찾는 문제
    - 최적화 문제, 특정한 어떤 도메인은 가리키는 용어
- 목적함수 J(\theta)를 maximise하는 input \theta를 찾는 것이 목적
    - \theta가 policy를 정해준다. 왜냐하면, policy가 theta로 parameterize되어 있기 떄문이다.(\theta로 표현된 함수)
    - \theta가 바뀌면 policy가 바뀌고, policy가 바뀌면 게임하는 동안 얻는 reward가 바퀼 테고, 그렇게 되면 J가 바뀌게 된다. 그러므로 우리는 \theta를 조정해서 J를 maximize하고 싶은 것이고, 이것이 최적화 문제를 푸는 것이다.
- 최적화 문제에는 다양한 방법론이 있다.
    - gradient를 쓰지 않는 방법
        - Hill climbing
        - Simplex / amoeba / Nelder Mead
        - Genetic algorithms
    - gradient를 알고 있을 경우, 좀더 효과적으로 최적화 문제를 풀 수 있다.
        - Gradient descent: 편미분 이용
        - Conjugate gradient
        - Quasi-newton: Newton method(2차 편미분)을 통해 좀더 빠른 속도로 문제를 푸는 방법, 대신 2차 편미분을 구하려면 연산이 너무 어려워져서 Quasi-newton이 나옴
- 여기서 우리는 gradient descent를 사용할 것이다.



## Policy Gradient
- 어떤 목적함수 J가 있어서 J에 대한 gradient를 구할 수 있는 상황일 때, gradient를 구해서 theta를 바꿨을 때, J가 제일 급격하게 변하는 방향으로 \alpha만큼 업데이트 해주는 방식
    - policy \pi는 우리가 정한 함수이기 때문에, 이 때 gradient를 구할 수 잇는 함수로 정의하면 된다.
    - neural net을 쓰는 이유도, gradient를 잘 계산할 수 있는 함수이기 때문이다.
- 지난 시간에서는 error를 minimize하는 방법이었고, 여기서는 maximize하는 문제이기 때문에 gradient ascent라고 부른다.
- J에 대한 gradient를 구하면 벡터가 나온다.
    - gradient는 각 칸에 대한 편미분이다.
    - 파라미터가 n개가 있으면(\theta_1 ~ \theta_n), \theta_1으로 편미분한 값, \theta_2으로 편미분한 값, ..., \theta_n으로 편미분한 값이 모여서 n개의 벡터가 된다. 벡터는 방향과 크기가 있기 때문에, 이 방향과 크기에 \alpha를 곱한 값 만큼 \theta를 업데이트를 해준다.
- policy gradient는 J에 대한 gradient를 업데이트하는 것이다. \pi에 대한 gradient가 그 과정에서 쓰일 수는 있어도, \pi에 대한 gradient를 업데이트하는 것이 아니다.
    - J에 대한 gradient를 구해서, 목적함수 J를 최대화하는 방법론



## Computing Gradients By Finite Differences
- J에 대한 gradient를 구하는 것이 쉽지가 않다. \pi에 대한 gradient를 알아도 쉽지가 않다.
- 제일 무식하고 쉬운 방법
    - \theta_1 ~ \theta_n까지 차례를 돌아가면서 \theta_1 차례에서 \theta_1을 조금 바꿔본다. \theta_1이 1이었는데, 1.01을 넣어본다. 그럼 J의 값이 나올텐데, 이때 \theta_1의 편미분을 구할 수 있다. 왜냐하면 0.01만큼 바꿨을 때, J가 어떻게 바뀌는지 값을 구할 수 있고, 이 값이 \theta_1의 기울기이기 때문이다. 다음으로 \theta_2를 조금 바꿔서 J값을 구하고 그 차이를 구하면 \theta_2의 기울기를 구할 수 있다. 이렇게 n번 반복하면, 길이가 n인 벡터가 나오고 이것이 gradient가 된다.
    - n dimension gradient를 한번 구하기 위해 evaluation을 n번 해야 한다. 이 evaluation은 비용이 높은 행동이다. 왜냐하면 이 policy를 게임을 여러번해서 평균을 내는 것이기 때문에 쉽지 않다. 그래서 한번 업데이트가 매우 느리다.
    - 굉장히 단순하지만, evaluation을 많이 할 수 없기 때문에 noisy하고, 비효율적이다.
        - 만약 weight가 1백만개가 있다면(neural net의 경우 1억까지 갈 수도 있다), 1번 업데이트하기 위해서 1백만개를 평가해야 할 것이다.
    - 장점은 미분가능하지 않는 어떤 policy에 대해서도 쓰일 수 있다. 즉 미분가능하지 않아도 업데이트할 방향을 알 수 있다.



## Training AIBO to Walk by Finite Difference Policy Gradient
- 12개의 파라미터
- 12개의 차원에 대해서 이 방법론을 사용해서 학습을 했더니 잘 되었다.



## Score Function
- policy gradient를 수학적으로 접근해보자.
- policy \pi가 항상 미분가능하고, \pi에 대한 gradient를 계산할 수 있다고 가정하자.
- \pi에 대한 gradient는 분모 분자에 \pi를 곱하면, \pi의 gradient / \pi는 log\pi의 gradient와 같다.
    - 나중에 무언가를 할 때 편하기 위해 이렇게 바꾸는 것을 likelihood ratio 트릭이라고 한다.
- score function: gradint log\pi



## One-Step MDPs
- 어떤 분포를 따르는 initial state s에서 시작해서(state가 100개가 있으면, 각각에 확률이 있어서 그 확률에 따라 첫 state가 정해지고), 딱 1step을 간 뒤에 reward를 받고 끝낸다.
    - ex. 카지노 bandit machine, 어느 머신을 돌릴지에 대한 확률분포가 있다고 가정(10%의 확률로 10개를 골고루 선택한다라던지)
- J(\theta): 목적함수 3개 중의 3번째 정의를 가져옴
    - d(s): 각각의 initial state 분포
    - \pi_\theta(s, a): 해당 state에서 어떤 action을 할 것인지에 대한 확률
    - R_{s, a}: 해당 state에서 어떤 action을 선택했을 떄의 reward
- likelihood ratio trick에 의해 gradient \pi 대신 \pi x gradient log \pi를 넣어준다.
    - likelihood ratio trick을 사용하면, \pi가 생김으로써 괄호 안에 있는 값의 기댓값이 된다. 즉 J(\theta)에 대한 gradient가 기댓값으로 표현된다.
    - r: 실제 s에서 a를 했을 때 받는 값, 같은 s에서 같은 a를 해도 매번 다를 수 있다. 실제 샘플을 의미한다.
    - R_{s, a}: MDP에서의 term, S에서 a를 햇을 때, 어느 state에서 어느 reward를 받을 지 모른다. 확률변수를 나타낸다.
- gradient J가 expectation으로 표현되었으므로, 우리는 주사위를 굴리면서 policy \pi를 따라서 계속 행동을 한 다음에, 각 행동 (s, a)에 대해서 log\pi(s, a)의 gradient를 구하고, 그 때의 r을 곱해준 것이 J에 대한 gradient가 된다는 의미이다.
    - 이렇게 우리는 gradient J에 대한 unbiased estimation을 얻게 되었다.
    - 우리는 샘플들을 얻어서 구할 수 있게 되었다. \pi를 따라서 action이 선택되서 움직이다보면, 각 transition들의 샘플들이 gradient J의 샘플이 된다.
    - 각 샘플들 한개 한개는 확률변수이기 때문에 gradient J와 같지 않을 수 있지만, 결국에는 기댓값이 gradient J와 같으므로 이 샘플들로 계속 업데이트를 하면 된다.
    - gradient J는 직접 구하기가 어려운 값이다.
    - \pi가 없었으면 expectation 형태로 만들 수 없었을 것이다. 우리는 \pi에 대한 expectation을 만들어야 하는데 왜냐하면, 우리는 실제 현장에서의 경험은 실제 policy를 따라 얻는 경험 밖에 없기 때문이다. 그래서 \pi_theta에 대한 expectation이 우리가 원하는 값을 구하기 편리하게 만든다.
- likelihood ratio trick이 없었다면, 직접 gradient \pi x R을 구해야 할 텐데, 이 값은 구하기가 힘들다. 왜냐하면, 이 environment에서 움직이는 agent가 얻는 경로와 별로 관련이 없게 되기 때문이다.



## Softmax Policy
- neural net에서 굉장히 많이 쓰이는 형태
- 분모는 모든 feature들의 합
- 확률 형태로 각 action들이 표현되는 operator
- softmax policy인 경우, score function을 쉽게 구할 수 있다.



## Gaussian Policy
- gaussian policy의 경우 아래와 같이 scroe function을 구할 수 있다.
- 어떤 state에서 대체로 어떤 action을 할 것인데, 약간의 variance를 줘서, 그 action으로부터 조금 퍼져있는 형태로 다른 action을 할 수 있는 policy
- gaussian 분포를 이용한 policy



## One-Step MDPs
- 각 샘플들로부터 J에 대한 gradient를 구할 수 있다.
- 통계적으로 엄밀한 표현은 J의 gradient에 대한 샘플을 얻는 것이다. 이 샘플들은 unbiased 샘플들이므로, 여러번 업데이트를 하면 대수의 법칙에 의해 gradient J에 수렴하게 된다.
- 개발하는 관점:
    - \pi를 정한다. ex. softmax policy
    - feature들을 정하고, feature의 weight를 곱해서 softmax를 거쳐서 값을 만든다.
    - log를 적용한다.
    - gradient를 적용한다. ex. compute gradient 함수(tensorflow)
    - reward를 곱한다.
    - policy로 게임하는 환경을 만들고, 각각 샘플들이 나오면 그 샘플들의 위의 gradient와 reward를 곱해서 업데이트하면 된다.



## Policy Gradient Theorem
- 지금까지는 one-step MDP에 대해서만 이야기했지만, 일반적으로 multi-step MDPs에서는 어떨까?
    - likelihood ratio trick을 이용한 방법론을 Multi-step MDPs까지 일반화할 수 있게 해주는 정리가 Policy gradient theorem이다.
    - 이전 term에서 r자리를 Q로 대체하면, 그게 J의 gradient와 같다.
    - 왜 r을 Q로 대체할 수 있을까?
        - 직관적으로 접근해보면, one-step MDPs에서는 reward r을 한번 받고 끝내는 것이니까, r이 곧 accumulate reward였다. 즉 r이 게임이 끝날때 까지 받은 reward의 총합이었는데, multi-step MDPs에서는 그 step에서 a를 했을 때 얼마를 받을 지를 모두 더해주어야 하는데, Q가 그 state에서 action을 했을 때 받을 reward의 총합을 의미하므로, Q가 r의 자리를 대체해주면 multi-step MDPs에서도 똑같이 성립힌다.
    - 목적함수 3개 중 어떤 것을 사용해도 성립힌다.
- score function x Q



## Monte-Carlo Policy Gradient(REINFORCE)
- 