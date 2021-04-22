---
layout: post
title:  "Lecture 10: Classic Games"
date:   2021-04-22 13:25:05 +0900
category: "Reinforcement Learning"
tags: YouTube
plugins: mathjax
---

David Silver 님의 [Introduction to reinforcement learning](https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) 강의 내용을 [팡요랩 Pang-Yo Lab](https://www.youtube.com/channel/UCwkGvF7xKz2E0Lv-fZ9wv2g)의 [강화학습의 기초 이론](https://youtube.com/playlist?list=PLpRS2w0xWHTcTZyyX8LMmtbcMXpd3s4TU)에서 풀어서 설명하기에, 이를 바탕으로 공부하여 정리한 내용입니다.

- Video: [[강화학습 10강] Classic Games](https://youtu.be/C5_2v4pRc5c)
- Slide: [Lecture 10: Classic Games](https://www.davidsilver.uk/wp-content/uploads/2020/03/games.pdf)


---

## State of the Art

### Why Study Classic Games?

- 규칙은 간단하지만, 개념은 복잡하다.
    - Ex: 바둑, 체스, 보드게임
- 수백, 수천 년 동안 경험과 지식이 축적된 분야이다.
- IQ 테스트로써 의미도 있다.
- AI의 초파리 실험과 같은 존재이다.
    - 초파리: 생애 주기가 짧고 번식력이 좋아서 실험하는데 많이 쓰인다.
- 현실 세계의 상황들이 포함된 작은 우주로 생각할 수 있다.
- 게임은 재미있기 때문이다.


### AI in Games: State of the Art

![AI in Games: State of the Art](/assets/rl/ai_in_games_1.png)

- 각 게임에 대해 어떤 AI 프로그램이 어느 수준까지 도달했는를 나타내는 표이다.
    - Perfect: 최적의 해를 알고 있다. 즉 신이 와도 못 이기는 완전한 플레이를 한다.
    - Superhuman: 챔피언을 이겼다.
        - Ex: IBM에서 만든 딥블루가 체스 챔피언을 이겼다.
    - Grandmaster: 프로 선수를 이겼다.
        - 지금은 AlphaGo로 인해 superhuman에 도달했다.

![AI in Games: State of the Art](/assets/rl/ai_in_games_2.png)

- 각 게임에 대해 어떤 RL 프로그램이 어느 수준까지 도달했는를 나타내는 표이다.
    - 각 게임에 대해 AI 프로그램과 비교해보면, 많은 게임에서 RL을 사용했다는 것을 알 수 있다.
    - IBM에서 만든 딥블루는 RL을 사용하지 않았음을 알 수 있다.
- AlphaGo Zero 방법론은 어디서든 쓸 수 있는 방법론이다.
    - 바둑은 턴 기반의 1:1 제로섬인 보드게임이므로, 몇 가지 조건만 만족되면 같은 방법으로 모든 게임을 할 수 있다.


---

## Game Theory

### Optimality in Games

- 게임의 optimality란 무엇일까?
    - 지금까지는 agent가 environment 안에서 reward를 가장 많이 받으면, 제일 잘한다고 할 수 있다.
        - Ex: Atari 비디오 게임은 정해진 곳에서 reward를 maximize 하는 것이 목적이었다.
    - 그럼, 2명의 플레이어가 1:1로 게임할 때의 optimality는 무엇일까?
        - 이 경우, 상대에 따라 optimality가 달라진다.
        - Ex: 가위바위보 게임에서 주먹만 내는 플레이어에게는 보를 내는 것이 optimal이고, 보만 내는 플레이어에게는 가위만 내는 것이 optimal일 것이다. 그런데, 가위바위보에서 optimal한 것은 모든 플레이어가 가위바위보를 1/3씩 완전히 랜덤하게 내는 것이다. *그렇다면 이 것을 어떻게 정의해야 할까?*{: style="color: red"}
            - 게임 이론
- 플레이어가 여러 명 있고, $i$번째 플레이어 입장에서의 optimal policy $\pi^i$는 무엇일까?
    - $i$번째 플레이어를 제외한 다른 플레이어들의 policy가 고정($\pi^{-i}$)되어 있다고 가정하자.
        - Best response $\pi^i_\*(\pi^{-i})$: 나머지 플레이어들의 policy가 고정되어 있을 때, 그 고정된 policy $\pi^{-i}$에 대해 최적의 policy를 의미한다.
            - Ex: 10명과 게임을 하고 있는데, 나는 3번 플레이어라고 하자. 이 때, 나머지 9명의 policy가 고정되어 있다고 하면, 이 9개의 policy에 대한 최적의 policy가 있을 것이고, 이를 best response라고 한다.
        - Nash equilibrium (내시 평형): 모든 플레이어 policy가 상대방에 대해 best response인 경우를 말한다.
            - Ex: 10명과 게임을 하고 있는데, 3번 플레이어 입장에서 나머지 9명에 대해 best response를 갖고 있고, 2번 플레이어 또한 나머지 9명에 대해서 best response를 갖고 있다. 이렇게 각 플레이어들의 policy가 모두 best response인 경우, 서로 policy를 바꿀 요인이 전혀 없다. 즉 서로 고정된 policy로 균형을 이루고 있는 상태를 내시 평형이라고 한다.


### Single-Agent and Self-Play Reinforcement Learning

- Best response: Single-agent RL problem의 해답이다.
    - Single-agent RL problem: 지금까지 봐왔던 문제들이다.
    - Agent 외 다른 플레이어들의 policy는 고정되어 있어야 하므로, 다른 플레이어는 environment의 일부가 된다.
    - 다른 플레이어 역할을 environment가 대신하게 되면, 수학적으로 MDP로 표현될 수 있고, best response는 이 MDP의 optimal policy가 된다.
- 1:n 플레이 게임으로 넘어오게되면, environment은 바뀌게 된다.
- Nash equilibrium: Self-play RL에서 어느 순간 policy가 변하지 않는 시점이다.
    - Experience: 플레이어들이 게임을 하면서 experience가 만들어 진다.
        - $a_1 \sim \pi^1, a_2 \sim \pi^2, \dots$
        - $a_1$은 policy $\pi^1$을 따르고, $a_2$은 policy $\pi^2$을 따른다.
    - 각 플레이어는 상대 플레이어들에 대해서 best response를 학습한다. 즉, 어떤 플레이어의 policy가 업데이트 될 때, 상대 플레이어들은 environment가 되고, 또 다른 플레이어들에게도 마찬가지의 환경에서 policy가 업데이트 된다. 이 과정에서 각자의 policy가 fixed-point를 찾아가는 과정이 Nash equilibrium을 찾아가는 과정이다.
    - 그렇다면 Nash equilibrium는 unique할까?
        - 각 플레이어들의 전략이 바뀌지 않는 전략들의 집합이 하나일 필요는 없으므로, 일반적으로 unique하지 않다.
            - Ex: 두 국가가 전쟁을 하려는데, 먼저 전쟁을 일이켜 쳐들어간 국가가 불리하다고 하자. 그렇다면, 두 국가 모두 쳐들어가지 않는 수가 최고의 수가 될 것이다. 또는, 두 국가가 전쟁 중일 때, 먼저 전쟁을 멈출 경우 손해라고 하자. 그렇다면, 두 국가 모두 전쟁을 멈추지 않을 것이다. 이 경우, Nesh equilibrium은 여러 개가 될 수 있다.
        - 어떤 조건들을 만족한다면, 하나의 Nash equilibrium만 존재한다.


### Two-Player Zero-Sum Games

- 하나의 Nash equilibrium만 존재할 3가지 조건
    1. Alternating two-player game: 턴 기반으로 2명이서 하는 게임
        - Ex: 바둑의 경우, 한명은 백 돌을 잡고, 상대방은 흑 돌을 잡아서 게임을 한다. 이 때, 흑과 백을 번갈아가면서 게임을 한다.
        - Ex: 보드게임
    2. Zero sum game
        - Ex: 바둑의 경우, 흑에게 이득이 되면 백은 손해를 보고, 백에게 이득이 되면, 흑은 손해를 보게 되어 모든 reward의 합이 0이 되는 게임이다. 두 플레이어가 힘을 합치는 게임이 아닌, 상호간에 공격하는 게임이다.
        - Ex: 철권
    3. Perfect information
        - Ex: 포커의 경우, 딜러의 패를 모르기 때문에 perfect information이 아니다. 만약 바둑의 경우 상대의 state를 정확히 알 수 있으므로, hidden 정보가 없는 perfect information이다.
- 위의 3개 조건이 만족되면 단 하나의 Nesh equilibrium이 존재한다.
    - Ex: 가위바위보를 할 때, 어떤 플레이어가 주먹만 내는 전략을 따르고 있다면, 상대방은 보를 내도록 전략을 바꿀 것이다. 이 때, 상대방은 해당 플레이어에 대해 best response이지만, 해당 플레이어에게 상대방은 best response가 아니다. 그리고 이 플레이어는 가위만 내는 전략으로 바꿀 것이다. 즉 어떤 한 플레이어가 전략을 수정할 동기가 발생했기 때문에, 상대방도 다시 전략을 바꾸게 된다. 만약 1/3의 확률로 가위바위보를 낸다면, 두 플레이어 모두 전략을 바꿀 요인이 없으므로 Nesh equilibrium에 도달하고, 이 경우 단 하나의 Nesh equilibrium을 가진다.
- 그렇다면, Nesh equilibrium을 찾기 위한 방법론을 알아보고자 한다.
    - Game tree search
    - Self-play reinforcement learning


### Perfect and Imperfect Information Games

- Perfect information or Markov game: Fully observable 게임
    - Ex: Chess, Checkers, Othello, Backgammon, Go
- Imperfect information game: Partially observed 게임
    - Ex: Scrabble (낱말 맞추기), Poker


---


## Minimax Search

### Minimax

- Minimax: Search 기반 방법론 중 가장 대표적인 방법론이다.
    - 지금까지의 value function은 agent가 고정된 environment안에서 어떤 policy를 따라 행동을 선택했을 때 받은 return의 기댓값이었다.
    - *2-player 게임에서는 policy가 2개이므로, value function는 각 플레이어의 policy $\pi = \langle \pi^1, \pi^2 \rangle$의 return의 기댓값이 된다.*{: style="color: red"}
        - $v_\pi = \mathbb{E} [G_t \| S_t = s]$
        - 바둑을 기준으로 설명하면, return은 백의 입장과 흑의 입장에서 다를 수 있다. 백이 먼저 게임을 시작한다고 가정하자. 백이 이길 때 reward를 1이라고 하면, 흑이 질 때 reward도 1이 될 것이다. 백이 질 때 reward를 -1이라고 하면, 흑이 이길 때 reward도 -1이 될 것이다. 따라서 value function을 하나로 고정해야 하는데, 여기서는 백 입장에서 본 value를 value function으로 정의한다.
- Minimax value function
    - $v_\* = \max_{\pi^1}\min_{\pi^2} v_\pi(s)$
    - 백의 policy를 $\pi^1$이라고 할 때, 백의 입장에서는 value function을 maximize하는 것이 목표이고, 흑의 policy를 $\pi_2$라고 할 때, 흑의 입장에서는 value function을 minimize하는 것이 목표이다.
    - 어떤 state에 대해서 백은 백의 최선을 다하고(value를 maximize하는 것), 흑은 흑의 최선을 다할 때(value를 minimize하는 것)의 value를 minimax value function이라고 하며, 이 value function을 찾는 것이 목표이다.
- Minimax policy: minimax value function을 계산할 때 사용하는 policy $\pi = \langle \pi^1, \pi^2 \rangle$이다.
- [Two-Player Zero-Sum Games](#two-player-zero-sum-games)의 3가지 조건을 만족하므로 unique minimax value function을 가지며, 이 때의 minimax policy는 Nash equilibrium이 된다.


### Minimax Search

![Minimax Search](/assets/rl/minimax_search.png)

- Minimax value는 depth-first game-tree search로 구할 수 있다.
- Introduced by Claude Shannon: Programming a Computer for Playing Chess
    - 어떤 AI 텍스트북을 보면, chapter 1이 minimax search인 경우가 있다. 요즘은 neural net으로 다 된다고 생각될 수 있겠지만, 실제로 이 search는 굉장히 도움된다. 요즘 학생들은 이런 부분들을 등한시 여기는 것 같다.


### Minimax Search Example

![Minimax Search Example](/assets/rl/minimax_search_example_1.png)

- Root 노드가 검은 색이지만, 바둑의 백 돌부터 시작한다.
- 실제로는 더 많은 action이 존재하지만, 문제를 간단하게 하기 위해 각 플레이어들의 action이 2개 밖에 없다고 가정한다.
- 처음에 백 돌이 어떤 action을 선택하면, 다음으로 흑 돌이 어떤 action을 선택하고 다시 턴을 넘기면서 반복하다가, 게임이 끝나고 value가 나온다.
- Leaf 노드의 value는 모두 다른데, 가장 왼쪽 하단의 검은 노드로 표시된 백 돌은 어떤 행동을 선택해야 할까?
    - 백은 value를 maximize하는 것을 목표로 하기 때문에, $a_1$를 선택한다.

![Minimax Search Example](/assets/rl/minimax_search_example_2.png)

- 3번째 열의 검은색 노드로 표시된 백 돌은 value를 maximize하는 action을 선택하므로, 자식 노드 중 가장 큰 값으로 채워진다.
- 다음으로 2번째 행 왼쪽의 흰색 노드로 표시된 흑 돌은 어떤 행동을 선택해야 할까?
    - 흑은 value를 minimize하는 것을 목표로 하기 때문에, $b_2$를 선택한다.

![Minimax Search Example](/assets/rl/minimax_search_example_3.png)

- 2번째 열의 흰색 노드로 표시된 흑 돌은 value를 minimize하는 action을 선택하므로, 자식 노드 중 가장 작은 값으로 채워진다.
- 다음으로 root 노드로 노드의 검은색 노드로 표시된 백 돌은 어떤 행동을 선택해야 할까?
    - 백은 value를 maximize하는 것을 목표로 하기 때문에, $a_1$를 선택한다.

![Minimax Search Example](/assets/rl/minimax_search_example_4.png)

- Root 노드의 검은색 노드로 표시된 백 돌은 value를 maximize하는 action을 선택하므로, -2가 채워진다.
    - 이 게임은 세 수만에 끝나는 굉장히 간단한 게임으로, 현재 state부터 완벽하게 플레이했을 때, 무조건 흑 돌이 이기도록 설계되었다.
    - 실제 바둑에서도 똑같이 만들어 낼 수 있지만, 이 때 만들어지는 tree는 말도 안 되게 클 것이다.


### Value Function in Minimax Search

- Search tree가 exponential하게 커질 것이다.
    - 선택할 수 있는 action이 2개 뿐이라도, 3턴을 거치면 $2^3$개의 leaf가 생긴다.
    - 일반적인 classic 게임에서는 훨씬 더 많은 action space가 있을 것이다.
- 게임이 끝날 때까지 search를 하는 것은 불가능하다.
- 대신 value function approximator $v(s, \mathbf{w}) \approx v_\*(s)$를 사용해서 tree 끝까지 search하지 않고 어느정도 깊이만큼 제한을 두고 minimax search를 진행한다.
    - Value function approximator $v(s, \mathbf{w})$
        - Evaluation function
        - Heuristic function
        - Neural net
    - 고정된 깊이만큼 search를 진행하면, 거기서 더 게임을 진행할 수 있지만 멈추고, 해당 leaf 노드에 value function을 계산한다.
        - Ex: 현재 state부터 번갈아가면서 3수까지만 플레이를 진행하고, 해당 state의 value function을 계산한다. 그리고 minimax search를 진행하면 현재 state의 minimax value를 계산할 수 있다.
- Minimax를 효과적으로 적용하기 위한 연구가 수십 년 동안 이어져 왔다.
    - Alpha-beta search: Tree에서 관심없는 부분은 제거하고, 관심있는 부분만 더 많이 search한다.
        - 강의에서 자세히 다루지 않는다.
    

### Binary-Linear Value Function

![Binary-Linear Value Function](/assets/rl/binary_linear_value_function.png)

- Minimax search에 value function도 함께 사용된다.
- 그렇다면 이 value function을 어떻게 구성할까?
    - 제일 간단한 방법은 binary-linear function을 사용하는 것이다.
        - Feature vector $\mathbf{x}(s)$에 weight vector $\mathbf{w}$를 곱한 값이 해당 state $s$의 value가 된다.
        - Weight vector $\mathbf{w}$를 어떻게 설정할까?
            - Ex: 위 예시에서, 체스 기물의 weight를 castle은 +5, bishop은 +3, pawn은 +1로 정의하고 상대방 기물의 weight에는 -를 곱해준다. 여기서 feature vector은 해당 기물의 유무로 정의되어 있으므로, feature vector와 weight vector를 곱한 값은 value function으로 사용한다. 이렇게 구한 값은 실제 value가 아닐 수도 있지만, 현재 state가 이 정도의 value를 가진다고 approximation한다.
        - 이 feature는 사람이 만든다. 수만 번 게임을 플레이해서 계산한 value가 아니라, 해당 domain의 전문가들이 만든 feature이다.
        - 기물의 존재 여부만으로 feature vector를 정의했지만, combination으로 정의하거나 특정 위치에 대해 점수를 줄 수도 있을 것이다. 그리고 이런 feature들을 다양하게 정의할 수 있다.


### Deep Blue

- Knowledge
    - 8천 개의 feature
    - Binary-linear value function
    - 전문가들이 각 상황들의 점수를 토의해서, weights를 만들었다.
- Search
    - Alpha-beta search: Minimax search의 한 종류
    - Tree에서 깊이를 16 ~ 40수까지 수행한 뒤, leaf 노드에 전문자의 knowledge로 계산한 value function을 넣은 뒤, 현재 state의 minimax value를 계산했다.
- Results
    - 1997년도 체스 챔피언 Garry Kasparov를 4-2로 이겼다.
    - 인터넷 역사상 가장 많이 시청한 행사였다.
- Value function을 사람이 직접 만들었기 때문에, RL이 사용되지 않았다.


### Chinook

- 비슷한 방법론을 체커 게임에 적용한 프로그램이다.
- Knowledge
    - Position, mobility 등의 지식 기반의 21개의 feature
    - Binary-linear value function
- Search
    - Alpha-beta search
    - Retrograde analysis
        - 게임이 이기는 state로부터 backward search
        - Lookup table에 이긴 위치들을 저장
        - 약 11개의 체커가 나오면, 완벽하게 플레이한다.
- Results
    - 1997년도 체커 챔피언 Marion Tinsley를 4-2로 이겼다.
        - Marion Tinsley는 평생 체커 게임에서 7판밖에 안진 게임 역사상 최고의 선수였고, 그 중 2판을 Chinook에게 졌다.
    - 2007년도에는 체커 게이을 풀어서, perfect하게 플레이하게 되었다.
        - 체커는 $10^{24}$정도 크기의 게임인데, 당시 computation이 많이 발전하면서 게임이 끝날 때까지 search를 할 수 있었다.


---


## Self-Play Reinforcement Learning


### Self-Play Temporal-Difference Learning

- Self-play를 통해 만들어낸 경기에 지금까지 배운 value-based RL 알고리즘을 적용한다.
    - 실제 value와 내가 예측한 value 사이의 차이를 제곱한 값을 minimize한다.
        - Mean squared error를 미분한 뒤, gradient descent를 적용한다.
    - MC: Return $G_t$를 통해 value function을 업데이트한다.
        - $\Delta\mathbf{w} = \alpha (G_t - v(S_t, \mathbf{w})) \nabla_\mathbf{w} v(S_t, \mathbf{w})$
    - TD(0): Successor value $v(S_{t+1})$을 통해 value function을 업데이트한다.
        - $\Delta\mathbf{w} = \alpha (v(S_{t+1}, \mathbf{w}) - v(S_t, \mathbf{w})) \nabla_\mathbf{w} v(S_t, \mathbf{w})$
    - TD($\lambda$): $\lambda$-return $G_t^\lambda$을 통해 value function을 업데이트한다.
        - $\Delta\mathbf{w} = \alpha (G_t^\lambda - v(S_t, \mathbf{w})) \nabla_\mathbf{w} v(S_t, \mathbf{w})$
- 2-player 게임은 대부분 intermediate reward가 없다. 즉, 게임 도중에는 reward가 발생하지 않고, 게임이 끝난 뒤 이겼는지 졌는지만 나온다.
    - Ex: 0, 0, 0, ... 1 또는, 0, 0, 0, ... -1
- *따라서 $\gamma$과 $R$을 target에서 모두 제외시켰다. 즉, 다음 state의 value로 현재 state의 value를 업데이트 하지만, 여기서의 value는 모두 minimax value function의 approximation이다.*{: style="color: red"}


### Policy Improvement with Afterstates

- Afterstate: 게임의 룰을 완전히 안다면, 다음 state로 가기 위해 어떤 action을 선택해야 할 지 알게 된다. Deterministic game의 경우, 현재 state에서 이동할 수 있는 다음 state가 무엇인지 알 수 있다.
    - Ex: 바둑의 경우, 현재 state에서 어떤 위치에 돌을 둘 때, 우연히 다른 위치에 돌을 두는 경우는 없다. 즉, 어떤 수를 두면, 정확히 원하는 다음 state로 위치하게 된다.
    - Ex: 체스의 경우, 현재 state에서 어떤 기물을 어떻게 움직이느냐에 따라, 의도했던 다음 state로 이동할 수 있다.
    - *$q_\*(s, a)$가 아니라 $v_\*(s)$를 알기만 해도 policy improvement를 할 수 있다.*{: style="color: red"}
        - $q_\*(s, a) = v_\*(\text{succ}(s, a))$
            - $\text{succ}(s, a)$: 게임 규칙으로 정의된 successor state
        - Model-free RL에서는 model을 모르기 때문에, 어떤 action을 선택해야 현재 state에서 원하는 state로 이동할 수 있는지 알 수 없었다. 그래서 해당 state에서 어떤 action을 선택했을 때의 $q_\*$를 통해 value를 계산할 수 있었고, 이를 통해 policy를 향상시킬 수 있었다.
    - 즉, 어떤 state의 $v_\*$가 무엇인지만 알면, 현재 state에서 가장 큰 value를 갖는 state로 움직일 수 있는 action을 선택하면 된다.
        - $\begin{aligned}
            A_t = \arg\max_a v_\*(\text{succ}(S_t, a)) \  \text{for white}  \newline
            A_t = \arg\min_a v_\*(\text{succ}(S_t, a)) \  \text{for black}
        \end{aligned}$
        - 백 돌의 입장에서는 다음 state가 가장 큰 value를 가지는 action을 선택하면 되고, 흑 돌의 입장에서는 가장 작은 value를 가지는 action을 선택한다.
    - 이렇게 하면, self-play RL에서는 policy iteration을 통해 $v_\*$만으로 policy improvement를 할 수 있다.
- 그럼 $v_\*$는 어떻게 계산할까?
    - MC 또는 TD를 이용해서 계산한다.


### Self-Play TD in Othello: Logistello

![Logistello](/assets/rl/logistello.png)

- 돌의 position 조합으로 150만개의 feature를 만들었다.
    - 위 그림의 2행을 보면, 대각선의 돌의 갯수와 위치에 따라 모든 combination으로 구성된 feature를 정의해서, binary linear function을 만들었다.
    - 이 150만 개의 조합은 게임에서 빈번하게 발생하는 조합이다.
- 이는 handcrafted feature와는 조금 다르다. Handcrafted feature은 위치에 따라 몇 점인지 직접 정해주어야 하는데, 이 프로그램에서는 돌들이 연결되었는지 분리되었는지 여부로 feature가 정의되었다.


### Reinforcement Learning in Logistello

- Logistello는 generalised policy iteration을 사용했다.
    1. 현재 policy로 self-play를 진행하여 경험을 만든다.
    2. 이 경험을 바탕으로 Monte-Carlo 방법으로 현재 policy를 평가한다.
        - Monte-Carlo 방법을 사용한다는 것은 outcome을 regression한다는 것과 같은 의미이다.
            - Outcome: 승패 여부를 의미한다.
            - Regression: Supervised learning처럼 숫자 맞추기를 의미한다.
            - 어떤 state에서 게임을 해서 이겼을 때는 1이고 졌을 때는 -1이라고 할 때, 이 경험들에 대해 input은 state이고 output은 1 또는 -1인 regression 관계를 학습하는 것이다.
    3. 평가된 value를 greedy policy improvement하여 새로운 policy 만든다.
        - Greedy하게 선택할 때는, 백 돌의 입장에서는 maximize value를 가지는 state로 이동하도록 하고, 흑 돌의 입장에서는 minimize value를 가지는 state로 이동하도록 한다. (minimax)
    4. 다시 새로운 policy를 통해 위 과정을 계속 반복한다.
        - Policy iteration: Policy evaluation을 하고, 각 state에 대해 평가된 value를 바탕으로 greedy 또는 $\epsilon$-greedy하게 action을 선택하여 새로운 policy를 만든다. 그리고 새로 만든 policy를 바탕으로 다시 evaluation을 하고 이 과정을 반복한다.
- 위와 같이 학습하여, 당시 오델로 챔피언인 Takeshi Murukami를 6-0으로 이겼다.
- Logistello는 superhuman level을 달성한 첫 강화학습 프로그램이다.


### TD-Gammon


#### TD Gammon: Non-Linear Value Function Approximation

![TD Gammon](/assets/rl/td_gammon.png)

- IBM에서 만든 TD-Gammon: Backgammon이라는 보드게임을 하는 프로그램이다.
    - 당시 가장 유명한 AI였다.
    - Neural net을 사용했지만, 당시 computation이 어려웠기 때문에 각 layer의 노드 수는 80개, 40개로 얕은 네트워크였다.
        - Raw feature을 사용했다.


#### Self-Play TD in Backgammon: TD-Gammon

- Neural net을 random weight로 초기화했다.
- Self-play로 학습했고, non-linear TD learning을 사용했다.
    - $\begin{aligned}
        \delta_t &= v(S_{t+1}, \mathbf{w}) - v(S_t, \mathbf{w})     \newline
        \Delta\mathbf{w} &= \alpha \delta_t \nabla_\mathbf{w} v(S_t, \mathbf{w})
    \end{aligned}$
        - 실제로는 TD($\lambda$)를 사용했다.
        - $R$과 $\gamma$가 사라졌다.
- Exploration을 젼혀 사용하지 않고, greedy policy improvement를 했음에도 항상 수렴했다.
    - Backgammon에는 주사위로 플레이를 하는데, 이 주사위에 확률성이 있어서 매번 다른 state를 방문하도록 만든다. 이 과정에서 exploration과 같은 효과를 내기 때문에 따로 exploration을 하지 않아도 학습됐다.
    - 하지만 일반적으로 사용할 수 있는 상황은 아니다.


#### TD Gammon: Results

- Zero expert knowledge → strong intermediate play
    - Raw position만 사용했고, 계속 연구되고 있었다.
- Hand-crafted features → advanced level of play (1991)
- 2-ply search → strong master play (1993)
- 3-ply search =⇒ superhuman play (1998)
    - Minimax search의 depth를 늘려나갈수록 점점 높은 수준에 도달했다.
- Defeated world champion Luigi Villa 7-1 (1992)


#### New TD-Gammon Results

![New TD-Gammon Results](/assets/rl/new_td_gammon.png)

- TD-gammon을 연구하던 분이 neural net의 hidden unit 갯수를 늘려나갔을 때, 성능을 확인했다.
- 그래프에서 점이 높게 표시되어 있을수록 성능이 좋은 것인데, hidden unit이 많을수록 성능이 좋았고, 80 hidden unit을 사용했을 때 가장 좋은 성능을 보였다.

---

## Combining Reinforcement Learning and Minimax Search

### Simple TD

![Simple TD](/assets/rl/simple_td.png)

- RL과 minimax search를 결합하기 위해 3가지 방법을 알아보자.
    - 그 전에 예전에 배웠던 simple TD를 생각해보자.
- TD: 현재 value를 successor value를 통해 업데이트한다.
    - $v(S_t, \mathbf{w}) \leftarrow v(S_{t+1}, \mathbf{w})$
        - $R$과 $\gamma$가 사라졌다.
    - 지금까지는 TD learning으로 value function을 학습한 뒤, 이 value function에 minimax search를 사용했다.
        - $v_+(S_t, \mathbf{w}) = \text{minimax}\_{s \in \text{leaves}(S_t)} v(s, \mathbf{w})$
        - MCTS를 사용하는 AlphaGo1와 Logistello, TD-Gammon 모두 비슷한 방식을 사용했다.
            - Ex: Minimax search에서 3-depth까지만 탐색하고, 3-depth 이후는 버린 leaf 노드에서 value function을 계산했다.
        - *Learning과 search가 분리되었기 때문에, minimax search의 결과가 value function 업데이트에 전혀 사용되지 않았다.*{: style="color: red"}
            - 그런데 꼭 이렇게 해야할 필요가 있을까? Search 결과가 learning에 영향을 주어서 더 정확한 value를 만들 수 있지 않을까? 그리고 더 정확해진 value가 다시 search에 더 좋은 영향을 줄 수 있지 않을까?


### Simple TD: Results

- Othello와 Backgammon에서는 좋은 성능을 보였지만, Chess와 Checkers에서는 잘 안됐다.
- Search를 아예 학습과정에 포함시켜서 value function을 학습시키는데 사용할 수 없을까?
    - TD Root
    - TD Leaf
    - TreeStrap


### TD Root

![TD Root](/assets/rl/td_root.png)

- 현재 state를 $S_t$, 다음 state를 $S_{t+1}$이라고 할 때, TD root는 아래 과정으로 진행된다.
    1. $S_{t+1}$에서 minimax search를 하여, search value를 구한다. 오른쪽 tree의 초록색 leaf 노드의 value가 minimax value라고 하자.
        - $v_+(S_{t+1}, \mathbf{w}) = \text{minimax}\_{s \in \text{leaves}(S_{t+1})} v(s, \mathbf{w})$
        - 이 초록색 leaf 노드의 value는 백 돌 입장에서 maximize하고, 흑 돌 입장에서 minimize한 value 값이다.
    2. 이 초록색 leaf 노드의 value를 $S_t$의 value에 덮어쓴다.
        - $v(S_t, \mathbf{w}) \leftarrow v_+(S_{t+1}, \mathbf{w}) = v(l_+(S_{t+1}), \mathbf{w})$
            - $l_+(s)$: State $s$의 minimax value가 된 leaf node
- *즉, $S_{t+1}$에서 minimax search를 하고, $S_{t+1}$의 minimax value로 $S_t$의 value를 덮어쓴다.*{: style="color: red"}
    - *Learning과 minimax search를 combine했다.*{: style="color: red"}
- TD learning에서는 $v(S_{t+1}, \mathbf{w})$로 $v(S_t, \mathbf{w})$를 업데이트 했지만, TD root에서는 $S_{t+1}$의 minimax value $v_+(S_{t+1}, \mathbf{w})$가 $v(S_{t+1}, \mathbf{w})$보다 더 정확하므로, $v_+(S_{t+1}, \mathbf{w})$로 $v(S_t, \mathbf{w})$를 업데이트한다.
    - $v_+(S_t, \mathbf{w})$: $S_t$의 minimax value
        - 왼쪽 tree의 파란색 노드
    - $v_+(S_{t+1}, \mathbf{w})$: $S_{t+1}$의 minimax value
        - 오른쪽 tree의 초록색 노드
    - 여기서는 게임의 규칙을 알고 있기 때문에, value function만 알아도 action을 선택할 수 있다. 따라서 policy는 다루지 않는다.


### TD Root in Checkers: Samuel’s Player

- First ever TD learning algorithm (Samuel 1959)
    - *TD root는 first ever TD learning algorithm이라고 한다.*{: style="color: red"}
- Applied to a Checkers program that learned by self-play
- Defeated an amateur human player
- Also used other ideas we might now consider strange


### TD Leaf

![TD Leaf](/assets/rl/td_leaf.png)

- 현재 state를 $S_t$, 다음 state를 $S_{t+1}$이라고 할 때, TD leaf는 아래 과정으로 진행된다.
    1. $S_t$와 $S_{t+1}$에서 minimax search를 하여, search value를 구한다. 왼쪽 tree의 파란색 leaf 노드의 value가 $S_t$의 minimax value이고, 오른쪽 tree의 초록색 leaf 노드의 value가 $S_{t+1}$의 minimax value라고 하자.
        - $\begin{aligned}
            v_+(S_t, \mathbf{w}) &= \text{minimax}\_{s \in \text{leaves}(S_t)} v(s, \mathbf{w})      \newline
            v_+(S_{t+1}, \mathbf{w}) &= \text{minimax}\_{s \in \text{leaves}(S_{t+1})} v(s, \mathbf{w})
        \end{aligned}$
    2. $S_{t+1}$의 minimax value $v_+(S_{t+1}, \mathbf{w})$를 $S_t$의 minimax value $v_+(S_t, \mathbf{w})$에 덮어쓴다.
        - $\begin{aligned}
            v_+(S_t, \mathbf{w}) &\leftarrow v_+(S_{t+1}, \mathbf{w})    \newline
            \implies v(l_+(S_t), \mathbf{w}) &\leftarrow v(l_+(S_{t+1}), \mathbf{w})
        \end{aligned}$
        - 1-step 더 search한 value가 더 정확할 것이기 때문이다.


### TD leaf in Chess: Knightcap

- Knightcap: TD leaf를 체스 게임에 적용한 프로그램이었고, self-play가 그렇게 효과적이지 않았다.
- Learning
    - Knightcap trained against expert opponent
    - Starting from standard piece values only
    - Learnt weights using TD leaf
- Search
    - Alpha-beta search with standard enhancements
- Results
    - Achieved master level play after a small number of games
    - Was not effective in self-play
    - Was not effective without starting from good weights


### TD leaf in Checkers: Chinook

- Chinook: TD leaf를 체커 게임에 적용한 프로그램이다.
    - Original Chinook은 hand-tuned weight를 사용했고, 나중 버전의 Chinook은 self-play로 학습시켰다.
        - Except material weights which were kept fixed
    - Self-play로 학습된 weight가 hand-tuned weight보다 더 성능이 좋았다.
- Learning to play at superhuman level


### TreeStrap

![TreeStrap](/assets/rl/treestrap.png)

- TD root, TD leaf: Real-step과 imaginary-step을 섞어서 사용하였다.
    - Real-step: 1-step 이후 실제로 더 경험하는 것을 의미한다.
    - Imaginary-step: Real-step 이후 search를 통해 tree를 만드는 것을 의미한다.
- *그런데, real-step과 imaginary-step을 굳이 섞어서 사용할 필요가 있을까? 모두 상상 속에서 계산해도 될까?*{: style="color: red"}
    - 현재 state에서 search를 통해 tree를 만들고, 생성된 모든 노드들을 업데이트한다.
        - $\begin{aligned}
            v(s, \mathbf{w}) &\leftarrow v_+(s, \mathbf{w})    \newline
            \implies v(s, \mathbf{w}) &\leftarrow v(l_+(s), \mathbf{w})
        \end{aligned}$
        - Leaf 노드의 value를 부모 노드의 value에 덮어쓴다.
        - 모든 노드들은 자식 노드의 value로 덮어 쓰인다.


### Treestrap in Chess: Meep

- Meep: TreeStrap을 체스 게임에 적용한 프로그램이다.
- Binary linear value function with 2000 features
- Starting from random initial weights (no prior knowledge)
- Weights adjusted by TreeStrap
- Won 13/15 vs. international masters
- Effective in self-play
- Effective from random initial weights


### Simulation-Based Search

- Self-play RL은 search로 대체될 수 있다.
- Root state $S_t$에서 시작하는 시뮬레이션을 Self-play로 만들고, 이 시뮬레이션된 경험들에 대해 model-free RL을 한다.
    - Monte-Carlo를 사용하면, Monte-Carlo Tree Search가 된다.
        - Search 안에 Monte-Carlo control을 적용한다. Simulated experience를 통해 각 action들에 대한 value를 학습하고, 그 중 가장 큰 action-value를 가지는 action을 선택한다(Exploitation).
        - Exploration도 섞어주기 위해, 모든 노드를 bandit으로 보고 UCB를 사용한다. 
            - 이렇게 사용한 알고리즘을 UCT 알고리즘이라고 한다. MCTS의 한 응용이다.
            - UCT: Upper Confidence Bound 1 applied to Trees
        - Self-play UCT의 minimax values는 수럼한다.
        - MCTS는 굉장히 효과적으로 학습한다.
    
### Performance of MCTS in Games

- *MCTS is best performing method in many challenging games*{: style="color: red"}
    - Go (last lecture)
    - Hex
    - Lines of Action
    - Amazons
- In many games simple Monte-Carlo search is enough
    - Scrabble
    - Backgammon


---

## Reinforcement Learning in Imperfect-Information Games

### Smooth UCT Search

![Smooth UCT Search](/assets/rl/smooth_uct_search.png)

- Imperfect information일 때, smooth UCT를 사용한다.
    - Average strategy를 만들어서 지금까지 배웠던 상대들의 average behavior를 배운다. 이를 통해 평균적인 어떤 행동을 선택해서 UCT를 업데이트한다.
    - 포커에서 많이 사용되었다.


---

## Conclusions

### RL in Games: A Successful Recipe

![RL in Games](/assets/rl/rl_in_games.png)

- 각 게임에 어떤 방법론들이 사용되었는지 정리한 표이다.
