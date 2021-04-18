---
layout: post
title:  "Classic Games"
date:   2021-03-18 01:00:00 +0900
category: "Reinforcement Learning"
tags: YouTube
---

## 참고

- video
    - [[강화학습 10강] Classic Games](https://youtu.be/C5_2v4pRc5c)
- slide
	- [Lecture 10: Classic Games](https://www.davidsilver.uk/wp-content/uploads/2020/03/games.pdf)



## Why Study Classic Games?
- 규칙은 간단하지만, 개념은 복잡하다.
    - ex. 바둑, 체스, 보드게임
- 수백, 수천년동안 경험과 지식이 축적된 분야
- 규칙이 간단한데, 생각이 많이 필요한 만큼 IQ 테스트로써의 의미도 있다.
- AI의 초파리 같은 개념
    - 초파리: 생애 주기가 짧고, 번식력이 좋아서 실험하는데 많이 쓰인다.
- 현실 세계의 상황들을 포함하는 소우주
- 게임은 재밌다.



## AI in Games: State of the Art
- 각 게임에 대해 AI가 어느 수준까지 왔는지 + 그 수준을 도달한 AI, 프로그램의 이름이 정의되어 있다.
- perfect: 완전히 풀렸다. 즉 신이 와도 못 이기는 완전한 플레이를 한다. 최적의 해를 알고 있다.
- superhuman: 사람 1등을 이긴 것
- grandmaster: 프로레벨을 이긴 것
    - 현재는 Go 때문에, superhuman level에 도달했다.
- chess의 deep blue: 체스 1등을 IBM에서 만든 딥블루가 이겼다.



## RL in Games: State of the Art
- AI in games에도 RL을 써서 학습된 것도 있고, 안써서 학습된 것도 있을 것이다.
    - RL로 한정지어서, RL로 학습된 것을 나타내고 있다.
- 이전 슬라이드와 비교해서 보면, RL이 많은 곳에서 쓰인 것을 알 수 있다.
- chess의 deep blue는 RL을 사용한 것이 아님을 알 수 있다.
    - 현재 RL이 deep blue보다 더 잘 둔다고 생각해도 될까?
        - david silver가 alphaGo zero를 만들었는데, 이 alphaGo zero 방법론은 어디서든 쓸 수 있는 방법론이다. 보드게임, 턴 기반, 1 vs. 1, 제로섬 게임이고 몇가지 조건만 만족되면, 똑같은 방법으로 모든 게임을 할 수 있다.
- Go
    - MoGo: 9x9 바둑판
    - Zen: 19x19 바둑판



## Optimality in Games
- 게임에서의 optimality란 무엇일까? 지금까지는 environment가 정해져있었다. 예를 들어 Atari 비디오 게임과 같은 정해진 곳에서 reward를 maximize하는 것은 optimality가 굉장히 간단하다. 즉 제일 잘 한다는 것은 reward를 제일 많이 받으면 되는 것을 제일 잘 한다고 하는 것이 된다.
- 지금은 2 player 게임으로 넘어왔다. ex. 체스, 체커, 바둑
    - 1:1로 2명이서 붙는 상황에서의 optimality란 무엇일까? 
    - 상대에 따라서 optimality가 달라진다. 예를 들면 가위바위보에서 주먹만 내는 player에게는 보를 내는 것이 optimal이고, 보를 내는 player에게는 가위만 내는 것이 optimal이다. 그런데 우리는 가위바위보에서 optimal은 1/3씩 완전히 랜덤하게 내는 것이라는 것을 알고 있다. 우리는 이렇게 무엇이 optimal인지는 느끼고 있는데, 이것을 어떻게 정의해야 할까?
        - 게임이론
- 플레이어가 여러명이 있고, i번째 플레이어의 입장에서 봤을 때 optimal은 무엇일까?
    - 만약, i번째 플레이어를 제외한 다른 모든 상대방들의 게임을 하는 정책 policy가 고정되어 있다고 가정하자.
        - Best response: 나머지 플레이어들의 policy가 고정되어 있다고 가정할 때, 그 고정된 policy에 대해서 최적의 policy를 말한다.
            - 예를 들어, 10명이 게임을 하는데 나는 3번째 플레이어라고 하자. 그럼 나머지 9명의 policy가 고정되어 있다고 가정할 때, 9개의 policy에 대한 최적의 policy가 있을 것이다. 이를 best response라고 한다.
            - 나머지 플레이어들의 고정된 policy에 대한 best response는 정해져 있다.
        - Nash equilibrium: 내쉬 평형
            - 모든 애들이 상호간의 policy인 것
            - 모든 player가 상대방에 대해 best response인 경우
            - 내가 10명의 플레이어와 게임을 하는데, 내 입장에서 나머지 9명에 대해서 best response를 하고 있고, 동시에 2번 플레이어 입장에서 나머지 9명에 대해서 best response를 하고 있다. 즉 서로 상호간에 서로 모두가 best response인 경우, 내 전략을 바꿀 요인이 전혀 없다. 왜냐하면 나는 지금 다른 모든 플레이어들에 대해서 best response를 하고 있기 때문, 여기서 무언가를 바꿀 이유가 없다. 그런데 나만 그런 것이 아니라, 2, 3, 4 모두 best response이기 때문에, 그 누구도 자신의 policy를 바꿀 요인이 없다.
            - 그래서 이 policy는 서로 안바꾸기 때문에, 서로 고정되고 서로 균형이 맞춰지며, 이를 nash 균형이라고 한다.




## Single-Agent and Self-Play Reinforcement Learning
- single-agnet RL: 우리가 지금까지 봐왔던 문제들
    - 게임으로 넘어오면서 2명의 플레이어가 생기면서 환경이 변하기 시작한다.
    - Best response는 single-agent RL 문제의 해답이다.
    - 왜냐하면 다른 얘들이 고정되어 있으니까, 이 고정되어 있는 애들은 환경이라고 생각할 수 있다. 상대방이 고정되어있으면, 내가 무슨 일을 하면 상대방이 무언가를 할테니까 상대방을 포함해서 환경이라고 볼 수 있다. 즉 나머지가 고정되어 있으니까 나를 제외한 모든 것을 환경이라고 볼 수 있다.
    - 게임을 MDP로 표현할 수 있다. 상대방의 policy가 환경으로 넘어가면서 수학적으로 MDP로 표현될 수 있다.
        - Best response는 이 MDP의 optiaml policy이다.
        - Best response는 다른 애들이 고정되어 있다 생각하고, single-agent RL을 푸는 것이다.
- 하지만, Nash equilibrium은 조금 더 복잡한 개념이다. self-play RL을 하는데, 나와 나 자신이 붙으면서 강화학습을 수행하는데, 어느 순간부터 policy가 바뀌지 않는 부동점을 nash equilibrium이라고 한다.
    - experience: agent들 간의 게임으로부터 experience가 만들어진다.
        - action 1은 policy 1을 따르고, action 2는 policy 2를 따르면서 서로 선택한다. 각 agent는 상대방에 대해서 best response를 학습해나가고, 한 명의 policy가 다른 player의 environment가 되니까, environment도 계속 바뀌면서 동적으로 fixed-point를 찾아 가는 과정이 nash equilibrium을 찾는 과정이다.
    - 모든 플레이어가 상대방에게 적응해가면서, 이 nash equilibrium을 찾아서 학습해나간다.
    - 그렇다면 nash equilibrium은 unique할까?
        - 각 플레이어들의 전략이 바뀌지 않는 전략의 집합이 하나일 필요는 없으므로, 일반적으로 unique하지 않다.
        - 어떤 조건들을 만족할 경우 nash equilibrium은 하나만 존재한다.
            - 이 다음장에서 계속 된다.



## Two-Player Zero-Sum Games
- 3가지 조건의 게임
    - alternating two-player game, turn 기반으로 동시에 무언가를 하는 것이 아니라 2명이서 한다.
        - 한명을 백, 한명을 흑이라고 보면, 흑/백이 번갈아가면서 게임을 하는 것
        - ex. 보드게임
    - zero sum game: 흑한테 이득이 되면 백한테 손해가 되고, 백한테 이득이 되면 흑한테 손해가 된다. 둘이 힘을 합쳐서 무언가를 하는게 아니라, 상호간에 공격을 하는 게임
        - 이들 reward의 합은 0이 된다.
        - ex. 철권
    - perfect information: 포커의 경우 상대 패를 모르기 때문에 perfect information이 아니다. 바둑, 체스의 경우 상대의 state를 정확히 알 수 있으므로 hidden 정보가 없는 perfect information이다.
- 이렇게 3개 조건을 만족되면, Nesh 평형은 unique하게 된다. 단 하나의 Nesh 평형이 존재한다.
    - Nesh 균형이라고 하면, 둘다 자신의 전략을 바꿀 요인이 없는 것, 예를 들어 두 국가가 전쟁을 하는데 전쟁을 일으켜서 쳐들어 간 국가가 불리한 상황이 된다고 한다면, 둘 다 쳐들어가지 않는 수가 최고의 추가 되고, 그러다 보면 계속 쳐들어가지 않는 상황이 계속 될 것이다. 또는 둘다 싸우고 있어서, 전쟁을 하고 있어서 먼저 전쟁을 멈출 경우 나만 손해일 경우, 그것도 두 국가간의 Nesh 균형이 될 것이다. 이 경우에는 Nesh 균형이 여러개가 있는 상황이 된다.
    - 그런데 위의 3개 조건이 만족되면, Nesh 균형이 하나밖에 존재하지 않게 된다.
- ex. 가위바위보를 하는데 내가 주먹만 내면, 상대방은 보만 내도록 수정될 것이다. 그럼 상대방은 나에 대해 Best response이지만, 나는 상대방에 대해 Best response가 아니다. 그 때 나는 가위만 내도록 할 것이다. 즉 전략을 수정할 동기가 생기고, 나는 best response가 되었을 때, 상대방은 best response가 아니므로, 다시 상대방은 전략을 수정하게 될 것이다. 이렇게 계속 바뀌어 나가니까 균형이 아닌데, 만약 내가 1/3의 확률로 완전 랜덤하게 주먹/가위/보를 내고, 상대방도 1/3로 랜덤하게 낼 경우, 두 플레이어 모두 전략을 바꿀 요인이 없으므로 이 경우 Nesh 균형이라고 할 수 있을 것이다.
- 그러므로 우리는 이 Nesh 평형을 찾는 것을 목적으로 하고 그에 대한 방법론을 알아보고자 한다.
    - Game tree search
    - Self-play reinforcement learning



## Perfect and Imperfect Information Games
- perfect information: Markov가 fully observable 한 상황
    - chess, checkers, othello, gackgammon, Go
- imperfect information
    - poker
        - 포커의 경우 nesh equilibrium이 매우 많다.
    - scrabble: 낱말 맞추기



## Minimax
- search와 self-play RL 방법론 중 search 기반 방법론
- search의 가장 대표적인 방법론이 minimax 방법론
- 지금까지의 value function은 environment가 고정되어 있고, 우리가 어떤 policy를 따랐을 때, return의 기댓값이었다. 그런데 이 2player 게임에서는 policy가 2개이므로, value function을 첫번째 player와 두번째 player의 policy가 정해졌을 때, 그 때 return의 기댓값으로 정의한다.
    - return이 백의 입장에서와 흑의 입장에서 다를 수 있다.
    - 이제부터 먼저 두는 것이 백의 입장이라고 할 때, 백 기준에서 이기면 1이고 지면 -1이면, 흑 기준에서는 이기면 -1이고 지면 1일 것이다. 이 때, value function을 하나로 고정한다. 즉, 백 입장에서 본 value로 고정해서 정의한다.
- minimax value function: 백 입장에서의 policy를 \pi^1이라고 하면, value function을 maximize하는 것이 목표이고, 반대로 흑 입장에서의  policy를 \pi^2라고 하면, value function을 minimize하는 것이 목표가 된다.
    - 현재 백 입장에서 이기면 1로 정의되었으므로, 흑 입장에서는 -1로 정의된다.
    - 흑 입장에서의 reward를 0으로도 설정할 수 있다.
    - 어쨌든, 흑은 return을 줄여야하고, 백은 return을 늘려야 한다.
    - 어떤 state에 대해서 흑은 흑의 최선(value를 줄이는 것)을 다하고, 백은 백의 최선(value를 늘리는 것)을 다해서, 둘 다 자신의 최선을 다 할때의 value 값을 minimax value function이라고 한다.
    - 어떤 state로부터 지금부터 양 플레이어가 완벽하게 플레이를 하면 누가 이길 것인가를 나타내는 의미이다.
    - 그리고 우리는 이것을 찾는 것이 목표이다.
- minimax policy: \pi^1과 \pi^2가 있어서, 이대로 하면 minimax value를 얻게되는 policy
    - 흑은 흑대로 잘하고, 백은 백대로 잘하는 policy
- 2player, alternative game이고, zero-sum game이고, perfect information game일 경우, minimax value function이 unique하다. 그리고 minimax policy는 nesh equilibrium을 만족한다.
- policy는 그 수에 대해 국한되는 개념이 아니라, 어느 상황과 관계없이 나의 정책을 의미한다. 흑도 흑의 정책이 있고, 백도 백의 정책이 있는데, minimax policy는 둘다 최선을 다해서 잘하는 경우이고, 그럴때 이 policy대로 하면 minimax value를 얻게 된다.
- minimax value는 이 상황에서 둘 다 완벽하게 플레이하면 누가 이길까의 의미이다.



## Minimax Search
- minimax value, minimax policy를 찾기 위한 방법
- 어떤 AI 텍스트북을 열어보면, chapter 1이 minimax search일 것이다.
    - 이전 시대에 AI에서 많이 쓰이는 방법이고, 지금도 굉장히 중요한 것을 잊지말자.
    - 요즘은 뉴럴넷이면 다 된다고 생각할 수 있겠지만, 이 search가 굉장히 힘을 준다.
    - 요즘 학생들이 이런 부분들을 등한시 여긴다.



## Minimax Search Example
- 검은 색으로 칠해져 있지만, 백부터 시작한다.
- 실제로는 더 많은 action이 있겠지만, 간단하게 그림으로 나타내기 위해, 백이 action이 2개밖에 없는 게임이라고 할 때, 백이 어떤 action을 하면, 다음으로 흑이 어떤 action을 하고, 또 다시 백이 어떤 action을 했더니 게임이 끝나고 value가 나왔다.
- value의 값이 다 다른데, 가장 왼쪽 하단의 검은 원(백)은 a_1과 a_2중 어떤 행동을 선택해야 할까?
    - a_1: 왜냐하면 백은 maximize하는 것이 목표이기 때문이다.



## Minimax Search Example
- 하단 검은색 원들은 백이 maximize하는 value 값들로 채워지게 된다.
- 백이 a_1을 선택해서 얻은 값이 해당 state의 minimax value가 된다.
- 마찬가지로, 다른 하단 검은색 원들 중에서 백 입장에서 두 값중에 더 큰 값을 고르면 된다. 그러면 해당 state의 minimax value가 계산된다.
- 반대로 가장 상단 흰색 원(흑 입장)에서는 b_1과 b_2 중 어떤 행동을 선택해야 할까?
    - b_2: 왜냐하면 흑은 minimize하는 것이 목표이다. -로 갈 수록 흑이 이기기 때문이다.


## Minimax Search Example
- 상단 흰색 원들은 흑이 minimize하는 value 값들로 채워지게 된다.
- 가장 최상단의 검은색 원(백 입장)에서는 이 두개의 value 값들 중에 maximize하는 값을 골라야 하므로, -2가 맨 위로 올라오게 된다.



## Minimax Search Example
- 현재 state의 minimax value는 minimax search를 해봤더니 -2라는 값을 알게 되었다.
    - 이 게임은 현재 state에서부터 완벽하게 두었을 때, 무조건 흑이 이기도록 설계된 게임이고
    - 첫 수에서부터 세 수만에 끝나는 굉장히 간단한 게임이다.
- 실제로 이 tree가 말도 안돼게 클 것이다.
- 바둑도 똑같이 나타낼 수 있다.



## Value Function in Minimax Search
- tree가 exponential하게 커진다.
    - 할 수 있는 action이 2개만 되도 2의 3승개의 leaf가 생긴다.
    - 하지만 보통 classic 게임에서는 action은 훨씬 많을 것이다.
- 그래서 이것을 게임 끝까지 다 해보는 것은 불가능하다.
- 대신 value function approximator을 사용해서(ex. neural net), tree 끝 까지 가보는 것이 아니라 어딘가에서 잘라서(ex. 3수 정도까지만 가본 다음에), 거기서부터 value function의 output을 그냥 그 자리에 써 놓는다.
    - leaf까지 가면, 거기서 더 수를 둘 수 있지만 거기서 자르고, 그 동그라미에 value function을 써 놓는다. 
        - truncated: 잘라내다.
    - 예컨데, 현재 상태부터 번갈아 가면서 세 수까지만 더 보고, 세 수만 두면 게임이 안끝날 텐데 그 자리에 value function값을 넣어주자. 그렇게 하면 똑같이 minimax search를 할 수 있을 것이다. 그래서 현재 상태의 minimax value를 계산할 수 있다.
- 이 minimax를 효과적으로 하기 위한 연구가 수십년 동안 이어져 왔고, 그 중 가장 간단한 방법을 설명하고 있다.
    - 대표적으로 alpha-beta search가 있다. 하지만 강의에서 자세히 다루지는 않는다.
    - alpha-beta search: tree에서 재미없는 부분을 잘라내고, 재미있는 부분만 search를 많이하는 방식



## Binary-Linear Value Function
- 결국에는 minimax search를 쓰는데, value function을 함께 쓰는 것이다.
- 그럼 value function을 어떻게 쓸까? ---> 제일 간단한 binary-linear value function
    - 먼저 feature vector을 만들고, feature vector에 weight를 곱한게 그 state의 value이다.
    - 이 weight를 어떻게 설정할까?
        - ex. 체스: 왼쪽 그림에서의 value function을 몇일까?
            - 캐슬은 +5점짜리 기물이니까 5점을 더해주고, 비숍은 +3점짜리 기물이니까 3점을 더해준다. 폰은 +1점짜리 기물이니까 한개도 없으니까 0이다.
            - 반대로 상대방에 캐슬이 있으면 -5를 더해준다.
            - 이 binary featur vector는 기물이 있냐 없냐로 어떤 feature vector를 만들고, 각 기물의 가중치 weight을 곱해서 이 두 vector의 weight product를 value function으로 사용한다.
- value function approximation이다. 실제 value일지는 모르겠지만, 현재 state에서 이정도 점수라고 approximation하는 것이다.
    - 잘 보면, 이 feature는 사람이 만든 feature이다. 즉, 게임을 수만번 수행해서 계산된 우위가 아닌, 이 domain 전문가들이 지식을 담아 만든 feature이다.
    - 지금은 기물의 있는지 여부에 대해서만 이용했지만, 기물의 combination으로도 점수를 줄 수 있는 것이고, 기물의 어떤 특정한 위치에 대해 점수를 줄 수 있을 것이다. 이렇게 이런 feature들은 계속 만들어 낼 수 있을 것이다.



## Deep Blue
- Knowledge
    - 8천 개의 feature
    - binary-linear value function
    - weights는 각 상황들의 점수를 전문가들의 지식을 통해 토의해서 만들어내서 정했다.
        - AI라고 할 수 있다. 왜냐하면 이것을 만드는 이유는 결국에 search를 하기 위해서이기 때문이다.
- Search
    - alpha-beta search: minimax search의 한 종류
        - 트리에서 depth를 16 ~ 40수까지 수행하고, 그 자리에서 잘라서 사람의 knowledge가 들어간 value function을 그 자리에 배치해서, 아래부터 위로 계산을 해서 현재 state의 minimax value를 계산한다.
- Results
    - Garry Kasparov라는 챔피언을 97년도에 4-2로 이겼다.
    - 인터넷 역사상 가장 많이 시청된 행사라고 한다.
- 보다시피 RL은 쓰이지 않았다. 이 value function이 RL로 학습된 것이 아니라, 사람이 만들었기 때문이다.



## Chinook
- 비슷한 방식으로 한 체커 프로그램
- Knowledge
    - 21개의 지식 기반의 feature
- Search
    - alpha-beta search 사용
    - Retrograde analysis
        - retrograde: 시간에 역행하는
        - 게임이 이기는 것으로 끝나는 것부터 backward로 search를 수행
        - 이 위치들을 lookup table에 저장해서, 최종적으로 n개의 나오면 그때부터 완벽하게 play를 한다.
            - n: 대충 11개의 체커
- Results
    - Marion Tinsley라는 챔피언을 94년도에 이겼다.
        - Marion Tinsley는 체커 게임에서 인생동안 7판밖에 안진, 게임 역사상 최고의 선수였고, 그 중 2판이 chinook에서 졌다.
    - chinook을 만든 사람이 계속 연구해서 2007년에는 perfect하게 play를 하도록 체커가 풀렸다.
        - 체커가 10^24정도 되는 크기의 게임인데, 2007년에는 computation이 많이 발전하면서 그냥 게임은 끝까지 search를 다 해서, 신을 대항해서도 완벽한 플레이를 하는 수준에 도달했다.



## Self-Play Temporal-Difference Learning
- 지금까지 사용한 RL 방법론을 그대로 쓰는 것인데, self-play로 경기를 만들어 내서 경기들에 대해서 value-based RL 알고리즘을 쓰는 것이다.
    - MC: 실제 value와 내가 예측하는 value 사이의 차이를 제곱한 값을 minimise한 것
        - Mean squared error를 미분하면, 제곱이 앞으로 나오고, 차이나 남고, gradient가 생긴다.
    - TD(0), TD(\lambda): 실제 value에 R + \gamma를 넣는 것
- 2player 게임은 대부분 intermediate reward가 없다. 게임 도중에는 reward가 발생하지 않고, 게임이 끝난 뒤에 이겼는지 졌는지만 나오므로 0, 0, 0, ..., 1 또는 0, 0, 0, ..., -1이 나온다. 그래서 \gamma를 빼고, R도 모두 0이어서 R도 빼고 식을 사용하였다.
    - 즉 다음 state의 value로 현재 state의 value를 업데이트 한다.
    - 여기서의 value는 모두 minimax value function의 approximatior이다.
- 차이점 정리
    - 양쪽에서 플레이하므로 minimax value function 사용
    - intermediate reward가 없으므로, R과 \gamma가 없어졌다.


## Policy Improvement with Afterstates
- afterstate: 게임의 룰을 완전히 안다고 가정하면, 내가 어떤 action을 해야 그 state로 가는지 알게 된다. 즉, deterministic game에 대한 경우, 현재 state에서 갈 수 있는 다음 state가 무엇인지를 알수 있다.
    - ex. 바둑: 어디에 두느냐에 따라 어떤 state로 우연히 가고 안가고가 없다. 어떤 수를 두면 정확히 해당 state로 가게 된다.
    - ex. 체스: 현재 state에서 갈 수 있는 state를 안다. 어느 말을 어디로 움직이느냐에 따라 state를 바꿀 수 있다.
    - 그래서 Q가 아니라 V만 알면 policy improvement를 할 수 있다.
    - 예전에 model-free와 model-based를 배울 때, model을 모를 때는 내가 어떤 action을 해야 그 state를 가는지 모르기 때문에, V만 갖고는 action을 할 수가 없었다.
        - model-free: Q는 action value인데, 예전에는 V만 갖고는 어떤 state를 갈 수 있는지 몰라서, action value만으로 학습을 해왔다.
- 이제는 어떤 state에서 V가 몇인지만 알면, 다른 state로 가는 법은 내가 아니까, 내가 현재 있는 state에서 다음 state가 뭐가 있는지 다 알고, 다음 state가 10개 있으면 백의 입장에서 그 중 state가 제일 높은 것을, 흑의 입장에서 가장 낮은 것을 고르기만 하면 되니까, V만 학습하면 된다.
- s에서 a를 하면 그다음 state가 무엇인지를 게임의 규칙이 알려준다. 그래서 action은 백의 입장에서 다음 state가 최대가 되는 action을 선택하고, 흑의 입장에서 최소가 되는 action을 선택하면 된다.
- 이렇게 하면, policy iteration에서 V만 가지고 policy improvement를 할 수 있다.
- 즉, self-play RL에서는 V만 갖고 학습을 하겠다는 뜻이다.



## Self-Play Temporal-Difference Learning
- V를 어떻게 계산할까?
    - MC나 TD를 사용해라.
- V만 갖고 action을 할 수 있다.



## Self-Play in Othello: Logistello
- Logistello라는 프로그램이 오델로 게임을 잘 했다.
- 돌들의 position의 조합으로 150만개의 feature를 만들었다.
    - 오른쪽 그림의 2행을 보면, 대각선의 돌의 갯수와 위치에 따라 갖가지 모든 combination을 만들어서 그것에 대해서 binary linear function을 만들었다.
    - 이 150만개는 게임에서 자주 발생하는 돌의 조합이다.
- handcrafted feature들과는 다르다. handcrafted feature는 이렇게 있으면 몇점이다를 정해주는 것인데, 이 프로그렘은 여러 다양한 상황들이 있냐 없냐로 만들어준 것이기 때문에, handcrafted와는 다르다.



## Reinforcement Learning in Logistello
- logistello는 generalised policy iteration을 이용했다.
    - policy iteration: policy evaluation을 하고, 평가한 것을 바탕으로 greedy(또는 e-greedy)하게 움직여서 새로운 policy를 만든다. 그리고 이 policy를 통해 다시 evaluation을 하고 이것을 반복한다.
    - 현재 policy로 self-play를 해서 경험을 만들고, 그 데이터를 바탕으로 Monte-Carlo 방법을 사용해서 현재 policy를 평가한다.
        - Monte-carlo 방법을 사용한다는 것은 outcome을 regression한다는 것과 같은 말이다. outcome은 승패 여부를 나타내고, regression은 supervised learning처럼 숫자 맞추기를 의미한다. 이 state에서 이겼으면 1, 졌으면 -1이면, input은 해당 state, output은 이겼나 졌나를 regression 관계를 학습하는 것이다.
    - 이렇게 policy를 통해 평가를 하고 난 뒤, 이 policy를 바탕으로 greedy policy improvement를 해서 새로운 player를 만든다. greedy하게 움직일 때, 백의 입장에서는 maximise를 하고, 흑의 입장에서 minimise한다.(minimax)
- 이런 식으로 반복해서 학습을 했고, 그 당시 세계 챔피언인 Takeshi Murukami를 6:0으로 이겼다.
- 참고로 logistello는 superhuman level을 달성한 첫 프로그램이라고 한다.



## TD Gammon: Non-Linear Value Function Approximation
- 백가몬이라는 보드게임이 있고, 그것을 푸는 IBM에서 만든 TD Gammon이라는 프로그램이 있다.
    - 강의 당시 가장 유명한 AI 였다.
    - Nerual net을 사용했지만, 이때는 computation이 어려워서 nerual net이 깊지는 않았고, feature는 raw한 feature을 사용했다. 그리고 노드 수가 80/40 정도였다.



## Self-Play TD in Backgammon: TD-Gammon
- neural net을 random weights로 초기화하고, self-play로 학습을 했는데, 학습에 TD learning을 사용했다.
    - 슬라이드에 나와있는 식은 TD(0)인데, 실제로는 TD(\lambda)를 사용했다.
    - R, \gamma는 사라졌다.
- exploration을 전혀 사용하지 않고, greedy policy improvement를 했는데 항상 수렴했다.
    - 왜 그럴 수 있었을까?
        - 백가몬에는 주사위가 있다. 주사위를 던지면서 플레이를 하는데, 이 주사위에 stochasticity, 즉 확률성이 있어서, 계속 다른 state를 가게 해줘서 exploration 효과가 있다. environment 자체안에 inherent stochasticity(내재된 확률성)가 있어서 굳이 exploration을 안해도 계속 다양한 상황을 맞닥들이면서 학습이 되었다.
        - 일반적으로는 성립되는 상황은 아니다.



## TD Gammon: Result
- zero expert knowledge ===> strong intermediate play
    - raw한 position만 사용했고, 시간에 따라서 계속 연구가 있었다.
 - 91년도: 처음 hand-crafted features를 썼을 때, advanced level에 도달했다.
 - 98년도: 2-play search, 3-play search: minimax search의 depth를 설정해주었고, 하나 더 할수록 점점 더 strong master, superhuman 수준에 도달했다.
 - 92년도: 세계 챔피언을 이겼다.



 ## New TD-Gammon Result
 - TD-gammon 연구하던 분이 나중에 nerual net이 발달하고, 이 hidden unit 갯수를 늘려가면서 연구를 다시 했다.
 - 그래프상 가장 위에 있는 것이 80 hidden units이다.
    - 위로 올라갈 수록 성능이 좋은 것이다.
    - hidden unit이 많을 수록 성능이 좋았다.
    - hidden unit: neural net에서 node의 갯수
        - hidden unit이 많을 수록 node의 갯수가 많아서 더 복잡한 네트워크라고 생각하면 된다.



## Simple TD
- RL과 minimax search를 결합한 것을 배우기 위해 3가지 방법론을 배운다. 일단 배우기 전에 예전에 배웠던 TD(simple TD)를 생각해보자.
- TD: 현재 value를 successor value로 업데이트한다.
    - successor value: 다음 다음 state value
    - v(s_t)에 v(s_{t+1})을 넣는다.
- 다음 state의 value가 현재 state의 value로 온다. 하나 더 보고 어떤지를 넣는 것이 TD이다.
    - R, \gamma는 빠진 상태이다.
- 지금까지는 value function을 TD learning으로 먼저 학습한 다음에, 그 value function을 이용해서 minimax search에 썼다. 예를 들면, minimax search에서 depth를 3까지만 가보고, 3번째에서는 잘라버리고 그 자리에 value function 값을 넣었었다.
    - MCTS를 사용하는 alpha go 1과 비슷하다.
    - logistello와 TD-gammon 모두 이 두 단계 방식이었다. 즉 minimax search의 결과가 value function을 낫게 만드는데는 전혀 쓰이지 않았다. 왜냐하면 learning과 search가 분리되었기 때문이다.
        - 그런데 그럴 필요가 있을까? search의 결과가 learning에 영향을 주어서 value가 더 정확해지고, value가 다시 search에 또 좋은 영향을 줄 수 있지 않을까?



## Simple TD: Results
- 오델로, 백가몬에서는 잘 됐는데, 체스나 체커에서는 잘 안됐다.
- search를 아예 학습과정에 포함시켜서 value function을 더 낫게 만드는데 쓸수 없을까?
    - TD Root
    - TD Leaf
    - TreeStrap



## TD Root
- 현재 state를 s_t라고 하고, 다음 state를 s_{t+1}이라고 할 때, s_{t+1}에서 minimax search를 하자. 이 때, 오른쪽 트리와 같이 minimax value가 초록색 원이 되고(흑 입장에서 minimise, 백 입장에서 maximise 해서 도달한 노드), s_t를 이 초록색 값이 되도록 value를 업데이트를 한다.
    - 게임의 규칙을 다 알고 있기 때문에, value function만 알아도 action을 할 수 있는 상황이기 때문에 policy는 다루지 않고 있다.
- 따라서 s_{t+1}에서 search를 하고, search한 value로 s_t의 value를 업데이트 한다.
    - learning과 search를 combine한 것이다.
- 원래는 V(s_{t+1})이 V(s_t)로 왔는데, V(S_{t+1})보다 s_{t+1}에서 search를 돌린 value가 더 정확하니까 이 값을 V(S_t)에 넣자.
- V_+(S_t): S_t에서 minimax를 한 결과
    - 왼쪽 트리의 파란색 값
- V_+(S_{t+1}): S_{t+1}에서 minimax 한 결과
    - 오른쪽 트리의 초록색 값
    - +는 minimax search를 돌린 값
- 따라서 s_t 자리에 s_{t+1}에서 search를 돌린 값을 넣자는 의미이다.
    - I_+(s_{t+1}): s_{t+1}에서 search를 돌린 노드를 가리킨다. 즉 어떤 state를 가리킨다.
    - v(I_+(s_{t+1}), w): s_{t+1}에서 search를 돌렸을 때의 state의 w의 value 값을 나타낸다. 즉 이 state를 value의 input으로 넣었을 때 어떤 값이 나올 것이다.
- 그리고 s_t의 value가 s_{t+1}에서 search를 돌린 값으로 업데이트 한다.
- TD 업데이트를 할 때, 원래 다음 state의 value로 업데이트 했었는데, 여기서는 다음 state의 minimax search를 돌려서 V_+(S_{t+1})을 구하고, 그 값으로 V(S_t)를 업데이트한다.
    - value function을 업데이트할 때, search의 결과를 사용했다. 지금까지는 value function은 TD learning으로 구하고, 그게 끝난 다음에 2단계로 minimax search를 수행했는데, 지금은 value function을 학습하는 도중에 search를 사용한 것이다.



## TD Root in Checkers: Samuel's Player
- TD root는 First ever TD learning algorithm이라고도 한다.
- 체커 프로그램에 쓰였다.



## TD Leaf
- s_t에서 V(s_t)를 업데이트 하는 것이 아니라, 현재 state에서 search한 값을 다음 state에서 search한 값으로 덮어 쓴다.
    - 왜냐하면 한 step 더 가서 search 한 것이 더 정확할 것이기 때문
- V_+(s_t): s_t에서 search한 값
- V_+(s_{t+1}): s_{t+1}에서 search한 값
- V_+(s_t)에서 V_+(s_{t+1}) 방향으로 조금 더 업데이트 한다는 의미이다.
    - 현재 값을 바꿔주는 것이 아니라, 현재의 search 결과를 다음 step에서의 search 결과로 바꿔준다.



## TD leaf in Chess: Knightcap
- TD leaf가 체스에 쓰인 프로그램이 Knightcap이다.
- self-play에서는 그렇게 효과적이지 않았다.



## TD leaf in Checkers: Chinook
- original chinook은 hand-tuned weights를 사용했다.
- 나중 버전의 chinook은 self-play를 썼고, self-play로 학습된 weight가 hand-tuned weight보다 더 잘했다.



## TreeStrap
- TD root와 TD leaf는 real step과 imaginary step을 섞는 것이다.
    - real step: 한 걸음 실제로 더 보는 것
    - imaginary step: 거기서 search를 만들고, 내 자리에서 search를 만드는 것
    - 그런데 real step과 imaginary step을 굳이 섞을 이유가 있을까?
        - 그냥 모든 것을 상상속에서 해버리자.
- 현재 자리에서 search를 돌려서 결과가 나오면, 현재 자리 아래 것으로 모든 노드들을 업데이트 한다.
    - 맨 아래 leaf 노드는 바로 위 부모 노드들로 업데이트가 된다.
    - 같은 단계에 있는 모든 노드들이 각각 자기 아래에 있는 노드들로부터 업데이트 된다.
- 실제 step을 밟지 않고, 상상속에도 모두 이루어지는 일이다.



## Treestrap in Chess: Meep
- 체스에서 treestrap이 쓰인 것으로 Meep이라는 프로그램이 있었다.



## Simulation-Based Search
- 이것도, RL과 search가 합쳐진 것
- 먼저 root state에서부터 시작해서 self-play로 게임을 시뮬레이션 한다. 이 시뮬레이션으로 얻은 경험으로 RL을 적용한다.(8강)
    - 즉 시뮬레이션한 경험에 model-free RL 방법론을 사용한 것이다.
    - 이때, Monte-carlo를 사용한 것이 Monte-Carlo Tree Search 이다.
        - MCTS는 현재 존재하는 것 중 최고로 좋은 것이라고 한다.
        - simulated 된 experience에서 각 value가 무엇이 될 지 실제 output으로 학습하는 것이다.
    - search안에 Monte-carlo control을 넣는 것이다. 각 action들의 value를 simulated experience로부터 학습을 하고, 그 중 좋을 것 같은 action value를 선택한다. 그렇게 되면 exploitation이 되므로, exploration을 섞어주기 위해 (모든 노드를 bandit으로 보고) UCB 알고리즘을 사용한다.
    - exploration과 exploitation을 밸런스있게 수행하기 위해 UCB를 적용한다. 이렇게 사용하면 MCTS의 한 응용이 된다.
- MCTS는 굉장히 효과적이다.



## Performance of MCTS in Games
- Go, Hex, Lines of Action 등 challenging game에서 가장 좋은 성능을 보여주는 방법론이다.



## Smooth UCT Search
- imperfect information일 때, smooth UCT를 사용한다.
- average strategy라는 것을 만들어서 지금까지 배웠던 상대들의 average behavior를 배운다. 즉 평균적으로 어떤 행동을 하는 것
    - 이를 이용해서 UCT를 수정하는데, 이를 smooth UCT라고 한다.
- 포커에서 많이 사용되었다.



## RL in Games: A Successful Recipe
- 각 게임에서 어떤 재료들이 사용되었는지 정리한 슬라이드
