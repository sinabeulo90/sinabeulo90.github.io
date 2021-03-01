---
layout: post
title:  "Model Free Prediction(2)"
date:   2021-03-01 00:00:00 +0900
category: "Reinforcement Learning"
tags: YouTube
---

## 참고

- video
    - [[강화학습 4강] Model Free Prediction](https://youtu.be/47FyZtBRglI)
- slide
	-[Lecture 4: Model-Free Prediction](https://www.davidsilver.uk/wp-content/uploads/2020/03/MC-TD.pdf)


## Temporal-Difference Learning
- 경험으로부터 직접적으로 배운다.
- Model-free, MDP에 대한 지식이 필요 없다.
- TD, MC와 차이점
	- TD: 에피소드가 끝나지 않아도 배울 수 있다.
	- MC: 에피소드가 끝나야지 배울 수 있다.
- Guess로 Guess를 업데이트 한다.


## MC and TD
- MC: G_t 방향으로 업데이트 한다
- TD: R_{t+1} + \gamma V(S_{t+1}) 방향으로 업데이트 한다.
	- TD Target: R_{t+1} + \gamma V(S_{t+1})
		- 현재 state s_t에서 앞으로 Reward를 얼마나 받을지 예측 하는 것을 V(s_t)
		- 한 step 이후 s_{t+1}에서 예측하는 Value
		- 그러면 step 이후에서 예측하는 예측치가 더 정확하니까, 그 방향으로 V를 업데이트 하는 것
	- Ex: 차를 운전하고 있는데, 실수로 중앙선을 넘었다고 하자. 그런데 건너편에서 트럭이 달려오고 있을 때, 트럭의 반응속도가 좋아서 비켜 지나가고 나는 다시 원래 차선으로 돌아온 상황이라고 하자.
		- MC의 경우: 나는 충돌하지 않앗으니까, 그 상황에 대한 - Reward(-1억)를 못받는다
		- TD의 경우: 중앙선을 침범하는 순간, 다음 step의 상태에서 내가 죽을 것 같다는 예측을 할 수 있기 떄문에, 다시 중앙선으로 돌아왔을 때 그 상황에 대해 추측을 사용해서 update를 한다.
			- 한 step만큼 가서 그 step에서 추측을 하는 것: V(s_{t+1})
			- 그 과정에서 받은 Reward: R_{t+1}
			- step만큼 가서 예측을 한 것: R_{t+1} + \gamma V(S_{t+1})
				- 이전의 예측치보다 정확할 것이다. 왜냐하면 한 step만큼 현심이 더 반영된 것이기 떄문이다. 이 값으로 현재 V(s_t)를 업데이트 한다.
- TD 의미: 순간적인 시간 차이라는 의미
- TD error: R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
- 한 스텝 더 간 예측치로 현재의 예측치를 업데이트 한다.
- MC: 실제 실현된 정확한 값으로 업데이트 한다.


## Driving Home Example
- TD 예시


## Driving Home Example: MC vs. TD
- 도착했을 때 총 43분이 걸렸다면,
	- MC: 지나온 각 state에서 43분을 기준으로 업데이트
		- 즉 도착해야 알 수 있는 값이 43분이다.
	- TD: 매 스텝으로 걸린 시간 + 다음 상태에서의 예측치를 기준으로 업데이트


## Advantages and Disadvantages of MC vs. TD
- TD: final outcome을 알기 전에 학습 할 수 있다.
	- continuing 한 환경, non-terminating 한 환경에서 학습 가능, 계속되는 환경
- MC: 에피소드가 끝날 때 까지 기다렸다가, 끝난 다음에 return을 알게 되면, 그 return을 기준으로 업데이트한다.
	- terminating한 환경에서 학습을 할 수 있다.


## Bias/Variance Trade-Off
- Return G_t는 편향되지 않은 V_\pi(S_t)의 측정 값
	- G_t의 기댓값 = V_\pi(S_t), G_t를 계속 샘플링하면, 그 평균(기댓값)이 V_\pi(S_t)로 수렴한다.
- True TD Target: 전지전능한 신이 V_\pi(S_{t+1})의 실제 값을 알려주면, R_{t+1} + \gamma V_\pi(S_{t+1})은 편향되지 않는 값이다.
	- 왜냐하면 Bellan Equation이 이를 보장하기 떄문
- 그런데 우리는 V_\pi(S_{t+1})의 실제 값을 모르기 떄문에, TD target R_{t+1} + \gamma V_\pi(S_{t+1})은 편향되어 있는 값이다.
	- 즉 백만번 한다고 해도 V_\pi(S_t)가 되리라는 보장이 없다.
	- bias 관점에서는 TD는 않좋지만, variance 관점에서는 TD target은 return보다 훨씬 낮다.
	- variance: 통계에서의 분산, 어떤 확률 변수에서 평균으로부터 얼마나 퍼져있는지를 나타내는 척도
		- ex. 평균이 0인 Gaussian 분포: 넓게 퍼져있는 variance가 큰 것도 있고, 작은 것도 있을 텐데, Variance가 큰 가우시안 분포를 생각해보면 그 가우시안 분포로부터 샘플링하면 매우 튀는 값이 나온다. 그러면 우리는 0을 학습하고 싶은데, 샘플은 1000, -1300 이런 식으로 매우 흔들리는 값이 나올 것이다. 하지만, 이는 unbiased estimate이다. 즉 1억번 샘플링 한 뒤에 평균을 내면 0이기 때문이다.
			- variance가 크고, bias가 없는 추정 량: MC
			- MC는 답이 0인데, -150을 기준으로 업데이트가 될 것이다.
		- ex. sigma = 1, 평균이 0.5인 가우시안 분포: 0.5 주변에 모여있는 어떤 종 모양을 상상하고, 샘플링을 하면 3, -1, 2... 이고 이것들의 평균이 2.3이라고 하자. 그러면 우리는 0을 찾아야 하는데, 2.3으로 업데이트가 된다. 대신 더 빨리 찾을 수 있을 것이다.
		- bias만 적다고 능사가 아니고, variance도 중요하다.
- return은 현재 스텝에서부터 게임 끝날 때 까지 정말 다양하게 얻게 되는데, 환경의 랜덤섬과 state transition probability가 있고, action도 policy에 따라 stochastic policy라면 랜덤성이 있을 텐데, 이 랜덤성과 랜덤성을 가지고 게임을 끝까지 하면 전혀 다른 결과를 계속 얻을 것이다. 때문에 variance가 크다.
- TD는 한 스템 사이에서의 랜덤성은 게임이 끝날 때 까지의 랜덤성보다 적기 때문에, variance가 적고, biased는 있는 추정값을 얻게 된다.


## Advantages and Disadvantages of MC vs. TD(2)
- MC: variance가 크고, bais는 0이다.
	- 수렴 성질이 좋고, 문제가 많아져서 테이블에 적을 수도 없게 될 경우 function approximation을 활용하게 되는데(Neural net, Deep learning), 딥러넹에서도 수렴성이 좋다.
	- 실제 값으로 업데이트하기 때문에 초기 값에 민감하지 않고, 이해하기 쉽고, 사용하기 쉽다.
- TD: variance가 작고, bias가 약간 있다.
	- 보통 MC 보다 더 효과적이다.
	- TD(0)는 V_\pi(S)에 수렴한다.
	- 초기 값에 민감하다. 왜냐하면 initial value를 이용해서 V_\pi(S)를 업데이트 하기 때문
- bias가 있으면, 이것이 동작은 할 수 있는 알고리즘 일까?
	- 왜냐하면 결국 틀린 값으로 수렴한다는 이야기이기 때문이지만, 다행히도 동작은 한다.


## Random Walk Example
- 0으로 초기화
- TD를 사용했을 때, 100번째에선 true value로 수렴한다.


## Random Walk: MC vs. TD
- \alpha 값에 대해서 MC, TD 비교
- 에피소드가 진행하면서, 실제값과 추정값의 차이를 RMS 지표로 구한 것
- MC는 완만히 줄어들고, TD는 줄어들다가 다시 늘어나기도 한다.
	- TD가 다시 늘어나는 것은 \alpha가 너무 크기 때문에 진동한다.


## Batch MC and TD
- 에피소드를 무한번 샘플링할 수 잇으면, MC, TD가 수렴한다.
- 그런데, 만약 k개의 제한된 에피소드만 있다면, 그 때 MC와 TD가 같은 곳으로 수렴할까?


## AB Example
- A, B state가 있고, 8개의 에피소드가 있을 경우
- Model free이기 떄문에 왜 이렇게 되는지 규칙을 알 수 없다.
- 단지 경험들만 주어지는데, 이전에 배운 것 처럼 경험으로 MC, TD를 할 수 있다.
- V(B) = 3/4 = 0.75
- V(A) = 0?


## AB Example
- 주어진 경험으로 부터 MDP를 추측할 수 있다.
- MC를 이용해서 V(A)를 학습하면, V(A) = 0
	- A를 한번 방문했고, 그때 받은 reward는 0
	- 따라서 0의 평균은 0
- TD를 이용해서 V(A)를 학습하면, V(A) = 0.75
	- A에서 reward를 받고, V(B)의 value로 A를 업데이트 하기 때문


## Advantages and Disadvantages of MC vs. TD(3)
- TD: Markov property는 사용해서 Value를 추측
	- Markov environment 에서는 더 효과적
	- Markov model의 likelihood를 maximize 한다.
- MD: Markov property를 모두 버리고, MSE를 minimize 한다.


## Monte-Carlo Backup
- backup: 한 state S_t에서 시작해서 한 에피소드가 끝날 때 까지 진행해서 그 G_t로 V(S_t)를 업데이트 하는 것


## Temporal-Difference Backup
- 한 스탭만 가보고, 그 state에서 추측을 해서 G_t를 대체한다. 이 것을 Bootstraping이라고 한다.
- Bootstraping: 다른 분야에서도 쓰이는 용어이다. Random forest 등
	- 끝까지 안해보고 추측치에서 추측을 하는것을 말한다.


## Dynamic Programming Backup
- DP: 샘플링을 하지 않고, 모든 칸에 대해서 한 step을 가보고, 그 칸에 적혀있는 value를 기준으로 업데이트 한다. 끝까지 가지 않고, full sweep 한다.


## Bootstrapping and Sampling
- Bootstrapping: 추측치로부터 추측치를 업데이트 하므로, 업데이트의 추측치가 포함됨
	- MC x: 에피소드가 끝날 때 까지 가보고 업데이트 하기 때문
	- DP, TD o: 깊이 관점에서, 한 스텝만 가고 멈추기 때문에 부트스트랩을 한다.
- Sampling: Full sweep하지 않고, 하나의 대해서 샘플을 가지고 업데이트
	- MC, TD o: 너비 관점에서, 해본 샘플들을 가지고 업데이트 하기 떄문
		- MC가 끝까지 가던, TD는 한 스텝만 하던, 했던 것을 가지고 업데이트 하기 떄문
	- DP x: 한 state에서 가능한 모든 action을 취하기 때문에, 샘플링하지 않는다.
	- agent가 policy를 가지고 움직이는 것


## Unified View of Reinforcement Learning
- Bootstrapping을 하는가, Sampling을 하는가
- MC: Bootstrapping(deep backups) X, Sampling(sample backup) O, 
- TD: Bootstrapping(shallow backups) O, Sampling(sample backup) O
- DP: Bootstrapping(shallow backups) O, Sampling(full backup) X
- Exhaustive search: 모든 것을 다 해볼 수 있는 트리를 만들어서 찾는 방법, 학습 방법이라고 하기에도 좀 그렇다. (비용이 너무 많이 듬)
- 모델을 안다면 full backup이 가능하기 때문에, TD, MC를 해야 한다.


## n-Step Prediction
- TD의 변형 들
- 한 스텝만 가서 업데이트 할 수 있고, 현실을 2스텝만큼 반영한 다음 부트스트래핑을 할 수 잇다.


## n-Step Return
- n-1 만큼은 reward를 넣고, S_{t+n} 부터는 추측치 V_{S_{t+1}}}를 넣는다.
- n-step TD return: G_t^{(n)}
- n-step TD error: G_t^{(n)} - V(S_t)


## Large Random Walk Example
- TD(1) ~ TD(1000) 까지 성능 비교
- Y축: RMS error, X축: \alpha
- 그래프가 아래로 갈 수록 좋을 텐데, 1이 제일 좋지도 않고, 1000이 제일 좋지도 않다.
- 3 or 5 step TD가 좋은 것을 확인 할 수 있는데, TD(0)와 MC 사이의 sweet spot(달콤한 구간)이 존재한다. 적당히 잘 정해줘야 한다.
- 에피소드가 10개로 제한되어 있기 때문에, MC return의 variance 크기 떄문에 성능이 좋지 않다.


## Averaging n-Step Returns
- 2 step, 4 step이 있다면, 그 둘의 평균을 정답으로 사용할 수도 있다.


## \lambda-return
- TD(\lambda)심지어는 TD(0) ~ MC까지 모든 것을 평균내서 정답으로 사용해도 된다.
- 논문에서도 꽤 쓰이는, 현실에서도 전혀 동떨어져 있는 얘기가 아니다.
- geometric mean을 사용한다.
	- (1-\lambda)\lambda, (1-\lambda)\lambda^2, (1-\lambda)\lambda^3, ...
	- 위의 모든 (1-\lambda)\lambda^{n-1}의 무한 급수 결과가 1이기 때문
	- 앞으로 갈 수록 가중치가 적게 들어간다.
	- Forward-view TD(\lambda): 미래를 관측하기 때문
	- Backward-view TD(\lambda)도 있다.


## TD(\lambda) Weighted Function
- 시간 t가 갈수록 가중치가 점점 줄어든다. geometric하게
- geometric mean을 쓰는 이유: 메모리를 확장시키지 않으면서 계산이 편하기 때문이다.
- TD(0)와 같은 비용으로 TD(\lambda)를 계산 할 수 있다.


## Forward-view TD(\lambda)
- 미래를 보고 업데이트를 한다.
- 단점: MC와 마찬가지로 게임이 다 끝난 뒤에 업데이트 할 수 있다. 왜냐하면 TD(\infinity)은 게임이 끝나야 알 수 있기 때문
- TD(0)의 장점이 사라진다.


## Forward-View TD(\lambda) on Large Random Walk
- \lambda값에 따라 결과가 다르다.


## Backward View TD(\lambda)
- 매 스텝마다 업데이트 할 수 있고, 에피소드가 끝나지 않아도 업데이트 할 수 있다.


## Eligibility Traces
- Eligibility: 적격, 적임
- 어느 사건이 일어나면 책임을 물으면서, 책임이 큰 state를 업데이트를 많이 하는 방식이다
- ex. 종 / 종 / 종 / 전구 / 전기 충격
	- 전기 충격은 가장 최근에 일어난 전구한테 책임을 물을 수도 있고, 가장 빈뻔하게 발생한 종에게도 책임을 물을 수 있을 것이다.
	- 책임을 나누는 기준
		- Frequency heuristic: 빈번하게 일어난 state에게 책임을 많이 물어야 한다.
		- Recency heuristic: 최근에 일어난 state에 책임을 많이 물어야 한다.
	- 위 2가지 기준을 사용해서 Eligibility trace가 나오게 된다.
- 어떤 state 별로 각각 Eligibility trace 값을 하나씩 갖고 있는데, 각 시점별로 해당 state를 방문하면 1을 올려주고, 방문하지 않을 때 마다 \gamma(ex. 0.9) 만큼 줄인다.
	- 방문할 때 1을 증가: Frequency heuristic
	- 방문하지 않을 때, 줄인다: Recency heuristic
	- 방문할 때마다 커지고, 최근에 방문했을수록 커진다.


## Backward View TD(\lambda)
- TD(0)의 장점과 TD(\lambda)의 장점을 가진다.
- E_t(s): 어느 시점 t일 때, state s에 대한 Eligibility Traces 값
- \delta_t: TD(0) error 값을 그대로 쓴다.
- 해당 state에 대한 책임 소재가 얼마나 큰지 기록해 두는 follow up하는 값이 있어서 그 값을 곱해서, 그 만큼 업데이트 해준다.
	- 이렇게 해주면, TD(\lambda)와 수학적으로 동일한 의미를 가진다.
- 장점: 매 과거의 책임소재를 기록해 오기 때문에, 매 step마다 업데이트를 할 수 있다.
- 따라서 TD(\lambda)를 쓸때는 backward view TD(\lambda)를 쓴다.


## TD(\lambda) and TD(0)
- Eligibility Traces 식에서 \lambda 값을 0으로 하면 TD(0)와 동등하다.


## TD(\lambda) and MC
- Eligibility Traces 식에서 \lambda 값을 1로 하면 MC와 동등하다.
- offine 업데이트를 하면, TD(\lambda)는 Forward-View TD(\lambda)와 Backward View TD(\lambda) 같다.
- 최근에는 online 업데이트에서도 같다는 연구결과도 있다고 한다.
- oneline update: agent가 학습하면서 환경에서 움직이는 것
	- 한 스텝이 끝나면, 한 스텝만큼 경험으로 학습을 해서 움직인다.
- offline update: 에피소드가 끝나면 업데이트하고 학습한 뒤 움직인다.
