---
layout: post
title:  "Q-Learning"
date:   2021-02-23 01:00:00 +0900
category: "Grepp/KDT"
tag: "Reinforcement Learning"
plugins: mathjax
---

## 강화학습
- Agent는 현재의 state를 통해 action을 선택 ---> Environment가 변화
- Agent는 environment에서 얻은 정보를 이용하여 학습

### Q-Learning
- 바로 다음 reward만을 통해서 학습하는 것은 좋지 않다.
- 미래에 받을 reward까지 고려할 필요가 있음
- 특정 state(s)에서 각 action(a) ---> 미래의 reward의 합(Q)

### Update
- Supervised Learning ---> Answer
- Q-Learning ---> Target
    - Q-Value: state에서 action을 취했을 때, 앞으로 받을 reward 합의 예측 값
    - 알고 있는 정보: state, action, next state, terminal
    - 앞으로 받을 reward 합의 예측 값 = Reward(현재 state) + Q-value(다음 state)
    - 목표: 앞으로 받을 reward 합을 최대로 하는 행동 선택
        - 앞으로 받을 reward의 합: Reward(현재 state) + Q-value(다음 state, 각 action에 대해 존재)
        - 앞으로 받을 reward의 최대합: Reward(현재 state) + Max Q(다음 state)
        - Target = Target = Reward(현재 state) + Gamma * Max Q(다음 state)
            - Gamma($\gamma$): discount factor
                - 0 ~ 1 사이의 값
                - 미래의 예측 reward를 얼마나 고려할 것인가?
    - 현재 Q-value에 target 값을 일정 비율로 더해주어 update
    - Q-value = (1 - $\alpha$) * Q-value + $\alpha$ * Target
        - $\alpha$: learning rate
            - 얼마의 비율로 학습할 것인가?
- 최종 Update
    - Target = Reward(현재 state) + $\gamma$ * Max Q(다음 state)
    - Q-value = (1 - $\alpha$) * Q-value + $\alpha$ * Target


### Update 예시
- Q-table에 없는 state에서는 state의 Q-value들은 0으로 초기화하고, random action
    - $\alpha$: 0.1
    - $\gamma$: 0.9
- Max Q를 따라 action을 취하는 경우, 같은 state와 action이 계속 반복됨
- 더 좋은 reward를 얻지 못함


### Epsilon Greedy
- 학습되지 않은 Q값은 믿을 수 없음
- 그렇다면 학습이 충분히 이루어지지도 않고, 다양한 경험을 해보지도 못한 게임 초반에는 어떻게 할까?
- Random하게 action을 하게 하여 다양한 경험을 하게 하자.
- Exploration: 랜덤하게 action 결점
- Exploitation: 학습된 action 선택
- Epsilon: 랜덤하게 action을 선택할 확률
    - epsilon = 0.8: 80%확률로 random action


### Q-Learning 코드
- [GYM](https://gym.openai.com)
    - OpenAI에서 제공하는 라이브러리
    - 다양한 강화학습 환경을 포함
- Environment: Taxi-v2
    - gym에서 제공하는 환경 중 하나
    - 노란색 박스: 빈 택시 / 초록색 박스: 승객이 탑승한 택시
    - 실선: 이동 불가 / 점선: 이동 가능
    - RGYB: 승객의 위치, 목적지
    - 상태
        - 택시의 위치: 25
        - 승객의 위치: 5(RGYB + 승객이 탑승한 경우)
        - 목적지: 4(RGYB)
        - 0~500의 scalar 값으로 변경(25*5*4)
    - 행동
        - 0: 아래
        - 1: 위
        - 2: 오른쪽
        - 3: 왼쪽
        - 4: 승객 태우기
        - 5: 승객 내리기
    - 보상
        - 매 이동시 -1
        - 승객을 잘못 태우거나 내리면 -10
        - 승객을 목적지에 내려주면 +20

```python
# 라이브러리 불러오기
import numpy as np
import random
import gym

# 환경 정의
env = gym.make("Taxi-v3")   # Taxi-v2 환경을 env로 정의

# 파라미터 설정
action_size = env.action_space.n    # actino의 수: 6

discount_factor = 0.9   # 감가율
learning_rate = 0.1     # 학습률

run_step = 500000       # 학습 진행 스텝
test_step = 10000       # 학습 이후 테스트 진행 스텝

print_episode = 100     # 해당 스텝마다 한번씩 진행 상황 출력

epsilon_init = 1.0      # 초기 epsilon
epsilon_min = 0.1       # 최소 epsilon

train_mode = True       # 학습 모드



# Q-Agent Class: Q-Learning 관련 함수들을 정의
class Q_Agent():
    def __init__(self):
        self.Q_table = {}               # Q-table을 dictionary로 초기화
        self.epsilon = epsilon_init     # epsilon 값을 epsilon_init으로 초기화


    # 만약 Q-table 내에 상태 정보가 없으면 Q-table에서 해당 상태 초기화
    def init_Q_table(self, state):
        if state not in self.Q_table:
            self.Q_table[state] = np.zeros(action_size)
    

    # Epsilon greedy에 따라 행동 결정
    def get_action(self, state):
        if self.epsilon > np.random.rand():
            # 랜덤 행동 결정
            return np.random.randint(0, action_size)

        self.init_Q_table(state)

        # Q-table 기반으로 행동 결정
        return np.argmax(self.Q_table[state])
    

    # 학습 수행
    def train_model(self, state, action, reward, next_state, done):
        self.init_Q_table(state)
        self.init_Q_table(next_state)

        # 타겟값 계산 및 Q-table 업데이트
        target = reward + discount_factor*np.max(self.Q_table[next_state])
        Q_val = self.Q_table[state][action]

        if done:
            self.Q_table[state][action] = reward
        else:
            self.Q_table[state][action] = (1 - learning_rate)*Q_val + learning_rate*target

    
# Main 함수
if __name__ == "__main__":
    # Q_Agent 클래스 초기화
    agent = Q_Agent()

    # 스텝, 에피소드, 보상을 저장할 리스트 초기화
    step = 0
    episode = 0
    reward_list = []

    # 게임 진행 반복문
    while step < run_step + test_step:
        # 상태, 에피소드 동안의 보상, 게임 종류 여부 초기화
        state = str(env.reset())
        episode_rewards = 0
        done = False

        # 에피소드 진행을 위한 반복문
        while not done:
            if step >= run_step:
                train_mode = False
                # env.render()
            
            # 행동 결정
            action = agent.get_action(state)

            # 다음 상태, 보상, 게임 종료 정보 취득
            next_state, reward, done, _ = env.step(action)
            next_state = str(next_state)
            episode_rewards += reward

            # 학습 모드인 경우 Q-table 업데이트
            if train_mode:
                # epsilon 감소
                if agent.epsilon > epsilon_min:
                    agent.epsilon -= 1 / run_step
                
                # 학습 수행
                agent.train_model(state, action, reward, next_state, done)
            else:
                agent.epsilon = 0.0     # 학습된 대로 행동을 결정
            
            state = next_state
            step += 1
        
        reward_list.append(episode_rewards)
        episode += 1
    
        if episode != 0 and episode % print_episode == 0:
            print("Step: {} | Episode: {} | Epsilon: {:.3f} | Mean Rewards: {:.3f}".format(
                step, episode, agent.epsilon, np.mean(reward_list)))
            reward_list = []
    
    env.close()
```