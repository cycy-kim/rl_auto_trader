import numpy as np

from agents.ddpg_agent import DDPGAgent
from agents.td3_agent import TD3Agent
from agents.ppo_agent import PPOAgent
from environment.train_environment import TrainEnvironment
from utils.data_utils import load_data
from utils.visualizer import TensorBoardLogger

def train(data_name, initial_balance, log=True, **params):
    chart_data = load_data(f'data/{data_name}')
    env = TrainEnvironment(initial_balance=initial_balance, chart_data=chart_data)
    env.reset()

    writer = TensorBoardLogger(log_dir='runs/train', enabled=log)
    writer.launch_tensorboard()

    state_dim = env.STATE_SPACE
    action_dim = env.ACTION_SPACE

    # TD3 에이전트 생성
    agent = TD3Agent(state_dim, action_dim, **params)
    # agent = PPOAgent(state_dim, action_dim, **params)
    # agent = DDPGAgent(state_dim, action_dim, **params)
    total_episode = 10000  # 총 학습 반복 횟수

    # 학습 루프
    for episode in range(total_episode):
        done = False
        env.reset()
        state = env.initial_state
        total_reward = 0
        minutes = 1
        finished = False
        while not done:

            action = agent.select_action(state) # TD3, DDPG, SAC
            # action, value_pred, log_prob = agent.select_action(state) # PPO
            
            next_state, reward, done = env.step(action)
            if next_state is None:
                finished = True
                break
            
            total_reward += reward
            agent.store_transition(state, action, reward, next_state, done) # TD3, DDPG, SAC
            # agent.store_transition(state, np.argmax(action), reward, value_pred, log_prob, next_state, done) # PPO
            
            state = next_state
            minutes += 1


        portfolio_value = env.get_portfolio_value()
        writer.log_scalar('Reward', total_reward, episode)
        writer.log_scalar('Portfolio Value', portfolio_value, episode)
        agent.train()

        # 주기적으로 모델 저장
        if episode % 100 == 0:
            agent.save(f"models/td3_model_{episode}")
        if finished:
            agent.save(f"models/td3_model_{episode}")
            break

    writer.close()
    
