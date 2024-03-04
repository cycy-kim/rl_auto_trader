from agent.agent import TD3Agent
import datetime

from environment.real_environment import RealEnvironment
from utils.visualizer import TensorBoardLogger

def predict(env, model, ticker_symbol, log=True):
    env = RealEnvironment(env, ticker_symbol=ticker_symbol)
    env.reset()

    writer = TensorBoardLogger(log_dir='runs/predict', enabled=log)
    writer.launch_tensorboard()
   
    state_dim = env.STATE_SPACE
    action_dim = env.ACTION_SPACE

    # TD3 에이전트 생성
    agent = TD3Agent(state_dim, action_dim)
    # 모델 로드
    agent.load(f'models/{model}')
    total_episode = 10000 

    for episode in range(total_episode):
        done = False
        env.reset()
        state = env.initial_state
        total_reward = 0
        minutes = 1
        finished = False

        last_checked_minute = datetime.datetime.now().minute
        while not done:
            # 매 1분마다 step
            current_minute = datetime.datetime.now().minute
            if last_checked_minute == current_minute:
                continue
            
            action = agent.select_action(state, add_noise=False)
            next_state, reward, done = env.step(action)
            if next_state is None:
                finished = True
                break
            total_reward += reward

            state = next_state
            minutes += 1

            last_checked_minute = current_minute

        portfolio_value = env.get_portfolio_value()
        writer.log_scalar('Portfolio Value', portfolio_value, episode)

        if finished:
            break

    writer.close()