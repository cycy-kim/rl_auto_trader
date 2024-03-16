from agents.td3_agent import TD3Agent
from environment.train_environment import TrainEnvironment
from utils.data_utils import load_data
from utils.visualizer import TensorBoardLogger

def backtest(data_name, initial_balance, log=True):
    chart_data = load_data(f'data/{data_name}')
    env = TrainEnvironment(initial_balance, chart_data)
    env.reset()

    writer = TensorBoardLogger(log_dir='runs/backtest', enabled=log)
    writer.launch_tensorboard()
   
    state_dim = env.STATE_SPACE
    action_dim = env.ACTION_SPACE

    # TD3 에이전트 생성
    agent = TD3Agent(state_dim, action_dim)
    # 모델 로드
    agent.load('models/td3_model_260')
    
    
    total_episode = 10000  # 총 학습 반복 횟수

    for episode in range(total_episode):
        done = False
        state = env.initial_state
        total_reward = 0
        minutes = 1
        finished = False
        while not done:
            # print(state)
            action = agent.select_action(state, add_noise=False)
            # print(action)
            next_state, reward, done = env.step(action)
            if next_state is None:
                finished = True
                break
            total_reward += reward
            
            state = next_state
            minutes += 1

        portfolio_value = env.get_portfolio_value()
        writer.log_scalar('Portfolio Value', portfolio_value, episode)

        if finished:
            break

    writer.close()