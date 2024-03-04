from environment.environment import Environment
from utils.data_utils import load_data
from utils.utils import get_random_action
from utils.visualizer import TensorBoardLogger

def monkey_test(data_name, initial_balance, log=True):
    chart_data = load_data(f'data/{data_name}')
    env = Environment(chart_data, initial_balance)
    env.reset()

    writer = TensorBoardLogger(log_dir='runs/monkey_test', enabled=log)
    writer.launch_tensorboard()

    action_dim = env.ACTION_SPACE

    total_episode = 10000

    for episode in range(total_episode):
        done = False
        total_reward = 0
        minutes = 1

        while not done:
            # 무작위 액션 선택
            action = get_random_action(action_dim)
            next_state, reward, done = env.step(action)

            if next_state is None:
                break

            total_reward += reward
            minutes += 1

        
        portfolio_value = env.get_portfolio_value()
        writer.log_scalar('Portfolio Value', portfolio_value, episode)

    writer.close()