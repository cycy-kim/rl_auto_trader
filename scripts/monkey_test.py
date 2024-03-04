from environment.environment import Environment
from utils.data_utils import load_data
from utils.utils import get_random_action
from utils.visualizer import TensorBoardLogger

def monkey_test(log=True):
    chart_data = load_data('data/삼성전자_분봉_2024-01-29.csv')
    env = Environment(chart_data, 1000000)
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


            # 안끝나는거 수정해야됨
            if next_state is None:
                break

            total_reward += reward
            print('total minutes:', minutes)
            minutes += 1

        
        portfolio_value = env.get_portfolio_value()
        print('--- Episode', episode, 'finished ---')
        print('Total Reward:', total_reward)
        print('Balance:', env.balance)
        print('Stock Quantity:', env.stock_quantity)
        print('Avg Purchase Price:', env.avg_purchase_price)
        print('Portfolio Value:', portfolio_value)
        print('Num Buy, Sell, Hold:', env.num_buy, env.num_sell, env.num_hold)
        print('----')

        writer.log_scalar('Portfolio Value', portfolio_value, episode)

    writer.close()