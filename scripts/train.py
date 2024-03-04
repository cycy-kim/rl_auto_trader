from agent.agent import TD3Agent
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
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            if next_state is None:
                finished = True
                break
            
            total_reward += reward

            agent.replay_buffer.append(state, action, reward, next_state, done)
            
            state = next_state
            minutes += 1
            # print('state : ' + str(state))
            # print('action : ' + str(action))
            # print('reward : ' + str(reward))
            # print('total minutes : ' + str(minutes))
            # print('----')

        portfolio_value = env.get_portfolio_value()
        print('---episode' + str(episode) + ' finished---')
        print('total_reward : ' +str(total_reward))
        # print('balance : ' +str(env.balance))
        # print('stock_quantity : ' +str(env.stock_quantity))
        # print('avg_purchase_price : ' +str(env.avg_purchase_price))
        # print('Portfolio Value : ' + str(portfolio_value))
        print('num buy, sell , hold : ' + str(env.num_buy) + ' ' + str(env.num_sell) + ' ' + str(env.num_hold))
        print('std_dev of gaussian noise : ' + str(agent.exploration_noise.std_dev))
        # print('----')
        # writer.add_scalar('Total Reward', total_reward, episode)
        writer.log_scalar('Portfolio Value', portfolio_value, episode)
        # 하루가 끝날때마다(episode가 끝날때마다) train
        # if len(agent.replay_buffer) > batch_size:
        agent.train()

        # 주기적으로 모델 저장
        if episode % 100 == 0:
            agent.save(f"models/td3_model_{episode}")
        if finished:
            agent.save(f"models/td3_model_{episode}")
            break

    writer.close()
    
