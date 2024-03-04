import numpy as np
from abc import abstractmethod

from utils.data_utils import scale, z_score_normalize, min_max_scale

class Environment():
    # 보유 잔고, 보유 주식 수, 평단가        -> 3
    # 시간, 시가, 저가, 고가, 종가, 거래량   -> 6
    # 나머지 이동평균 (120 제외)            -> 8
    STATE_SPACE = 3 + 6 + 8

    BUY = 0
    SELL = 1
    HOLD = 2
    ACTIONS = [BUY, SELL, HOLD]
    ACTION_SPACE = len(ACTIONS)

    ACTION_THRESHOLD = 0.5

    # 사용할 hts에 맞게 수정
    TRADING_CHARGE = 0.00015  # 거래 수수료
    TRADING_TAX = 0.0025  # 거래세

    BALANCE_IDX = 0
    STOCK_QUANTITY_IDX = 1
    AVG_PURCHASE_PRICE_IDX = 2
    TIME_IDX = 3
    OPEN_PRICE_IDX = 4  # 시가
    HIGH_PRICE_IDX = 5  # 고가
    LOW_PRICE_IDX = 6   # 저가
    CLOSE_PRICE_IDX = 7 # 종가
    VOLUMN_IDX = 12     # 거래량

    MONEY_SCALING_FACTOR = 100000

    def __init__(self, initial_balance):
        self.observation = None
        self.idx = -1

        self.initial_balance = initial_balance
        self.previous_portfolio_value = initial_balance
        # 포트폴리오 가치 = balance + stock_quantity * _get_price()
        self.balance = initial_balance
        self.stock_quantity = 0
        self.avg_purchase_price = 0
        
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0

    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def _observe(self):
        pass

    def _get_reward(self):
        current_portfolio_value = self.balance + (self.stock_quantity * self._get_price())
        reward = scale(current_portfolio_value - self.previous_portfolio_value, self.MONEY_SCALING_FACTOR)
        self.previous_portfolio_value = current_portfolio_value
        return reward


    def _get_state(self):
        stock_quantity_scaled = min_max_scale(self.stock_quantity, 0, 100)

        price_data = np.array(self.observation[1:9], dtype=float)  
        volume_data = np.array(self.observation[9:], dtype=float)  

        price_data_normalized = z_score_normalize(price_data)
        volume_data_normalized = z_score_normalize(volume_data)

        minute_scaled = min_max_scale(self.observation[0], 0, 400)

        agent_state = np.array([scale(self.balance, self.MONEY_SCALING_FACTOR), stock_quantity_scaled, scale(self.avg_purchase_price, self.MONEY_SCALING_FACTOR)], dtype=float)
        observation_state = np.concatenate(([minute_scaled], price_data_normalized, volume_data_normalized))

        state = np.concatenate((agent_state, observation_state))
        self.state = state
        return state

    @abstractmethod
    def _act(self, action):
        pass

    def _is_done(self, state):
        if self.observation[0] == -1:
            return True
        return False

    def _get_price(self):
        if self.state is not None:
            return int(self.observation[4]) 
        return None

    # next_state, reward, done = env.step(action)
    def step(self, action):
        if self._observe() is None:
            return None, None, None
        
        action_mode, quantity = self._act(action)
        next_state = self._get_state()

        reward = self._get_reward()
        
        done = self._is_done(next_state)

        return next_state, reward, done

    def get_portfolio_value(self):
        cur_portfolio_value = self.balance + self.stock_quantity * self._get_price()
        return cur_portfolio_value

