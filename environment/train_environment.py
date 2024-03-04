import numpy as np

from environment.environment import Environment

class TrainEnvironment(Environment):
    def __init__(self, initial_balance, chart_data):
        super().__init__(initial_balance)
        self.chart_data = chart_data
        
        self.initial_state = np.append([self.balance, self.stock_quantity, self.avg_purchase_price] , np.array(self.observation))

    def reset(self):
        self.idx = -1
        self._observe()

        self.balance = self.initial_balance
        self.stock_quantity = 0
        self.avg_purchase_price = 0
        self.initial_state = self._get_state()

    def _observe(self):
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None


    def _act(self, action):
        action_mode = np.argmax(action)
        action_ratio = (action[action_mode] - self.ACTION_THRESHOLD) / (1 - self.ACTION_THRESHOLD)
        # action_ratio = 1

        if action_ratio < 0:
            action_mode = self.HOLD
        quantity = 0
        # 나중에 api랑 연결했을땐 실제로 구매하도록 바꿔야함
        # util에 구현하면 될듯
        if action_mode == self.BUY:
            self.num_buy += 1
            cur_stock_price = self._get_price()
            quantity = self.balance * action_ratio                  # 구매하는데 쓰일 금액
            quantity_for_purchase = int(quantity / cur_stock_price) # 구매할수 있는 주식 수
            actual_purchase_budget = quantity_for_purchase * cur_stock_price    # 실제로 구매한 주식 금액

            # 평단가 갱신
            if self.stock_quantity + quantity_for_purchase == 0:
                self.avg_purchase_price = 0
            else:
                self.avg_purchase_price = (self.avg_purchase_price * self.stock_quantity + cur_stock_price * quantity_for_purchase) \
                / (self.stock_quantity + quantity_for_purchase)

            # 잔고에서 실제로 주식사는데 사용한 금액 -
            self.balance -= actual_purchase_budget
            # 보유 주식 수 갱신
            self.stock_quantity += quantity_for_purchase
            
            quantity = quantity_for_purchase

        elif action_mode == self.SELL:
            self.num_sell += 1
            cur_stock_price = self._get_price()
            quantity = int(self.stock_quantity * action_ratio)
            if self.stock_quantity == 1:
                quantity = 1

            if quantity != 0:
                self.balance += cur_stock_price * quantity
                self.stock_quantity -= quantity
        elif action_mode == self.HOLD:
            self.num_hold += 1
            """"""
        
        return action_mode, quantity
