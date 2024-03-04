from datetime import datetime
import numpy as np

from utils.buffers import SlidingWindowBuffer
from environment.environment import Environment
from environment.api.api_client import APIClient


class RealEnvironment(Environment):
    def __init__(self, env, ticker_symbol):
        self.api_client = APIClient(trading_env = env)
        initial_balance = self.api_client.get_balance()['dnca_tot_amt']
        self.ticker_symbol = ticker_symbol
        super().__init__(initial_balance)
        
        
        self.market_start = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

        self.intervals = [5, 10, 20, 60]
        self.closing_price_buffer = SlidingWindowBuffer(self.intervals)
        self.trading_volume_buffer = SlidingWindowBuffer(self.intervals)

        self.initial_state = None

    def reset(self):
        self.closing_price_buffer.clear()
        self.trading_volume_buffer.clear()

        self._observe()

        self.balance = int(self.api_client.get_balance()['dnca_tot_amt'])
        self.stock_quantity = 0
        self.avg_purchase_price = 0
        self.initial_state = self._get_state()

    def _observe(self):
        now = datetime.now()
        
        elapsed_time_minutes = int((now - self.market_start).total_seconds() / 60)
        
        current_time_hhmmss = now.strftime("%H%M%S")
        cur_chartprice = self.api_client.get_cur_chartprice('005930', current_time_hhmmss)[0]

        closing_price = int(cur_chartprice['stck_prpr'])
        trading_volume = int(cur_chartprice['cntg_vol'])
        self.closing_price_buffer.add(closing_price)
        self.trading_volume_buffer.add(trading_volume)
        
        avg_closing_prices = self.closing_price_buffer.get_mean(self.intervals)
        avg_trading_volumes = self.trading_volume_buffer.get_mean(self.intervals)

        elapsed_time_minutes = int((now - self.market_start).total_seconds() / 60)

        self.observation = ([
            elapsed_time_minutes,
            int(cur_chartprice['stck_oprc']),
            int(cur_chartprice['stck_hgpr']),
            int(cur_chartprice['stck_lwpr']),
            closing_price]
            + avg_closing_prices
            + [trading_volume]
            + avg_trading_volumes)
        
        return self.observation

    def _act(self, action):
        action_mode = np.argmax(action)
        action_ratio = (action[action_mode] - self.ACTION_THRESHOLD) / (1 - self.ACTION_THRESHOLD)
        # action_ratio = 1

        if action_ratio < 0:
            action_mode = self.HOLD
        quantity = 0
        if action_mode == self.BUY:
            self.num_buy += 1
            cur_stock_price = self._get_price()
            quantity = self.balance * action_ratio                               # 구매하는데 쓰일 금액
            quantity_for_purchase = int(quantity / cur_stock_price)              # 구매할수 있는 주식 수
            actual_purchase_budget = quantity_for_purchase * cur_stock_price     # 실제로 구매한 주식 금액

            order_response = self.api_client.order('buy', self.ticker_symbol, quantity_for_purchase)

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

