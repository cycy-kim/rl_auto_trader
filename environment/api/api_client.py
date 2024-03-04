import requests
import os
import yaml
import json
import sys

"""
trading_env = 'real' or 'virtual'
"""
class APIClient:
    def __init__(self, trading_env):
        self.api_config = self._get_api_config()

        self.trading_env = trading_env
        self.account_number = os.getenv('ACCOUNT_NO', None)
        self.appkey = os.getenv('APPKEY', None)
        self.appsecret = os.getenv('APPSECRET', None)
        self.base_url = self.api_config['api']['base_url'][self.trading_env]
        
        if self.appkey is None or self.appsecret is None or self.account_number is None:
            print("no appkey or appsecret")
            """
            appkey랑 appsecret재발급 안내문구 추가
            """


        self.base_header = {
            'appkey' : self.appkey,
            'appsecret': self.appsecret
        }
        ouauth_token = self._get_ouauth_token()
        self.base_header['authorization'] = f'Bearer {ouauth_token}'

    def _get_ouauth_token(self):
        header = self.base_header
        header['grant_type'] = 'client_credentials'
        response = requests.post(self.base_url + self.api_config['api']['auth']['endpoint'], data=json.dumps(header))
        try:
            access_token = (response.json())['access_token']
            return access_token
        except KeyError:
            print(f"[{response.status_code}] 'access_token'을 찾을 수 없습니다. 요청이 너무 자주 발생했을 수 있습니다. 1분 후에 다시 시도해주세요.")
            sys.exit(1)
    
    def _get_api_config(self):
        with open('api_config.yaml', 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
        

    def order(self, trade_type, ticker_symbol, quantity, order_price=0):
        order_api_config = self.api_config['api']['order']

        url = f"{self.base_url}{order_api_config['endpoint']}"

        header = self.base_header
        header['tr_id'] = order_api_config['tr_id'][self.trading_env][trade_type]

        body = {
            'CANO' : self.account_number,
            'ACNT_PRDT_CD' : "01",
            'PDNO' : ticker_symbol,
            'ORD_DVSN' : "01",  # 시장가
            'ORD_QTY' : str(quantity),
            'ORD_UNPR' : "0",
        } 

        response = requests.post(url, headers=header, data=json.dumps(body))
        return response.json()

    def get_balance(self):
        balance_inquiry_api_config = self.api_config['api']['balance_inquiry']

        url = f"{self.base_url}{balance_inquiry_api_config['endpoint']}"

        header = self.base_header
        header['tr_id'] = balance_inquiry_api_config['tr_id'][self.trading_env]

        params = {
            'CANO':self.account_number,
            'ACNT_PRDT_CD':'01',
            'AFHR_FLPR_YN':'N',
            'OFL_YN':'',
            'INQR_DVSN':'02',
            'UNPR_DVSN':'01',
            'FUND_STTL_ICLD_YN':'N',
            'FNCG_AMT_AUTO_RDPT_YN':'N',
            'PRCS_DVSN':'01',
            'CTX_AREA_FK100':'',
            'CTX_AREA_NK100':'',
        }

        response = requests.get(url, headers=header, params=params)

        return json.loads(response.text)['output2'][0]


    # 30분전~현재 분봉 받아옴
    def get_cur_chartprice(self, iscd, cur_time):
        price_inquiry_api_config = self.api_config['api']['price_inquiry']

        url = f"{self.base_url}{price_inquiry_api_config['endpoint']}"

        header = self.base_header
        header['content-type'] = 'application/json; charset=utf-8'
        header['tr_id'] = price_inquiry_api_config['tr_id'][self.trading_env]
        header['custtype'] = 'P'

        time_hhmmss = cur_time
        params = {
            'FID_ETC_CLS_CODE':'',
            'FID_COND_MRKT_DIV_CODE':'J',
            'FID_INPUT_ISCD':iscd,
            'FID_INPUT_HOUR_1':time_hhmmss,
            'FID_PW_DATA_INCU_YN':'N',
        }

        response = requests.get(url, headers=header, params=params)

        return json.loads(response.text)['output2']