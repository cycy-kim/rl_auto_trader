# rl_auto_trader

강화학습을 활용하여 모델을 학습시키고, 학습된 모델을 통해 실시간 주식 거래를 자동화하는 시스템을 구축하는 프로젝트입니다.

## 시작하기
### 설치
```cmd
pip install -r requirements.txt
```
### 계좌 개설 및 API사용신청
1. 'predict' 모드에서 훈련된 모델을 바탕으로 실시간 자동 거래 기능을 이용하려면 [한국투자증권](https://securities.koreainvestment.com/main/Main.jsp) 계좌가 필요합니다.
2. 추가적으로, 가상 모의투자 기능을 사용하시려면 [모의투자 참가 신청](https://securities.koreainvestment.com/main/research/virtual/_static/TF07da010000.jsp)을 해주세요.
3. 계좌 개설이 완료되었다면, [API 신청](https://apiportal.koreainvestment.com/intro)을 해주세요.
4. API 신청까지 완료되었다면, 실전/모의 계좌번호, APPKEY, APPSECRET을 확인해주세요.

### .env파일 작성
다음의 양식으로 프로젝트의 main.py와 동일한 위치에 .env 파일을 만들어주세요.<br/>
```env
# 실전 투자 계좌 번호
ACCOUNT_NO_real=your_real_account_number
# 모의 투자 계좌 번호
ACCOUNT_NO_virtual=your_virtual_account_number
APPKEY=your_appkey
APPSECRET=your_appsecret
```

## 사용예시
### train mode
```cmd
python main.py train --data_name 삼성전자_분봉_1년.csv --initial_balance 10000 --sequence_length 8 --iterations 100 --batch_size 100 --discount 0.99 --tau 0.005 --noise_clip 0.7 --policy_freq 2 --log
```
### test mode
```cmd
python main.py test --data_name 삼성전자_분봉_2024-01-29-.csv --initial_balance 10000 --log
```
### predict mode
predict 모드는 실행되는 동안 학습된 모델을 사용하여 해당 종목에 대해 매 분마다 자동 매매를 진행합니다.
#### 모의투자
```cmd
python main.py predict --environment virtual --model td3_model_260 --ticker_symbol 005930
```
#### 실전투자
```cmd
python main.py predict --environment real --model td3_model_260 --ticker_symbol 005930
```
