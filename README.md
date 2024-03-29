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
python main.py train --data_name 삼성전자_분봉_1년.csv --initial_balance 1000000 --sequence_length 8 --iterations 100 --batch_size 100 --discount 0.99 --tau 0.005 --noise_clip 0.7 --policy_freq 2
```
### test mode
```cmd
python main.py test --data_name 삼성전자_분봉_2024-01-29-.csv --initial_balance 1000000
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

## 에이전트 추가하기
본 프로젝트는 사용자가 새로운 강화학습 에이전트를 쉽게 추가하고 실험할 수 있도록 설계되었습니다. 새로운 에이전트를 추가하기 위한 기본적인 단계는 다음과 같습니다:
1. agents 디렉토리에 새로운 에이전트 클래스 파일을 생성합니다. 예를 들어, MyAgent.py라는 이름으로 파일을 만듭니다.
2. 에이전트 클래스는 BaseAgent를 상속받아 구현됩니다. 필요한 메소드(select_action, train, cuda, save, load)를 오버라이드하여 구현합니다.
3. main.py 파일 내에서사용자가 명령줄 인터페이스를 통해 새로운 에이전트에 필요한 parameter들을 받을 수 있도록 argparse를 사용하여 새로운 옵션을 추가합니다.
4. 필요한 경우, 새로운 에이전트에 맞게 train, test, predict 모드 등에서의 동작을 조정합니다
  - 예를 들어, TD3Agent를 사용하는 train 스크립트에서 transition을 저장하는 방법은 다음과 같습니다:
    ```python
    agent = TD3Agent(state_dim, action_dim, **params)
    action = agent.select_action(state)
    agent.store_transition(state, action, reward, next_state, done)
    ```
  - PPO의 경우는 다음과 같습니다:
    ```python
    agent = PPOAgent(state_dim, action_dim, **params)
    action, value_pred, log_prob = agent.select_action(state)
    agent.store_transition(state, action, reward, value_pred, log_prob, next_state, done)
    ```
5. 새로운 에이전트를 사용하여 학습, 테스트, 예측을 진행합니다.

## 시각화
TensorBoard를 진행 중 여러 지표의 변화를 시각화하여 볼 수 있습니다. 
시각화 기능을 사용하기 위해선, 실행 시 커맨드라인에 --log 옵션을 추가합니다:
```cmd
python main.py --log train --data_name 삼성전자_분봉_1년.csv --initial_balance 1000000 --sequence_length 8 --iterations 100 --batch_size 100 --discount 0.99 --tau 0.005 --noise_clip 0.7 --policy_freq 2
```
--log 옵션 추가 시, TensorBoard는 자동으로 실행됩니다. 실행 결과를 보시려면 브라우저에서 localhost:6006로 접속해주세요.
![localhost_6006_](https://github.com/cycy-kim/rl_auto_trader/assets/112456373/4e4b2162-8a51-4ca7-8371-ddc14b6eff57)
