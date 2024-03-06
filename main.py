import argparse
import os
import re
from dotenv import load_dotenv

from scripts.train import train
from scripts.backtest import backtest
from scripts.monkey_test import monkey_test
from scripts.predict import predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model in different modes.")
    parser.add_argument('--mode', choices=['train', 'test', 'predict'], default='train', help="Mode to run the model: train, test, or predict.")
    parser.add_argument('--log', action='store_true', help="Enable logging. If not specified, logging is disabled by default.")

    subparsers = parser.add_subparsers(dest='subcommand', help='Subcommand help')

    # train
    train_parser = subparsers.add_parser('train', help='Train mode help')
    train_parser.add_argument('--data_name', required=True, \
        help="Specify the name of the data file located in the 'data/' directory to be used for training. For example, 'train_data.csv'.")
    train_parser.add_argument('--initial_balance', type=float, required=True, \
        help="Specify the initial balance for the trading account used in training. For example, '10000' for an initial balance of 10,000won.")
    # train parameters
    train_parser.add_argument('--sequence_length', type=int, default=8)
    train_parser.add_argument('--iterations', type=int, default=100)
    train_parser.add_argument('--batch_size', type=int, default=100)
    train_parser.add_argument('--discount', type=float, default=0.99)
    train_parser.add_argument('--tau', type=float, default=0.005)
    train_parser.add_argument('--noise_clip',type=float, default=0.7)
    train_parser.add_argument('--policy_freq', type=int, default=2)

    # test
    test_parser = subparsers.add_parser('test', help='Test mode help')
    test_parser.add_argument('--data_name', required=True, \
        help="Specify the name of the data file located in the 'data/' directory to be used for testing. For example, 'test_data.csv'.")
    test_parser.add_argument('--initial_balance', type=float, required=True, \
        help="Specify the initial balance for the trading account used in testing. For example, '10000' for an initial balance of 10,000won.")

    # predict
    predict_parser = subparsers.add_parser('predict', help='Predict mode help')
    predict_parser.add_argument('--environment', choices=['real', 'virtual'], required=True, help="Environment for prediction: real or virtual.")
    predict_parser.add_argument('--model', required=True, \
        help="Specify the model filename located in the 'models/' directory to be used for prediction. For example, 'td3_model_300'.")
    predict_parser.add_argument('--ticker_symbol', required=True, \
        help="Specify the ticker symbol of the stock for which the prediction will be made. For example, '005930' for 삼성전자.")

    args = parser.parse_args()
    if args.subcommand == 'predict':
        # ticker_symbol이 숫자로만 이루어져 있는지 확인
        assert re.match(r'^\d+$', args.ticker_symbol), "The ticker_symbol must be composed of digits only."

        # model 파일 확인
        model_path = os.path.join('models', args.model)
        assert os.path.isfile(model_path), f"The model file '{args.model}' does not exist in the 'models/' directory."

    if args.subcommand in ['train', 'test']:
        # data 파일 확인
        data_path = os.path.join('data', args.data_name)
        assert os.path.isfile(data_path), f"The data file '{args.data_name}' does not exist in the 'data/' directory."

    train_hyperparameters = {
        'sequence_length': getattr(args, 'sequence_length', 8),
        'iterations': getattr(args, 'iterations', 100),
        'batch_size': getattr(args, 'batch_size', 100),
        'discount': getattr(args, 'discount', 0.99),
        'tau': getattr(args, 'tau', 0.005),
        'noise_clip': getattr(args, 'noise_clip', 0.7),
        'policy_freq': getattr(args, 'policy_freq', 2),
    }

    load_dotenv()

    if args.mode == 'train':
        print("Start training...")
        train(data_name=args.data_name, initial_balance=args.initial_balance, log=args.log, **train_hyperparameters) 
        print("Training finished")
    elif args.mode == 'test':
        print("Start testing...")
        backtest(data_name=args.data_name, initial_balance=args.initial_balance, log=args.log)
        # monkey_test(log=args.log)  
        print("Testing finished")
    elif args.mode == 'predict':
        print("Start prediction...")
        predict(env=args.environment, model=args.model, ticker_symbol=args.ticker_symbol, log=args.log)
        print("Prediction finished")