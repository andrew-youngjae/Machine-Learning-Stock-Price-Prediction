import columns
import training_feature_engineering
import training_data_preprocessing
import model_input
import labeling
import model
import model_training
import evaluation
import ranking
from datetime import datetime, timedelta

if __name__ == '__main__':
    training_feature_engineering.feature_engineering()
    training_data_preprocessing.preprocess_data()
    model_input.load_model_input()
    labeling.macd_labeling()
    model.construct_model()
    model_training.training_model()
    labeling.volatility_labeling()
    model_training.training_model()
    evaluation.evaluate_prediction()

    # 시작 날짜와 종료 날짜 설정
    start_date = datetime(2022, 4, 1)
    end_date = datetime(2022, 6,24)

    # 날짜 간격 설정 (1주일)
    interval = timedelta(weeks=1)

    ranking.stock_ranking(start_date, end_date, interval)
