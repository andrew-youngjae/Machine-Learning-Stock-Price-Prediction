# training
# code block 3 : 모델의 input으로 들어갈 학습 데이터 분리
from tqdm import tqdm
import scipy
import numpy as np
import pandas as pd
import os
from columns import COLUMNS
import pickle

def load_model_input():
    print('Model Input Process Started')
    # 위에서 저장한 Scaler들을 불러와 학습 데이터들을 표준화/일반화
    with open(os.path.join('models', 'scaler_std.pkl'), 'rb') as f:
        scaler_std = pickle.load(f)

    with open(os.path.join('models', 'scaler_label_std.pkl'), 'rb') as f:
        scaler_label_std = pickle.load(f)

    with open(os.path.join('models', 'scaler_norm.pkl'), 'rb') as f:
        scaler_norm = pickle.load(f)

    with open(os.path.join('models', 'scaler_label_norm.pkl'), 'rb') as f:
        scaler_label_norm = pickle.load(f)

    df_training_market = pd.read_csv(os.path.join('data', 'training', 'training_market', 'trainingdata', 'trainingdata_market.csv'), index_col=False, converters={'date': lambda x: str(x)})

    for path, dirs, files in os.walk(os.path.join('data', 'training', 'training_stock', 'trainingdata')):
        pbar = tqdm(files)
        for filename in pbar:
            code = filename.split('.')[0].split('_')[-2]
            name = filename.split('.')[0].split('_')[-1]
            pbar.set_description(code)
            file_path = os.path.join(path, filename)
            df_training = pd.read_csv(file_path, index_col=False, converters={k: lambda x: str(x) for k in ['code', 'name', 'date']})
            df = pd.merge(df_training, df_training_market, on='date', how='left')

            # trainingdata 파일에서 label에 사용할 수익성/위험성/유동성 지표만 따로 df_label로 분리
            df_label = df[['peak_diffratio', 'interpeak_mdd', 'interpeak_trans_price_exp', 'ema5', 'ema20', 'ema100', 'ema200', 'wma5', 'wma20', 'wma100', 'wma200', 'macd', 'signal', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_bandwidth', 'daily_return','stddev_returns', 'close']].copy()

            # feature 데이터들은 X로 저장
            X = df[COLUMNS].values

            # label 데이터들은 X_label로 저장
            X_label = df_label.values

            # feaure와 label 표준화 + 일반화
            X_std_norm = scaler_norm.transform(scaler_std.transform(X))
            X_label_std_norm = scaler_label_norm.transform(scaler_label_std.transform(X_label))

            # feature 데이터 X 파일로 저장
            file_path = os.path.join('data', 'training', 'training_stock', 'X', f'X_{code}_{name}.npy')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, X_std_norm)

            # label 데이터 X_label 파일로 저장
            file_path = os.path.join('data', 'training', 'training_stock', 'X_label', f'X_label_{code}_{name}.npy')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, X_label_std_norm)
    print('Model Input Process finished')