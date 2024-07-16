# training
# code block 2 : 추출해낸 데이터에 대해 정규화 및 일반화 수행
from tqdm import tqdm
import scipy
import numpy as np
import pandas as pd
import os
from columns import COLUMNS
import pickle

def preprocess_data():

    print('Data Preprocessing Started')

    df_training_market = pd.read_csv(os.path.join('data', 'training', 'training_market', 'trainingdata', 'trainingdata_market.csv'), index_col=False, converters={'date': lambda x: str(x)})

    X_samples = None
    X_label_samples = None
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
            df_label = df[['peak_diffratio', 'interpeak_mdd', 'interpeak_trans_price_exp', 'ema5', 'ema20', 'ema100', 'ema200', 'wma5', 'wma20', 'wma100', 'wma200','macd', 'signal', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_bandwidth', 'daily_return','stddev_returns', 'close']].copy()

            _X = df[COLUMNS].values
            _X_label = df_label.values
            _X_samples = _X[np.random.choice(_X.shape[0], 100), :]
            _X_label_samples = _X_label[np.random.choice(_X_label.shape[0], 100), :]
            if X_samples is None:
                X_samples = _X_samples
            else:
                try:
                    X_samples = np.concatenate((X_samples, _X_samples))
                except Exception as e:
                    print(str(e))
            if X_label_samples is None:
                X_label_samples = _X_label_samples
            else:
                try:
                    X_label_samples = np.concatenate((X_label_samples, _X_label_samples))
                except Exception as e:
                    print(str(e))

    # 표준화(StandardScaler), 일반화(MaxAbsScaler)를 위한 Scaler를 생성하고 저장
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler
    scaler_std = StandardScaler()
    scaler_std.fit(X_samples)
    scaler_label_std = StandardScaler()
    scaler_label_std.fit(X_label_samples)
    scaler_norm = MaxAbsScaler()
    scaler_norm.fit(scaler_std.transform(X_samples))
    scaler_label_norm = MaxAbsScaler()
    scaler_label_norm.fit(scaler_label_std.transform(X_label_samples))

    os.makedirs('models', exist_ok=True)
    for name, scaler in zip(['scaler_std', 'scaler_label_std', 'scaler_norm', 'scaler_label_norm'], [scaler_std, scaler_label_std, scaler_norm, scaler_label_norm]):
        path = os.path.join('models', f'{name}.pkl')
        with open(path, 'wb') as fout:
            pickle.dump(scaler, fout)
    print('Data Preprocessing finished')