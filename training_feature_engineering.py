# training
# code block 1 : 데이터로부터 각종 금융 지표 추출하기
from tqdm import tqdm
import scipy
import numpy as np
import pandas as pd
import os

def weighted_mean(weight_array):
    def inner(x):
        return (weight_array * x).mean()
    return inner

def feature_engineering():
    print('Data Feature Engineering Started')
    # 시장 데이터 가져오기
    df_feature_market = pd.read_csv(os.path.join('data', 'training', 'training_market', 'feature', 'feature_market.csv'), index_col=False, converters={'date': lambda x: str(x)})

    # 시장 데이터 전처리 - nan을 제거하거나 적절하게 대체
    df_training_market = df_feature_market.fillna(method='ffill')
    df_training_market = df_training_market.fillna(0)

    # trainingdata 폴더 만들고 저장
    file_path = os.path.join('data', 'training', 'training_market', 'trainingdata', 'trainingdata_market.csv')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df_training_market.to_csv(file_path, index=False)

    # 주식 feature 데이터 가져오기
    for path, dirs, files in os.walk(os.path.join('data', 'training', 'training_stock', 'feature')):
        pbar = tqdm(files)
        for filename in pbar:
            code = filename.split('.')[0].split('_')[-2]
            name = filename.split('.')[0].split('_')[-1]
            pbar.set_description(code)
            file_path = os.path.join(path, filename)
            df_feature = pd.read_csv(file_path, index_col=False, converters={k: lambda x: str(x) for k in ['code', 'name', 'date']})

            # preporcess
            df_training = df_feature.fillna(method='ffill')
            df_training = df_training.fillna(0)

            # find peaks - 레이블 생성을 위해 종가 차트에서 peak를 찾아내기
            peaks = []
            peaks.extend(scipy.signal.find_peaks(df_training['close'], distance=5, width=10)[0])
            peaks.extend(scipy.signal.find_peaks(-df_training['close'], distance=5, width=10)[0])
            if len(df_training)-1 not in peaks:
                peaks.append(len(df_training)-1)

            # 수익성, 안전성, 유동성 지표 추가 - peak_date, peak_close는 수익성/위험성/유동성 지표 구하기 위해 임시로 생성
            df_training.loc[:, 'peak_date'] = ''
            df_training.loc[:, 'peak_close'] = np.nan
            df_training.loc[:, 'peak_diffratio'] = np.nan
            df_training.loc[:, 'interpeak_mdd'] = np.nan
            df_training.loc[:, 'interpeak_trans_price_exp'] = np.nan
            
            # ema start
            df_training['ema5'] = df_training['close'].ewm(5).mean()
            df_training['ema20'] = df_training['close'].ewm(20).mean()
            df_training['ema100'] = df_training['close'].ewm(100).mean()
            df_training['ema200'] = df_training['close'].ewm(200).mean()
            # ema end
            
            # wma start
            weights = np.arange(1,6)
            wma5 = df_training['close'].rolling(5).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
            
            weights = np.arange(1,21)
            wma20 = df_training['close'].rolling(20).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
            
            weights = np.arange(1,101)
            wma100 = df_training['close'].rolling(100).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
            
            weights = np.arange(1,201)
            wma200 = df_training['close'].rolling(200).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

            df_training['wma5'] = wma5
            df_training['wma20'] = wma20
            df_training['wma100'] = wma100
            df_training['wma200'] = wma200
            # wma end
            
            # macd start
            # 단기(window = 12) 지수 이동 평균선
            short_ema = df_training['close'].ewm(span = 12, adjust = False).mean()

            # 장기(window = 26) 지수 이동 평균선
            long_ema = df_training['close'].ewm(span = 26, adjust = False).mean()

            macd_line = short_ema - long_ema

            # macd column 추가
            df_training['macd'] = macd_line

            #signal(window = 9) column 추가
            df_training['signal'] = df_training['macd'].ewm(span = 9, adjust = False).mean()

            # macd 선이 signal 선을 상향 돌파 -> 매수 사인
            # macd 선이 signal 선을 하향 돌파 -> 매도 사인

            # 빈 데이터 채워넣기
            #df_training = df_training.fillna(method='bfill')
            # macd end
            
            # for bb
            df_training.loc[:, 'bb_middle'] = np.nan
            df_training.loc[:, 'stddev'] = np.nan
            df_training.loc[:, 'bb_upper'] = np.nan
            df_training.loc[:, 'bb_lower'] = np.nan
            df_training.loc[:, 'bb_bandwidth'] = np.nan
            
            # bb start        
            df_training.loc[:, 'bb_middle'] = df_training.loc[:, 'close'].rolling(window=20).mean()
            df_training.loc[:, 'stddev'] = df_training.loc[:, 'close'].rolling(window=20).std()
            df_training.loc[:, 'bb_upper'] = df_training.loc[:, 'bb_middle'] + (df_training.loc[:, 'stddev'] * 2)
            df_training.loc[:, 'bb_lower'] = df_training.loc[:, 'bb_middle'] - (df_training.loc[:, 'stddev'] * 2)
            df_training.loc[:, 'bb_bandwidth'] = (df_training.loc[:, 'close'] - df_training.loc[:, 'bb_lower']) / df_training.loc[:, 'bb_middle'] * 100

            df_training = df_training.fillna(method='bfill')
            
            df_training = df_training.drop(['stddev'], axis=1)        
            # bb end

            #daily_return start
            df_training['daily_return']=df_training['close'].pct_change()*100
            #daily_return end
            
            #수익률 표준편차 start
            rolling_window=20
            df_training['stddev_returns'] = df_training['daily_return'].rolling(window=rolling_window).std()
            #수익률 표준편차 end

            _last_date = ''
            for peak in peaks:
                _date = df_training.iloc[peak]['date']
                _close = df_training.iloc[peak]['close']
                mask = (df_training['date'] >= _last_date) & (df_training['date'] <= _date)
                _last_date = _date
                if len(df_training[mask]) > 0:
                    df_training.loc[mask, 'peak_date'] = _date
                    df_training.loc[mask, 'peak_close'] = _close
                    _x = np.array(df_training.loc[mask, 'close'])
                    lower = np.argmax(np.maximum.accumulate(_x) - _x)
                    upper = np.argmax(_x[:lower+1])
                    
                    # 위험성 지표 - 다음 피크까지의 최대 낙폭(MDD)
                    df_training.loc[mask, 'interpeak_mdd'] = (_x[lower] - _x[upper]) / _x[upper]
                    
                    # 유동성 지표 - 다음 피크까지의 거래대금 예측치의 평균
                    df_training.loc[mask, 'interpeak_trans_price_exp'] = df_training.loc[mask, 'trans_price_exp'].mean()

            # 수익성 지표 - 현재의 종가와 다음 피크와의 종가 비율
            df_training.loc[:, 'peak_diffratio'] = (df_training.loc[:, 'peak_close'] - df_training.loc[:, 'close']) / df_training.loc[:, 'close']

            # peak_date, peak_close feature는 이제 필요없으니 제거
            df_training = df_training.drop(['peak_date', 'peak_close'], axis=1)

            # trainingdata 파일에 저장
            file_path = os.path.join('data', 'training', 'training_stock', 'trainingdata', f'trainingdata_{code}_{name}.csv')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df_training.to_csv(file_path, index=False)
    print('Data Feature Engineering Finished')