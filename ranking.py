#일주일 간격으로 9개의 주차에 대해 랭킹된 파일을 rankdata 폴더에 저장하게 수정된 랭킹 코드
# ranking with test data
from tqdm import tqdm
from datetime import datetime, timedelta
import scipy
import numpy as np
import pandas as pd
import os
from model import model, device
import pickle
import torch
from columns import COLUMNS

def stock_ranking(start_date, end_date, interval):
    print('Stock Ranking by Model Output Started')
    # 시작 날짜부터 종료 날짜까지 1주일 간격으로 날짜 범위 생성
    date_range = []
    current_date = start_date
    while current_date <= end_date:
        date_range.append(current_date)
        current_date += interval
    df_training_market = pd.read_csv(os.path.join('data', 'training', 'training_market', 'trainingdata', 'trainingdata_market.csv'), index_col=False, converters={'date': lambda x: str(x)})
    for target_date in date_range:
        # 날짜 형식을 문자열로 변환 ('YYYYMMDD')
        target_date_str = target_date.strftime('%Y%m%d')
        
        # 각 날짜에 대한 데이터 수집 및 파일 저장
        
        
        # df_ranking 데이터프레임을 생성하는 코드 
        # 위에서 저장한 Scaler들을 불러와 학습 데이터들을 표준화/일반화
        with open(os.path.join('models', 'scaler_std.pkl'), 'rb') as f:
            scaler_std = pickle.load(f)

        with open(os.path.join('models', 'scaler_label_std.pkl'), 'rb') as f:
            scaler_label_std = pickle.load(f)

        with open(os.path.join('models', 'scaler_norm.pkl'), 'rb') as f:
            scaler_norm = pickle.load(f)

        with open(os.path.join('models', 'scaler_label_norm.pkl'), 'rb') as f:
            scaler_label_norm = pickle.load(f)

        df_ranking = pd.DataFrame()
        list_tmp_df = []

        for path, dirs, files in os.walk(os.path.join('data', 'testing', 'testing_stock', 'trainingdata')):
            pbar = tqdm(files)
            for filename in pbar:
                code = filename.split('.')[0].split('_')[-2]
                name = filename.split('.')[0].split('_')[-1]
                pbar.set_description(code)
                file_path = os.path.join(path, filename)
                df_training = pd.read_csv(file_path, index_col=False, converters={k: lambda x: str(x) for k in ['code', 'name', 'date']})
                
                df = pd.merge(df_training, df_training_market, on='date', how='left')

                # feature 데이터들은 X로 저장
                X = df[COLUMNS].values

                # feaure와 label 표준화 + 일반화
                X_std_norm = scaler_norm.transform(scaler_std.transform(X))

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()
                with torch.no_grad():
                    X = torch.from_numpy(X_std_norm).float().to(device)
                    pred = model(X).detach().cpu().numpy()
                    pred = pred.flatten()

                df['pred'] = pred
                
                df_tmp = df.loc[df.date == target_date_str, ['code', 'name', 'date', 'pred','close','volume']]
                list_tmp_df.append(df_tmp)

            df_ranking = pd.concat(list_tmp_df, ignore_index=True)
            
            df_ranking.sort_values(by='pred', ascending=False)[['code', 'name', 'date', 'pred','close','volume']]
            print(df_ranking)
            # data 폴더 경로 생성
            ranking_data_folder = 'rankdata'
            os.makedirs(ranking_data_folder, exist_ok=True)

            # 파일 이름 생성
            file_name = os.path.join(ranking_data_folder, f'df_ranking_{target_date_str}.csv')

            # df_ranking 데이터프레임을 CSV 파일로 저장
            df_ranking.to_csv(file_name, index=False)
    print('Stock Ranking by Model Output Finished')