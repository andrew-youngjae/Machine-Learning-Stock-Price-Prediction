import os
from tqdm import tqdm
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler, MinMaxScaler

label_type = ''

def ema_labeling():
    print('EMA Labeling Started')
    label_type = 'Pretrained by Label 4 + Label 2 : Long-Term EMA - Short-Term EMA'

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    y = []
    for path, dirs, files in os.walk(os.path.join('data', 'training', 'training_stock', 'X_label')):
        pbar = tqdm(files)
        for filename in pbar:
            code = filename.split('.')[0].split('_')[-2]
            name = filename.split('.')[0].split('_')[-1]
            pbar.set_description(code)
            file_path = os.path.join(path, filename)
            X_label = np.load(file_path)

            #new label start
            #ema. 볼린저 밴드

            peak_diffratio = X_label[:,0]
            interpeak_mdd = X_label[:,1]
            interpeak_trans_price_exp = X_label[:,2]
            ema_5 = X_label[:,3]
            ema_20 = X_label[:,4]
            ema_100=X_label[:,5]
            ema_200=X_label[:,6]
            wma_20=X_label[:,7]
            wma_200=X_label[:,9]
            macd = X_label[:,11]
            signal = X_label[:,12]
            bb_middle=X_label[:,13]
            bb_upper = X_label[:,14]
            bb_lower = X_label[:,15]
            bb_bandwidth=X_label[:,16]
            
            labels=[]
            
            for i in range(len(ema_5)):
                if (ema_5[i]>ema_100[i]): #and (macd[i]>macd_signal[i]):
                    label=sigmoid(ema_5[i]-ema_100[i]) #매수
                elif ema_5[i]<ema_100[i]: #and (macd[i]<macd_signal[i]):
                    label=sigmoid(ema_100[i]-ema_5[i])  #매도
                labels.append(label)
            y=labels
            
    
            file_path = os.path.join('data', 'training', 'training_stock', 'y', f'y_{code}_{name}.npy')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, y)
            #new label end


            #file_path = os.path.join('drive','MyDrive','Colab Notebooks','data', 'training', 'qlds_2022a', 'y', f'y_{code}_{name}.npy')
            #os.makedirs(os.path.dirname(file_path), exist_ok=True)
            #np.save(file_path, _y)
    #print(len(labels))
    scipy.stats.describe(y)
    print('EMA Labeling Finished')

def macd_labeling():
    print('MACD Labeling Started')
    label_type = 'Label 4 : MACD - Signal'

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    y = []
    for path, dirs, files in os.walk(os.path.join('data', 'training', 'training_stock', 'X_label')):
        pbar = tqdm(files)
        for filename in pbar:
            code = filename.split('.')[0].split('_')[-2]
            name = filename.split('.')[0].split('_')[-1]
            pbar.set_description(code)
            file_path = os.path.join(path, filename)
            X_label = np.load(file_path)

            #_y = sigmoid((X_label[:, 0] * 0.5 + X_label[:, 1] * 0.3 + X_label[:, 2] * 0.2) * 3)     ##각각 수익성, 안전성, 유동성
            #y.extend(_y)

            #new label start
            #ema. 볼린저 밴드

            peak_diffratio = X_label[:,0]
            interpeak_mdd = X_label[:,1]
            interpeak_trans_price_exp = X_label[:,2]
            ema_5 = X_label[:,3]
            ema_20 = X_label[:,4]
            ema_100=X_label[:,5]
            ema_200=X_label[:,6]
            wma_20=X_label[:,7]
            wma_200=X_label[:,9]
            macd = X_label[:,11]
            signal = X_label[:,12]
            bb_middle=X_label[:,13]
            bb_upper = X_label[:,14]
            bb_lower = X_label[:,15]
            bb_bandwidth=X_label[:,16]

            labels=[]
            
            for i in range(len(ema_5)):
                if macd[i]>signal[i]:
                    label=sigmoid(macd[i]-signal[i])  #매수
                elif macd[i]<signal[i]:
                    label=sigmoid(signal[i]-macd[i])  #매도
                labels.append(label)
            y=labels

            file_path = os.path.join('data', 'training', 'training_stock', 'y', f'y_{code}_{name}.npy')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, y)
            
    scipy.stats.describe(y)
    print('MACD Labeling Finished')

def volatility_labeling():
    #변동성 label
    print('Volatility Labeling Started')
    label_type = 'Pretrained by Label 4 + Label 6 : Daily Returns - Standard Deviation of Returns'

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    y = []
    for path, dirs, files in os.walk(os.path.join('data', 'training', 'training_stock', 'X_label')):
        pbar = tqdm(files)
        for filename in pbar:
            code = filename.split('.')[0].split('_')[-2]
            name = filename.split('.')[0].split('_')[-1]
            pbar.set_description(code)
            file_path = os.path.join(path, filename)
            X_label = np.load(file_path)

            #_y = sigmoid((X_label[:, 0] * 0.5 + X_label[:, 1] * 0.3 + X_label[:, 2] * 0.2) * 3)     ##각각 수익성, 안전성, 유동성
            #y.extend(_y)


            #new label start
            #ema. 볼린저 밴드

            peak_diffratio = X_label[:,0]
            interpeak_mdd = X_label[:,1]
            interpeak_trans_price_exp = X_label[:,2]
            ema_5 = X_label[:,3]
            ema_20 = X_label[:,4]
            ema_100=X_label[:,5]
            ema_200=X_label[:,6]
            wma_20=X_label[:,7]
            wma_200=X_label[:,9]
            macd = X_label[:,11]
            signal = X_label[:,12]
            bb_middle=X_label[:13]
            bb_upper = X_label[:,14]
            bb_lower = X_label[:,15]
            bb_bandwidth=X_label[:,16]
            daily_return=X_label[:,17]
            std_return=X_label[:,18]

            alpha=0.5
            labels=[]
            label = sigmoid(0)
            for i in range(len(ema_5)):
                if daily_return[i]>=alpha*std_return[i]:
                    label=sigmoid(daily_return[i]- alpha*std_return[i])#매수
                elif daily_return[i]<-alpha*std_return[i]:
                    label=sigmoid(daily_return[i] + alpha*std_return[i]) #매도
                labels.append(label)
            y=labels


            file_path = os.path.join('data', 'training', 'training_stock', 'y', f'y_{code}_{name}.npy')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, y)
            #new label end

    #print(len(labels))
    scipy.stats.describe(y)
    print('Volatility Labeling Finished')