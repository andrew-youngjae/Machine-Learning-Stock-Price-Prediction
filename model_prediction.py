# code block 8 : testing data에 대해 label과 prediction 비교
import pandas as pd
import torch
import numpy as np
import os
from tqdm import tqdm
from model import model
from model import device

def model_prediction():
    y_test = []
    y_pred_test = []
    for path, dirs, files in os.walk(os.path.join('data', 'testing', 'testing_stock', 'X')):
        loss = 0
        for filename in tqdm(files):
            code = filename.split('.')[0].split('_')[-2]
            name = filename.split('.')[0].split('_')[-1]
            file_path = os.path.join(path, filename)
            X = np.load(file_path)
            file_path = os.path.join(os.path.dirname(path), 'y', f'y_{code}_{name}.npy')
            _y = np.load(file_path)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            with torch.no_grad():
                X = torch.from_numpy(X).float().to(device)
                pred = model(X).detach().cpu().numpy()
                pred = pred.flatten()
            
            y_test.extend(_y)
            y_pred_test.extend(pred)

    df_eval_test = pd.DataFrame(list(zip(y_test, y_pred_test)), columns=['y', 'y_pred']).dropna()
    df_eval_test.plot.scatter('y', 'y_pred', s=0.01)