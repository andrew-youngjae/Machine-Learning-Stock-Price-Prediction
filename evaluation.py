# code block 7 : training data에 대해 label과 prediction 비교
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_prediction():
    print('Evaluation Started')
    path = os.path.join('models', 'model.mdl')
    model = torch.load(path)

    y = []
    y_pred = []
    for path, dirs, files in os.walk(os.path.join('data', 'training', 'training_stock', 'X')):
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
            
            y.extend(_y)
            y_pred.extend(pred)

    #print(label_type)
    #df_eval = pd.DataFrame(list(zip(y, y_pred)), columns=['y', 'y_pred'])
    #ax = df_eval.plot.scatter('y', 'y_pred', s=0.01)
    Y = y
    Y_Pred = y_pred
    plt.figure(figsize=(8,6))
    plt.xlabel('Y', fontsize = 10)
    plt.ylabel('Y_Pred', fontsize = 10)
    plt.scatter(Y, Y_Pred)
    print('Evaluation Finished')