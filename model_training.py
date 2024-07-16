import os
from tqdm import tqdm
import torch
import numpy as np
from model import model
from model import device

def training_model():
    print('Model Training Started')
    for epoch in tqdm(range(10)):
        for path, dirs, files in os.walk(os.path.join('data', 'training', 'training_stock', 'X')):
            loss = 0
            for filename in tqdm(files):
                code = filename.split('.')[0].split('_')[-2]
                name = filename.split('.')[0].split('_')[-1]
                # feature 데이터 X 가져오기
                file_path = os.path.join(path, filename)
                X = np.load(file_path)
                # label y 가져오기
                file_path = os.path.join(os.path.dirname(path), 'y', f'y_{code}_{name}.npy')
                y = np.load(file_path)
                
                X = torch.from_numpy(X).float().to(device)
                y = torch.from_numpy(y).float().to(device)
                
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
                optimizer.zero_grad()

                # 모델에 feature X 넣어 y_pred(예측값) 도출
                y_pred = model(X)

                # y_pred(예측값)와 y(정답값)의 차이(loss)를 줄여가는 방향으로 모델을 학습시키기
                _loss = criterion(y_pred[:, 0], y)
                _loss.backward()
                optimizer.step()
                _loss = _loss.item()
                loss += _loss
            print(f'epoch={epoch}\tloss={loss}')

    # 학습이 완료된 모델을 파일로 저장
    os.makedirs('models', exist_ok=True)
    path = os.path.join('models', 'model.mdl')
    torch.save(model, path)
    print('Model Training Finished')