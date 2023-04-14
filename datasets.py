#%%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#%% The exposed proxy the get dataloader of a specific dataset
def CPS_data_loader(dataset_cfg):
    if dataset_cfg['dataset_name'] == 'SWaT':
        return load_SWaT(dataset_cfg['data_path'], dataset_cfg['batch_size'], dataset_cfg['num_features'])
    
    pass
# %%
def load_SWaT(root, batch_size, num_features):
    data =  pd.read_csv(root)
    
    # Modify timestamp and labels
    data = data.rename(columns={"Normal/Attack":"label"})
    data.label[data.label!="Normal"]=1
    data.label[data.label=="Normal"]=0
    
    data["Timestamp"] = data[" Timestamp"].apply(lambda x: x.strip())
    data = data.drop(columns=[" Timestamp"])

    data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%d/%m/%Y %I:%M:%S %p")
    data["Timestamp"].apply(lambda x: x.strftime('%Y/%m/%d %H:%M:%S'))
    data = data.set_index("Timestamp")

    # Normalize the features. 
    feature = data.iloc[:, :num_features]
    mean_df = feature.mean(axis=0)
    std_df = feature.std(axis=0)

    norm_feature = (feature-mean_df)/std_df
    norm_feature = norm_feature.dropna(axis=1)
    
    # Get the num of sensors
    n_sensor = len(norm_feature.columns)

    # Then Split it into train, val,test.
    train_features = norm_feature.iloc[:int(0.6*len(data))]
    train_label = data.label.iloc[:int(0.6*len(data))]

    val_features = norm_feature.iloc[int(0.6*len(data)):int(0.8*len(data))]
    val_label = data.label.iloc[int(0.6*len(data)):int(0.8*len(data))]
    
    test_features = norm_feature.iloc[int(0.8*len(data)):]
    test_label = data.label.iloc[int(0.8*len(data)):]
       
    # Dataframe ——> Dataset ——> DataLoader
    train_loader = DataLoader(Water(train_features,train_label), batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(Water(val_features,val_label), batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(Water(test_features,test_label), batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader, n_sensor

class Water(Dataset):
    """
        features(dataframe): timestamp * features
        label: timestamp * label
        window_size:
        stride_size:
    """
    def __init__(self, features, label, window_size=60, stride_size=10):
        super(Water, self).__init__()
        self.features = features
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(features, label)
        # self.label = 1.0-2*self.label 


    """Get the features(numpy), valide window idx, each window's label
    """
    def preprocess(self, features, label):
        start_idx = np.arange(0, len(features) - self.window_size, self.stride_size)
        end_idx = np.arange(self.window_size, len(features), self.stride_size)

        delat_time = features.index[end_idx] - features.index[start_idx]
        idx_mask = delat_time == pd.Timedelta(self.window_size, unit='s')

        # Convert the features into numpy and set the first label as a window's label
        return features.values, start_idx[idx_mask], label[start_idx[idx_mask]]

    def __len__(self):
        
        length = len(self.idx)  # num of valid samples

        return length


    """ Return feature_window(L * K * D) and the corresponding label
        K: num of features in a timestep
        L: Timesteps, aka size of a window
        D: num of dimensions of a feature (1 for original version)
    """
    def __getitem__(self, index):
        start = self.idx[index]
        end = start + self.window_size
        
        feature_window = self.data[start:end]
        feature_window = feature_window.reshape([self.window_size, -1, 1]).astype(np.float64)  # L * K * D 

        return torch.FloatTensor(feature_window), self.label[index]
    
# #%%
# def train_batch(features, labels, model, optimizer, criterion, device):
#     features, labels = features.to(device), labels.to(device)

#     # Forward pass ➡
#     outputs = model(features)
#     labels = torch.squeeze(features)
#     print(outputs.shape)
#     print(labels.shape)
    
#     loss = criterion(outputs, labels)
    
#     # Backward pass ⬅
#     optimizer.zero_grad()
#     loss.backward()

#     # Step with optimizer
#     optimizer.step()

#     return loss

#%%
# dataset_cfg = {'dataset_name': 'SWaT','data_path': 'E:/Code_Zero/raw_data/Swat_physical/SWaT_Dataset_Attack_v0.csv',
#                'num_features': 51, 'batch_size': 16}
# train_loader, val_loader, test_loader, n_sensor = CPS_data_loader(dataset_cfg)

# example_ct = 0  # number of examples seen
# batch_ct = 0
# device = 'cpu'

# from models import LSTM
# import torch.nn
# import torch
# model = LSTM.RecurrentAE(n_features=44, latent_dim=32, device=device) # input: N * L * K * (D)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# criterion = torch.nn.CrossEntropyLoss()

# for _, (features, labels) in enumerate(train_loader):
#             # Average loss within a batch
#             print(features.shape)
#             print('------------------------------------------')
#             print(torch.squeeze(features,-1).shape)
#             print('------------------------------------------')
            
#             loss = train_batch(torch.squeeze(features,-1), labels, model, optimizer, criterion, device)
#             example_ct += len(features)
#             batch_ct += 1
