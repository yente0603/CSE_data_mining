import csv
import numpy as np
import dataProcess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import normalize, StandardScaler
class NNClassifier(nn.Module):
    def __init__(self, list_layer):
        super(NNClassifier, self).__init__()
        self.func_in = nn.Linear(*list_layer[0])
        self.hiddn_layers = []
        for layer in range(len(list_layer) - 2 ):
            self.add_module('h_layer_' + str(layer), nn.Linear(*list_layer[layer+1]))
            # getattr(self, 'h_layer_' + str(layer)).weight.data.normal_(0, 0.1)
            self.hiddn_layers.append(getattr(self, 'h_layer_' + str(layer)))
        self.func_out = nn.Linear(*list_layer[-1])
        self.bool_GPU_available = False

    def forward(self, x):
        x = F.softsign(self.func_in(x))
        for layer in self.hiddn_layers:
            x = F.relu(layer(x))
        x = self.func_out(x)
        # x = F.softmax(x, dim=1)
        return x

    def selectDevice(self, bool_force_cpu=False, int_gpu_id=None) -> None:
        if not bool_force_cpu and self.bool_GPU_available:
            gpu_candidate = list()
            for gpu_index in range(torch.cuda.device_count()):
                print("    device [{0}]: {1}".format(gpu_index, torch.cuda.get_device_name(gpu_index)))
                gpu_candidate.append(gpu_index)
            select = -1
            if int_gpu_id is None:
                while select not in gpu_candidate:
                    select = int(input("Please select your device:"))
            else:
                print(f"Designate GPU ID {int_gpu_id}")
                select = int_gpu_id
            torch.cuda.set_device(select)
            self.device = torch.device("cuda")
            print(f"Device choosed: {torch.cuda.get_device_name(self.device)}")
        else:
            print(f"Device choosed : CPU")
            self.device = torch.device("cpu")
    def checkDeviceStatus(self):
        if torch.cuda.is_available():
            self.bool_GPU_available = True

def generateNetwork(list_network_shape):
    interpreted_network_shape = []
    for index in range(len(list_network_shape) - 1):
        interpreted_network_shape.append((list_network_shape[index], list_network_shape[index + 1]))
    return interpreted_network_shape

datset_path = 'final/data/Arrhythmia Data Set/'

train_data, test_data, train_label, test_label = dataProcess.loadFile(datset_path)
# print(test_label)
data = dataProcess.dataClean([train_data, test_data, train_label, test_label])
# print(data[3])
# print(np.where(np.isnan(train_label)))

network_shape = [data[0].shape[1], 64, 128, 64, 9]
list_network = generateNetwork(network_shape)
train_data = data[0]
test_data = data[1]
train_label = data[2] - 1
test_label = data[3] - 1
# unknow_label = np.where(test_label>7)[0]
# test_label = np.delete(test_label, unknow_label, axis=0)
# test_data = np.delete(test_data, unknow_label, axis=0)

# print(test_label)
# print(train_data.shape)
# from imblearn.over_sampling import SMOTE

# smote = SMOTE(random_state=42)

# # fit predictor and target variable
# x_smote, y_smote = smote.fit_resample(train_data, train_label)

# print('Original dataset shape', np.bincount(train_label))
# print('Resample dataset shape', np.bincount(y_smote))
X_train = torch.from_numpy(train_data).float()
y_train = torch.from_numpy(train_label).long()

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
learning_rate = 0.01
num_epochs = 1000
model = NNClassifier(list_network)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    # for batch_idx, (X_train, y_train) in enumerate(dataloader):
    # forward
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    # print(outputs.shape)
    
    # bp
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# std_scaler = StandardScaler()
# test_data = std_scaler.fit_transform(test_data) 
X_test = torch.from_numpy(test_data).float()

# y_test = torch.from_numpy(test_label).long()
outputs = model(X_test)
with torch.no_grad():
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.numpy()
total = predicted.shape[0]
correct = (predicted == test_label).sum().item()
accuracy = correct / total
print(f"accuracy = {accuracy*100:.2f} %")
# i = 10
# print(outputs[-i:])
# print(predicted[-i:])
# print(test_label[-i:])
# print(predicted)
# print(test_label)

