import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import os
from dataProcess import *
import copy
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import itertools
from sklearn.metrics import f1_score, accuracy_score

class FC(nn.Module):
	"""Fully connected neural network"""
	def __init__(self, input_size, hidden_size, output_size):
		super(FC, self).__init__()
		self.layer = nn.ParameterList([
			nn.Linear(input_size, hidden_size//2),
			nn.Linear(hidden_size//2, hidden_size//2),
			nn.Linear(hidden_size, hidden_size),
			nn.Linear(hidden_size//2, hidden_size),
			nn.Linear(hidden_size, output_size)
		])
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(0.2)

	def forward(self, x):
		for i in range(len(self.layer) - 1):
			x = self.layer[i](x)
			x = self.activation(x)
			x = self.dropout(x)

		x = self.layer[-1](x)
		return x

def using_SMOTEENN(X_train, y_train):
	ros = RandomOverSampler()
	smote_enn = SMOTEENN()
	pipeline = Pipeline([
		('oversample', ros),
		('smote_enn', smote_enn),
	])
	X_train, y_train = pipeline.fit_resample(X_train, y_train)
	X_train = torch.Tensor(X_train)
	y_train = torch.Tensor(y_train).type(torch.LongTensor)

	return X_train, y_train

def using_SMOTETomek(X_train, y_train):
	ros = RandomOverSampler()
	smote_tomek = SMOTETomek()
	pipeline = Pipeline([
		('oversample', ros),
		('smote_tomek', smote_tomek),
	])
	X_train, y_train = pipeline.fit_resample(X_train, y_train)
	X_train = torch.Tensor(X_train)
	y_train = torch.Tensor(y_train).type(torch.LongTensor)

	return X_train, y_train

def using_smote_oversample(X_train, y_train):
	ros = RandomOverSampler()
	X_train, y_train = ros.fit_resample(X_train, y_train)
	X_train = torch.Tensor(X_train)
	y_train = torch.Tensor(y_train).type(torch.LongTensor)

	return X_train, y_train

def using_smote_undersample(X_train, y_train):
	rus = RandomUnderSampler(random_state=42, sampling_strategy='majority')
	X_train, y_train = rus.fit_resample(X_train, y_train)
	X_train = torch.Tensor(X_train)
	y_train = torch.Tensor(y_train).type(torch.LongTensor)

	return X_train, y_train

def predict(model, dataset, y_test):
	'''
	return: a dict of predictions with key being the prediction and value being the true label
	(every prediction contains 8 different values representing the predicted probabilities of each class)
	'''
	model.eval()
	with torch.no_grad():
		acc = []
		pred = []
		total_samples = 0
		prob_list = {}
		for inputs, labels in dataset:
			outputs = model(inputs)
			_, predicted = torch.max(outputs, dim=0)
			pred.append(predicted.item())
			predicted_probabilities = torch.softmax(outputs, dim=0)
			prob_list[tuple(predicted_probabilities.tolist())] = labels

		print(f'Accuracy: {accuracy_score(torch.Tensor.tolist(y_test), pred)}')
		print(f'F1 score: {f1_score(torch.Tensor.tolist(y_test), pred, average="macro")}')

	return prob_list

def main():
	dataset_path = 'final/data/Arrhythmia Data Set/'
	clean = 'is_clean'
	pca = ['without_pca', 'using_pca']
	smote = ['using_SMOTETomek', 'using_smote_oversample', 'using_smote_undersample', 'using_SMOTEENN', 'without_smote']
	norm_data = ['MinMax', 'Z_score', 'None']

	# Get all combinations of options
	all_options = itertools.product(pca, smote, norm_data)

	# Iterate through all the combinations
	for option in all_options:
		pca, smote, norm_data = option
		print('--------------------Options--------------------')
		print(f'Clean: {clean}')
		print(f'PCA: {pca}')
		print(f'SMOTE: {smote}')
		print(f'Normalization: {norm_data}')

		train_data, _, train_label, _ =\
			dataProcessing(dataset_path, clean_data=clean, PCAMethod=pca, norm_data='None')

		train_data = train_data[1:]
		if norm_data == 'MinMax':
			scaler = MinMaxScaler()
			train_data = scaler.fit_transform(train_data)
		elif norm_data == 'Z_score':
			scaler = StandardScaler()
			train_data = scaler.fit_transform(train_data)

		train_label = train_label[1:]
		
		# Minus one from every label because when using crossentropy loss the starting label is 0
		train_label = [y - 1 for y in train_label]
		X = torch.Tensor(train_data)
		y = torch.Tensor(train_label).type(torch.LongTensor)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=train_label)

		if smote == 'using_smote_oversample':
			X_train, y_train = using_smote_oversample(X_train, y_train)
		elif smote == 'using_smote_undersample':
			X_train, y_train = using_smote_undersample(X_train, y_train)
		elif smote == 'using_SMOTEENN':
			X_train, y_train = using_SMOTEENN(X_train, y_train)
		elif smote == 'using_SMOTETomek':
			X_train, y_train = using_SMOTETomek(X_train, y_train)

		train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
		test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

		# Hyperparameters
		batch_size = train_data.shape[0]
		shuffle_data = True
		input_size = train_data.shape[1]
		hidden_size = 512
		output_size = 8
		num_epochs = 50

		model = FC(input_size, hidden_size, output_size)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)

		max_acc = 0
		lowest_val_loss = 999999

		data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_data)
		model_save = None

		# Training
		for epoch in range(num_epochs):
			model.train()
			# Iterate over the data_loader for training
			for batch in data_loader:
				inputs, labels = batch

		        # Forward pass
				outputs = model(inputs)

		        # Compute the loss
				loss = criterion(outputs, labels)

		        # Backward pass and optimization
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		    # Print the loss after each epoch
		    #print(f"Epoch {epoch + 1} / {num_epochs}, Loss: {loss.item():.4f}")

			model.eval()
			with torch.no_grad():
				acc = []
				val_loss = 0
				total_samples = 0
				for inputs, labels in test_dataset:
					outputs = model(inputs)
					_, predicted = torch.max(outputs, dim=0)
					if predicted == labels:
						acc.append(1)
					else:
						acc.append(0)

					loss = criterion(outputs, labels)
					val_loss += loss.item() * len(inputs)
					total_samples += len(inputs)

				pred_acc = np.array(acc).mean()
				if pred_acc > max_acc:
					max_acc = pred_acc
					max_acc_epoch = epoch + 1
		    		# Save the checkpoint that has the highest validation accuracy
					model_save = copy.deepcopy(model)
		    	#print(f'val acc: {pred_acc}')

				val_loss /= total_samples
				if lowest_val_loss > val_loss:
					lowest_val_loss = val_loss
					lowest_val_loss_epoch = epoch + 1

		    	#print(f'val loss: {val_loss:.4f}')

			scheduler.step(val_loss)

		print(f'Max acc: {max_acc} at {max_acc_epoch} epoch')
		print(f'Lowest vaidation loss: {lowest_val_loss} at {lowest_val_loss_epoch} epoch')

		# Save the best model to PATH
		PATH = f'final/algorithm/model/model_{clean}_{pca}_{smote}_norm_option_{norm_data}.pt'
		torch.save(model_save.state_dict(), PATH)
		predict(model_save, test_dataset, y_test)

if __name__ == "__main__":
	main()