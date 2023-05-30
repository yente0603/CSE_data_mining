import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from dataProcess import *
import copy
from collections import Counter
from imblearn.over_sampling import SMOTE
import itertools

class FC(nn.Module):
	"""Fully connected neural network"""
	def __init__(self, input_size, hidden_size, output_size):
		super(FC, self).__init__()
		self.layer = nn.ParameterList([
			nn.Linear(input_size, hidden_size),
			nn.Linear(hidden_size, output_size)
		])
		self.activation = nn.LeakyReLU()

	def forward(self, x):
		for i in range(len(self.layer) - 1):
			x = self.layer[i](x)
			x = self.activation(x)

		x = self.layer[-1](x)
		return x

def predict(model, dataset):
	'''
	return: a dict of predictions with key being the prediction and value being the true label
	(every prediction contains 8 different values representing the predicted probabilities of each class)
	'''
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
			if predicted == labels:
				acc.append(1)
			else:
				acc.append(0)

			total_samples += len(inputs)

		pred_acc = np.array(acc).mean()
		#print(f'Accuracy: {pred_acc}')

	return prob_list

def without_smote(train_data, train_label):
	# Minus one from every label because when using crossentropy loss the starting label is 0
	train_label = [y - 1 for y in train_label]
	X = torch.Tensor(train_data)
	y = torch.Tensor(train_label).type(torch.LongTensor)

	return X, y

def main():

<<<<<<< HEAD
	dataset_path = 'Arrhythmia Data Set/'
=======
	dataset_path = '/final/algorithm/Arrhythmia Data Set/'
>>>>>>> b72c1af70d36b7c99d39bd216c3e7ef0062895de
	clean = 'is_clean'
	pca = 'without_pca'
	norm_data = 'None'

	train_data, test_data, train_label, test_label =\
		dataProcessing(dataset_path, clean_data=clean, PCAMethod=pca, norm_data=norm_data)

	print(test_label.shape)
	print(test_data.shape)

	print(test_label)
	print(test_data)

	X_test, y_test = without_smote(test_data, test_label)

	test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
	input_size = X_test.shape[1]
	hidden_size = 128
	output_size = 8

	# Inference
	model_path = 'model_is_clean_without_pca_using_smote_norm_option_None.pt'
	model = FC(input_size, hidden_size, output_size)
	model.load_state_dict(torch.load(model_path))
	predicted_probabilities = predict(model, test_dataset)

	acc = []

	for pred, label in predicted_probabilities.items():
		high_conf = np.argmax(pred, axis=0)
<<<<<<< HEAD
		print(["%0.2f" % i for i in pred])
		if label < 8:
			print(f'True:{label} Pred:{high_conf}', end='\n\n')
=======
		if label < 8:
			print(f'True:{label} Pred:{high_conf}')
>>>>>>> b72c1af70d36b7c99d39bd216c3e7ef0062895de
			if label == high_conf:
				acc.append(1)
			else:
				acc.append(0)
		else:
<<<<<<< HEAD
			high_conf = -1
			for i, p in enumerate(pred):
				if p >= 0.9:
					high_conf = i
			print(f'True:{label} Pred:{high_conf}', end='\n\n')
=======
			for i in pred:
				if i >= 0.9:
					high_conf = i
			high_conf = -1
			print(f'True:{label} Pred:{high_conf}')
>>>>>>> b72c1af70d36b7c99d39bd216c3e7ef0062895de
			if high_conf == -1:
				acc.append(1)
			else:
				acc.append(0)

	pred_acc = np.array(acc).mean()
	print(pred_acc)

<<<<<<< HEAD
	pred_acc = np.array(acc).mean()
	print(pred_acc)

=======
>>>>>>> b72c1af70d36b7c99d39bd216c3e7ef0062895de

if __name__ == "__main__":
	main()