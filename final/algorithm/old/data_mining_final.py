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

def using_smote(train_data, train_label):
	# Extract the eighth data from dataset first as we cannot apply SMOTE on a class that contains only one data point
    # We'll duplicate this data to the same size as the data in other classes
	eighth_data = train_data[0]
	eighth_label = train_label[0]
	train_data = train_data[1:]
	train_label = train_label[1:]

	# Minus one from every label because when using crossentropy loss the starting label is 0
	train_label = [y - 1 for y in train_label]

	sm = SMOTE(random_state=42, k_neighbors=2)
	X_res, y_res = sm.fit_resample(train_data, train_label)

	# Count the max number of elements in one class for duplicating eighth class element later
	counter = Counter(y_res)
	sorted_counts = counter.most_common()
	max_count = sorted_counts[0][1]

	X_res = torch.cat((torch.Tensor(X_res), torch.Tensor([eighth_data for _ in range(max_count)])))
	y_res = torch.cat((torch.Tensor(y_res), torch.Tensor([eighth_label - 1 for _ in range(max_count)]))).type(torch.LongTensor)

	return X_res, y_res

def without_smote(train_data, train_label):
	# Minus one from every label because when using crossentropy loss the starting label is 0
	train_label = [y - 1 for y in train_label]
	X = torch.Tensor(train_data)
	y = torch.Tensor(train_label).type(torch.LongTensor)

	return X, y

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
		print(f'Accuracy: {pred_acc}')

	return prob_list

def classifier(is_inference=False):

	dataset_path = 'final/data/Arrhythmia Data Set/'
	clean = 'is_clean'
	pca = ['using_pca', 'without_pca']
	smote = ['using_smote', 'without_smote']
	norm_data = ['None']
	# is_inference = True

	# Get all combinations of options
	all_options = itertools.product(pca, smote, norm_data)

	if not is_inference:
		# Iterate through all the combinations
		for option in all_options:
			pca, smote, norm_data = option
			print('--------------------Options--------------------')
			print(f'Clean: {clean}')
			print(f'PCA: {pca}')
			print(f'SMOTE: {smote}')
			print(f'Normalization: {norm_data}')

			train_data, _, train_label, _ =\
				dataProcessing(dataset_path, clean_data=clean, PCAMethod=pca, norm_data=norm_data)

			if smote == 'using_smote':
				X, y = using_smote(train_data, train_label)
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)
			else:
				X, y = without_smote(train_data, train_label)
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

			train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
			test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

			# Hyperparameters
			batch_size = 128
			shuffle_data = True
			input_size = train_data.shape[1]
			hidden_size = 128
			output_size = 8
			num_epochs = 30

			model = FC(input_size, hidden_size, output_size)
			criterion = nn.CrossEntropyLoss()
			optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)
			scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=1/3)

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
			    	#print(f'val acc: {pred_acc}')

					val_loss /= total_samples
					if lowest_val_loss > val_loss:
						lowest_val_loss = val_loss
						lowest_val_loss_epoch = epoch + 1
						# Save the checkpoint that has the lowest validation loss
						model_save = copy.deepcopy(model)

			    	#print(f'val loss: {val_loss:.4f}')

				scheduler.step(val_loss)

			print(f'Max acc: {max_acc*100:.2f}% at {max_acc_epoch} epoch')
			print(f'Lowest vaidation loss: {lowest_val_loss:.4f} at {lowest_val_loss_epoch} epoch')

			PATH = f'model_{clean}_{pca}_{smote}_norm_option_{norm_data}.pt'
			# Save the best model to PATH
			torch.save(model_save.state_dict(), PATH)

			with torch.no_grad():
				acc = []
				pred = []
				val_loss = 0
				total_samples = 0
				for inputs, labels in test_dataset:
					outputs = model(inputs)
					_, predicted = torch.max(outputs, dim=0)

					pred.append(predicted.item())
					if predicted == labels:
						acc.append(1)
					else:
						acc.append(0)

					val_loss += loss.item() * len(inputs)
					total_samples += len(inputs)

				pred_acc = np.array(acc).mean()
				print(f'final acc: {pred_acc*100:.2f}%')

	else:
		# Preparing data, using the above test set as example
		train_data, test_data, train_label, test_label =\
			dataProcessing(dataset_path, clean_data=clean, PCAMethod='without_pca', norm_data='None')
		X, y = using_smote(train_data, train_label)
		_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)
		test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
		test_data = torch.from_numpy(test_data).float()
		test_label = torch.from_numpy(test_label).long()
		# test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
		input_size = X_test.shape[1]
		hidden_size = 128
		output_size = 8

		# Inference
		model_path = 'model_is_clean_without_pca_using_smote_norm_option_None.pt'
		model = FC(input_size, hidden_size, output_size)
		model.load_state_dict(torch.load(model_path))
		predicted_probabilities = predict(model, test_dataset)
		
		for index, (pred, label) in enumerate(predicted_probabilities.items()):
			if np.argmax(pred, axis=0) == label:
				# print(f'{label}: {["%0.2f" % i for i in pred]} right prediction')
				pass
			else:
				# print(pred)
				print(f'{label}: {["%0.2f" % i for i in pred]} wrong prediction')

def main():
	classifier(is_inference=True)

if __name__ == "__main__":
	main()