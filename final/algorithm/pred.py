import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from dataProcess import *
import itertools
from sklearn.metrics import f1_score, accuracy_score
from scipy.optimize import linear_sum_assignment

class FC(nn.Module):
	"""Fully connected neural network"""
	def __init__(self, input_size, hidden_size, output_size):
		super(FC, self).__init__()
		self.layer = nn.ParameterList([
			nn.Linear(input_size, hidden_size//2),
			nn.Linear(hidden_size//2, hidden_size//2),
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

def predict(model, dataset):
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
			if predicted == labels:
				acc.append(1)
			else:
				acc.append(0)

			total_samples += len(inputs)

		pred_acc = np.array(acc).mean()

	return prob_list

def kmeans(data, k, iter=100):
	centroids = data[np.random.choice(range(data.shape[0]), size=k, replace=False)]

	for _ in range(iter):
		# Assign each data point to the nearest centroid
		distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
		labels = np.argmin(distances, axis=0)

		# Update centroids by taking the mean of assigned data points
		for i in range(k):
			centroids[i] = data[labels == i].mean(axis=0)

		return labels, centroids

def separate_lists_indices(l):
    id1 = []
    id2 = []
    for i, li in enumerate(l):
        if li == -1:
            id2.append(i)
        else:
            id1.append(i)

    return id1, id2

def main():
	dataset_path = 'final/data/Arrhythmia Data Set/'
	clean = 'is_clean'
	pca = ['without_pca', 'using_pca']
	smote = ['without_smote', 'using_SMOTETomek', 'using_smote_oversample', 'using_smote_undersample', 'using_SMOTEENN']
	norm_data = ['MinMax', 'Z_score', 'None']
	threshold = [0.5, 0.75]

	# Get all combinations of options
	all_options = itertools.product(pca, smote, norm_data, threshold)
	acc_list = []
	i = 0
	output = open('final/output/output.txt', 'w')
	for option in all_options:
		pca, smote, norm_data, threshold = option
		# print('--------------------Options--------------------')
		# print(f'Clean: {clean}')
		# print(f'PCA: {pca}')
		# print(f'SMOTE: {smote}')
		# print(f'Normalization: {norm_data}')
		# print(f'Confidence threshold: {threshold}')

		_, test_data, _, test_label =\
			dataProcessing(dataset_path, clean_data=clean, PCAMethod=pca, norm_data='None')

		if norm_data == 'MinMax':
			scaler = MinMaxScaler()
			test_data = scaler.fit_transform(test_data)
		elif norm_data == 'Z_score':
			scaler = StandardScaler()
			test_data = scaler.fit_transform(test_data)

		test_label = [y - 1 for y in test_label]
		X_test = torch.Tensor(test_data)
		y_test = torch.Tensor(test_label).type(torch.LongTensor)

		test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
		input_size = X_test.shape[1]
		hidden_size = 512
		output_size = 8

		# Inference
		model_path = f'final/algorithm/model/model_{clean}_{pca}_{smote}_norm_option_{norm_data}.pt'
		model = FC(input_size, hidden_size, output_size)
		model.load_state_dict(torch.load(model_path))
		predicted_probabilities = predict(model, test_dataset)

		classify_pred = []
		prediction = []

		for pred, label in predicted_probabilities.items():
			pred_label = -1
			for i, p in enumerate(pred):
				if p > threshold:
					pred_label = i
			prediction.append(pred_label)
			if pred_label != -1:
				if pred_label == label:
					classify_pred.append(1)
				else:
					classify_pred.append(0)

		if len(classify_pred) == 0:
			classify_acc = 0
		else:
			classify_acc = np.array(classify_pred).mean()
		# print(f'Original 8 class accuracy: {classify_acc}')

		_, cluster_index = separate_lists_indices(prediction)
		data_to_be_clustered = test_data[cluster_index]

		# Use kmeans on other classes
		if data_to_be_clustered.shape[0] >= 5:
			predicted_labels, centroids = kmeans(data_to_be_clustered, 2)
			# predicted_labels, _ = kmeans2(data_to_be_clustered, 5, 100)

			predicted_labels = np.array(predicted_labels, dtype='int64')
			ground_truth = np.array(test_label, dtype='int64')[cluster_index]

			num_classes_ground_truth = np.max(ground_truth) + 1
			num_classes_predicted = np.max(predicted_labels) + 1

			confusion_matrix = np.zeros((num_classes_ground_truth, num_classes_predicted), dtype=int)

			for true_label, predicted_label in zip(ground_truth, predicted_labels):
				confusion_matrix[true_label, predicted_label] += 1

			# Apply the Hungarian algorithm
			row_ind, col_ind = linear_sum_assignment(confusion_matrix.max() - confusion_matrix)

			# Use the optimal assignment to remap the predicted labels:
			remapped_labels = np.zeros_like(predicted_labels)
			for true_label, predicted_label in zip(row_ind, col_ind):
				remapped_labels[predicted_labels == predicted_label] = true_label

			cluster_accuracy = []
			for i, label in enumerate(remapped_labels):
				if label > 8 and label == ground_truth[i]:
					cluster_accuracy.append(1)
				else:
					cluster_accuracy.append(0)

			cluster_accuracy = np.array(cluster_accuracy).mean()
		else:
			cluster_accuracy = 0
		acc_list.append(cluster_accuracy)
		# print("Clustering accuracy:", cluster_accuracy)
		# if cluster_accuracy > 0.4:
		# 	print(f'path:{model_path}')
		# 	print("Clustering accuracy:", cluster_accuracy)
		total_acc = classify_acc * len(classify_pred) / len(test_label) + cluster_accuracy * (len(test_label) - len(classify_pred)) / len(test_label)
		output.write(f'model : {clean}_{pca}_{smote}_norm_option_{norm_data}\nAcc : {total_acc*100:.2f}%\n\n')
		if total_acc > 0.35:
			print(f'model : {clean}_{pca}_{smote}_norm_option_{norm_data}\nAcc : {total_acc*100:.2f}%')
		# print(f'Overall accuracy: {classify_acc * len(classify_pred) / len(test_label) + cluster_accuracy * (len(test_label) - len(classify_pred)) / len(test_label)}', end='\n\n')
		i += 1
	# print(f'max acc = {max(acc_list)*100:.2f} index = {np.argmax(acc_list)}')
if __name__ == "__main__":
	import classifier
	classifier.main()
	main()