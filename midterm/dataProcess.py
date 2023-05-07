import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA # only for reduce dimension

def excutePCA(X, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def loadFile(path):
    train_file = open(path + 'train_data.csv', 'r')
    test_file = open(path + 'test_data.csv', 'r')
    list_data = list(csv.reader(train_file))
    test_list_data = list(csv.reader(test_file))
    list_data.pop(0)
    test_list_data.pop(0)
    return np.array(list_data).astype(float), np.array(test_list_data).astype(float)

def dataClean(train_data, test_data):
    clean_idx = [1, 2, 3, 4, 5, 7]
    for i in clean_idx:
        train_data = np.delete(train_data, np.where(train_data[:, i] == 0.0)[0], axis=0)
        test_data = np.delete(test_data, np.where(test_data[:, i] == 0.0)[0], axis=0)
    return train_data, test_data

def dataBalance(train_data, test_data, sample_type):
    """
    find the class 0 & 1 of training data and test data retunring index.
    calculate the length to choose the method to use.
    """
    train_class_0_idx = np.where(train_data[:, -1] == 0)[0]
    train_class_1_idx = np.where(train_data[:, -1] == 1)[0]
    test_class_0_idx = np.where(test_data[:, -1] == 0)[0]
    test_class_1_idx = np.where(test_data[:, -1] == 1)[0]
    train_class = min(len(train_class_0_idx), len(train_class_1_idx)) if sample_type == 'under sampling' else max(len(train_class_0_idx), len(train_class_1_idx))
    test_class = min(len(test_class_0_idx), len(test_class_1_idx)) if sample_type == 'under sampling' else max(len(test_class_0_idx), len(test_class_1_idx))
    org_data = [len(train_class_0_idx), len(train_class_1_idx), len(test_class_0_idx), len(test_class_1_idx), train_class, test_class]

    if sample_type == 'under sampling':
        train_idx = np.sort(np.concatenate((train_class_0_idx[:train_class], 
                                            train_class_1_idx[:train_class])))
        test_idx = np.sort(np.concatenate((test_class_0_idx[:test_class], 
                                            test_class_1_idx[:test_class])))
    else:
        if train_class == len(train_class_0_idx): # class 0 is more
            sample = np.concatenate((train_class_1_idx, np.random.choice(train_class_1_idx, train_class-len(train_class_1_idx))))
            train_idx = np.sort(np.concatenate((train_class_0_idx, sample)))
            sample = np.concatenate((test_class_1_idx, np.random.choice(test_class_1_idx, test_class-len(test_class_1_idx))))
            test_idx = np.sort(np.concatenate((test_class_0_idx, sample)))
        else: # class 1 is more
            sample = np.concatenate((train_class_0_idx, np.random.choice(train_class_0_idx, train_class-len(train_class_0_idx))))
            train_idx = np.sort(np.concatenate((train_class_1_idx, sample)))
            sample = np.concatenate((test_class_0_idx, np.random.choice(test_class_0_idx, test_class-len(test_class_0_idx))))
            test_idx = np.sort(np.concatenate((test_class_1_idx, sample)))
    return train_data[train_idx], test_data[test_idx], org_data


def dataProcessing(path, sample_type, clean_data=True, balance_data=True):
    print('===========================================================================')
    print('Loading datasets...')
    train_data, test_data = loadFile(path)
    org_train = train_data.shape[0]
    org_test = test_data.shape[0]
    print(f'The original training data have {org_train:3d}.')
    print(f'The original testing  data have {org_test:3d}.')
    if clean_data:
        train_data, test_data = dataClean(train_data, test_data)
        print('\nCleaning data...')
        print(f'clean original training data: {org_train:3d} -> {train_data.shape[0]:3d}')
        print(f'clean original testing  data: {org_test:3d} -> {test_data.shape[0]:3d}')
    if balance_data:
        train_data, test_data, org_data = dataBalance(train_data, test_data, sample_type=sample_type)
        print(f'\nBalancing data by {sample_type}...')
        print(f'original training data class 0: {org_data[0]:3d} / class 1: {org_data[1]:3d}')
        print(f'original testing  data class 0: {org_data[2]:3d} / class 1: {org_data[3]:3d}')
        print(f'balanced training data class 0: {org_data[4]:3d} / class 1: {org_data[4]:3d}')
        print(f'balanced testing  data class 0: {org_data[5]:3d} / class 1: {org_data[5]:3d}')
    print('===========================================================================')
    X = train_data[:, :8]
    Y = train_data[:, -1]
    test_X = test_data[:, :8]
    test_Y = test_data[:, -1]
    return X, Y, test_X, test_Y

def plot(X, test_Y, y_pred, title, sub):
    '''
    using PCA dimensionality reduction data for visualizaiton
    Blue dot indicates prediction category of the test data is 0
    Red dot indicates prediction category of the test data is  1
    Purple dot indicates the prediction is wrong
    '''
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    plt.subplot(sub[0],sub[1],sub[2])
    plt.scatter(X[:, 0], X[:, 1], c=test_Y, cmap=cmap_bold)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.5, cmap=cmap_bold)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def norm(X, test_X):
    norm_X = X
    norm_test_X = test_X
    for i in range(X.shape[1]):
        norm_X[:,i] = (X[:,i] - np.min(X[:,i]))/(np.max(X[:,i]) - np.min(X[:,i]))
        norm_test_X[:,i] = (test_X[:,i] - np.min(test_X[:,i]))/(np.max(test_X[:,i]) - np.min(test_X[:,i]))
    return norm_X, norm_test_X

def standard(X, test_X):
    train_mu = np.mean(X, axis=0)
    train_std = np.std(X, axis=0)
    test_mu = np.mean(test_X, axis=0)
    test_std = np.std(test_X, axis=0)
    return (X - train_mu)/train_std, (test_X - test_mu)/test_std
