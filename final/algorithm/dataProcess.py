import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA # only for reduce dimension
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize, StandardScaler
def excutePCA(X, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def loadFile(path):
    # train_data = np.genfromtxt(path + 'train_data.csv', delimiter=',')[:, :15]
    train_data = np.genfromtxt(path + 'train_data.csv', delimiter=',')
    test_data = np.genfromtxt(path + 'test_data.csv', delimiter=',')
    train_label = np.genfromtxt(path + 'train_label.csv', delimiter=',')
    test_label = np.genfromtxt(path + 'test_label.csv', delimiter=',')
    return train_data, test_data, train_label, test_label

def dataClean(data):
    # missing value
    ratio = 0.4
    train_missing = np.where(np.isnan(data[0]))
    missing_index = np.where(np.bincount(train_missing[1])>=data[0].shape[0]*ratio)[0]
    for i in reversed(missing_index):
        for j in range(2):
            data[j] = np.delete(data[j], i, axis=1)
    imp_mean = SimpleImputer(missing_values=np.NaN, strategy='median')
    # imputer = imp_mean.fit(data[0])
    # df_imp = imputer.transform(data[0])
    # print(df_imp.shape)
    # print(data[0].shape)
    for i in range(2):
        imputer = imp_mean.fit(data[i])
        data[i] = imputer.transform(data[i])
        # std_scaler = StandardScaler()
        # data[i] = std_scaler.fit_transform(data[i]) 
    # print(data[0].shape)
    # df_data = pd.DataFrame(df_imp)
    
    # for i in range(2):
    #     missing_value = np.where(np.isnan(data[i]))[0]
    #     for j in reversed(missing_value):
    #         for k in range(2):
    #             data[i+k*2] = np.delete(data[i+k*2], j, axis=0)

    # outlier value
    for i in reversed([0, 2, 3]): #age, Height, Weight
        for j in range(2):
            outlier = np.where(data[j][:, i] == 0.0)[0]
            outlier = np.concatenate((outlier, np.where(data[j][:, i] >= 250.0)[0]))
            data[j] = np.delete(data[j], outlier, axis=0)
            data[j+2] = np.delete(data[j+2], outlier, axis=0)
    #balance
    # Implementation of PCA
    # pca_dict = {}
    # eigen_dict = {}
    # for n_comp in range(data[0].shape[1]):
    #     pca = PCA(n_components=n_comp)
    #     temp_train_pca = pca.fit_transform(data[0])
    #     temp_test_pca = pca.transform(data[1])
    #     eigen_values = pca.explained_variance_[:n_comp]
        
    #     if n_comp > 0:
    #         #print (n_comp,pca.explained_variance_ratio_.sum(),eigen_values)[-1])
    #         pca_dict[n_comp] = pca.explained_variance_ratio_.sum()
    #         eigen_dict[n_comp] = eigen_values[-1]
    # # Selecting components with Eigen value greater than 1 from the list
    # pca_comp_eigen = max([key for key,val in eigen_dict.items() if val >= 1])
    # pca_comp_eigen = max([key for key,val in pca_dict.items() if val < 0.95])

    # print('Components from Feature selection using PCA (Having Eigen values >=1)- ' + str(pca_comp_eigen) + '\n')

    # # Performing PCA for the train data with the fixed components
    # pca = PCA(n_components=pca_comp_eigen)
    # data[0] = pca.fit_transform(data[0])
    # data[1] = pca.transform(data[1])
    # print('Feature Selection using PCA complete for the train data.\n\n')
    
    return data

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

