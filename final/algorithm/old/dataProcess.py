import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # only for reduce dimension
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize, StandardScaler

def loadFile(path):
    train_data = np.genfromtxt(path + 'train_data.csv', delimiter=',')
    test_data = np.genfromtxt(path + 'test_data.csv', delimiter=',')
    train_label = np.genfromtxt(path + 'train_label.csv', delimiter=',')
    test_label = np.genfromtxt(path + 'test_label.csv', delimiter=',')
    return [train_data, test_data, train_label, test_label]

def dataClean(data):
    # missing value
    ratio = 0.3
    train_missing = np.where(np.isnan(data[0]))
    missing_index = np.where(np.bincount(train_missing[1])>=data[0].shape[0]*ratio)[0]
    for i in reversed(missing_index):
        for j in range(2):
            data[j] = np.delete(data[j], i, axis=1)
    imp_mean = SimpleImputer(missing_values=np.NaN, strategy='median')
    for i in range(2):
        imputer = imp_mean.fit(data[i])
        data[i] = imputer.transform(data[i])
        std_scaler = StandardScaler()
        data[i] = std_scaler.fit_transform(data[i]) 

    # outlier value
    for i in reversed([0, 2, 3]): #age, Height, Weight
        for j in range(2):
            outlier = np.where(data[j][:, i] == 0.0)[0]
            outlier = np.concatenate((outlier, np.where(data[j][:, i] >= 250.0)[0]))
            data[j] = np.delete(data[j], outlier, axis=0)
            data[j+2] = np.delete(data[j+2], outlier, axis=0)
    return data

def excutePCA(data, showPCA=False):
    # Implementation of PCA
    pca_dict = {}
    eigen_dict = {}
    for n_comp in range(data[0].shape[1]):
        pca = PCA(n_components=n_comp)
        temp_train_pca = pca.fit_transform(data[0])
        temp_test_pca = pca.transform(data[1])
        eigen_values = pca.explained_variance_[:n_comp]
        if n_comp > 0:
            pca_dict[n_comp] = pca.explained_variance_ratio_.sum()
            eigen_dict[n_comp] = eigen_values[-1]
    if showPCA:
        f = plt.figure(1)
        f.patch.set_facecolor('white')
        plt.title('PCA Variance')
        plt.xlabel('Principal Component Number')
        plt.ylabel('Variance Ratio')
        plt.plot(list(pca_dict.keys()),list(pca_dict.values()),'r')
        f.show()

        g = plt.figure(2)
        g.patch.set_facecolor('white')
        plt.title('PCA Eigen value')
        plt.xlabel('Principal Component Number')
        plt.ylabel('Eigen Values')
        plt.plot(list(eigen_dict.keys()),list(eigen_dict.values()),'r')
        g.show()
        g.savefig('pca.png')
    # Selecting components with Eigen value greater than 1 from the list
    pca_comp_eigen = max([key for key,val in eigen_dict.items() if val >= 1])
    pca_comp_eigen = max([key for key,val in pca_dict.items() if val < 0.95])

    print('Components from Feature selection using PCA (Having Eigen values >=1)- ' + str(pca_comp_eigen) + '')
    # Performing PCA for the train data with the fixed components
    pca = PCA(n_components=pca_comp_eigen)
    data[0] = pca.fit_transform(data[0])
    data[1] = pca.transform(data[1])
    print('Feature Selection using PCA complete for the train data.')
    return data


def dataProcessing(path, clean_data, PCAMethod, norm_data=None):
    print('===========================================================================')
    print('Loading datasets...')
    data = loadFile(path)
    org_train = data[0].shape
    org_test = data[1].shape
    print(f'The original training data have {org_train}.')
    print(f'The original testing  data have {org_test}.')
    if clean_data:
        data = dataClean(data)
        print('\nCleaning data...')
        print(f'clean original training data: {org_train} -> {data[0].shape}')
        print(f'clean original testing  data: {org_test} -> {data[1].shape}')
    org_train = data[0].shape
    org_test = data[1].shape
    if PCAMethod:
        data = excutePCA(data)
        print('\nPCA...')
        print(f'original training data: {org_train} -> {data[0].shape}')
        print(f'original testing  data: {org_test} -> {data[1].shape}')
    if norm_data in 'Min-Max':
        data = norm(data)
        print('\nMin-Max normalization method...')
    elif norm_data == 'Z-score':
        data = standard(data)
        print('\nZ-score normalization method...')
    print('===========================================================================')
    train_data = data[0]
    test_data = data[1]
    train_label = data[2]
    test_label = data[3]
    return train_data, test_data, train_label, test_label

def norm(data):
    for i in range(data[0].shape[1]):
        data[0][:,i] = (data[0][:,i] - np.min(data[0][:,i]))/(np.max(data[0][:,i]) - np.min(data[0][:,i]))
        data[1][:,i] = (data[1][:,i] - np.min(data[1][:,i]))/(np.max(data[1][:,i]) - np.min(data[1][:,i]))
    return data

def standard(data):
    train_mu = np.mean(data[0], axis=0)
    train_std = np.std(data[0], axis=0)
    test_mu = np.mean(data[1], axis=0)
    test_std = np.std(data[1], axis=0)
    data[0] = (data[0] - train_mu)/train_std
    data[1] = (data[1] - test_mu)/test_std
    return data

def main():
    path = 'final/data/Arrhythmia Data Set/'
    clean = True
    pca = True
    return dataProcessing(path, clean_data=clean, PCAMethod=pca, norm_data='Z-score')

if __name__ == "__main__":
    data = main()