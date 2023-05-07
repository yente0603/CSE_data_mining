import numpy as np
import matplotlib.pyplot as plt # plot
import dataProcess
import time

class KNearestNeighbor:
    """
    KNN algorithm
    K value must be given when creating an algorithm object.
    k must be positive integer.
    """
    def __init__(self, k):
        self.k = range(1, k)
        
    def train(self, X, Y): # input training dataset and label
        self.X_train = X
        self.Y_train = Y

    def predict(self, X_test): # Calculate test data distance and predict classification
        distances = self.compute_distance(X_test)
        return self.predict_labels(distances)

    def compute_distance(self, X_test): 
        distances = []
        [distances.append(np.sqrt(np.sum((X_test[i]-self.X_train) ** 2, axis=1))) for i in range(len(X_test))]
        return np.array(distances)

    def predict_labels(self, distances):
        num_test = distances.shape[0]
        pred_list = []
        for neighbor in self.k:
            y_pred = np.zeros(num_test)
            for i in range(num_test):
                y_indices = np.argsort(distances[i, :])
                k_closest_classes = self.Y_train[y_indices[:neighbor]].astype(int)
                y_pred[i] = np.argmax(np.bincount(k_closest_classes))
            pred_list.append(y_pred)
        return pred_list


class KNN(KNearestNeighbor):
    def __init__(self, path=None, fig=None, k=30):
        super().__init__(k)
        self.K_value = k
        self.path = path
        self.fig  = fig
        self.algorithm_name = "KNN"

    def updateAttributes(self, path, fig):
        self.path = path
        self.fig  = fig

    def excuteKnn(self, train_x, train_y, test_x, test_y, k):
        pred_list = []
        self.train(train_x, train_y)
        y_pred = self.predict(test_x)
        for i in range(self.K_value-1):
            pred_list.append(sum(y_pred[i] == test_y) / test_y.shape[0]) #accurancy
        best_k = np.argmax(pred_list) + 1
        best_pred_list = y_pred[np.argmax(pred_list)]
        acc = max(pred_list)
        return best_k, acc, best_pred_list

    def run(self, clean_data=True, balance_data=True, show_result=True):
        X, Y, test_X, test_Y = dataProcess.dataProcessing(self.path, sample_type='under sampling', clean_data=clean_data, balance_data=balance_data)
        acc_list = []
        k_list = []
        pred_list = []
        method_name = ('original', 'Min-Max', 'Z-score')
        best_k, acc, best_pred = self.excuteKnn(X, Y, test_X, test_Y, k=self.k)
        print(f'Each method search best k value in 1 ~ {self.K_value}')
        print(f"Run KNN by original              method / k value = {best_k:2d} Accuracy: {acc:.4f}")
        acc_list.append(acc)
        k_list.append(best_k)
        pred_list.append(best_pred)

        norm_X, norm_test_X = dataProcess.norm(X, test_X)
        best_k, acc, best_pred = self.excuteKnn(norm_X, Y, norm_test_X, test_Y, k=self.k)
        print(f"Run KNN by Min-Max normalization method / k value = {best_k:2d} Accuracy: {acc:.4f}")
        acc_list.append(acc)
        k_list.append(best_k)
        pred_list.append(best_pred)
        
        standard_X, standard_test_X = dataProcess.standard(X, test_X)
        best_k, acc, best_pred = self.excuteKnn(standard_X, Y, standard_test_X, test_Y, k=self.k)
        print(f"Run KNN by Z-score standard      method / k value = {best_k:2d} Accuracy: {acc:.4f}")
        acc_list.append(acc)
        k_list.append(best_k)
        pred_list.append(best_pred)

        # x = excutePca(standard_X, 2)
        test_list = [test_X, norm_test_X, standard_test_X]
        t_x = dataProcess.excutePCA(test_list[np.argmax(acc_list)], 2) #reduce to 2D
        name = self.path.split('/')
        fig_title = 'KNN ' + name[-2] + f' / k = {max(k_list)} Accuracy: {max(acc_list):.4f} by {method_name[np.argmax(acc_list)]} method'
        print(f'\n{name[-2]} use {method_name[np.argmax(acc_list)]} method has the highest accurancy.')
        print(f'accuracy: {max(acc_list):.4f} / k = {k_list[np.argmax(acc_list)]}\n')
        dataProcess.plot(t_x, test_Y, pred_list[np.argmax(acc_list)], fig_title, self.fig) if show_result else print("Don't plot the result\n")
        plot_data = [t_x, test_Y, pred_list[np.argmax(acc_list)]]
        return self.algorithm_name, name[-2], method_name[np.argmax(acc_list)], max(acc_list), acc_list, plot_data

 
def main():
    """
    Check path points to dataset
    use run function with these parameter 
    path          : path points to dataset
    plot location : Where will the results be displayed on the diagram
    k             : K value iteration size
    clean_data    : Whether to perform outlier detection on the dataset
    balance_data  : Whether to class balance the dataset
    """
    path_A = 'midterm/data/labA/'
    path_B = 'midterm/data/labB/'
    print('KNN algorithm\n')
    start_time = time.time()
    print('lab A')
    knn = KNN(path_A, [2, 1, 1], k=30)
    knn.run()
    print('lab B')
    knn.updateAttributes(path_B, [2, 1, 2])
    knn.run()
    print(f'Execution time: {time.time() - start_time:.4f} s')
    plt.show()

if __name__ == "__main__":
    main()