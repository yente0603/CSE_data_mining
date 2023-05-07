from sklearn import ensemble, metrics
import matplotlib.pyplot as plt
import dataProcess
import numpy as np
import time


class RandomForest:
    def __init__(self, path=None, fig=None):
        self.path = path
        self.fig  = fig
        self.algorithm_name = "RandomForest"
    
    def updateAttributes(self, path, fig):
        self.path = path
        self.fig  = fig

    def randomForest(self, train_X, train_y, test_X, test_y, n_estimators, class_weight):
        forest = ensemble.RandomForestClassifier(n_estimators = n_estimators)
        forest.class_weight=class_weight
        forest.fit(train_X, train_y)
        test_y_predicted = forest.predict(test_X)
        acc = metrics.accuracy_score(test_y, test_y_predicted)
        return test_y_predicted, acc

    def run(self, n_estimators=100, clean_data=True, balance_data=True, show_result=True, class_weight={0: 1, 1: 1}):   
        X, Y, test_X, test_Y = dataProcess.dataProcessing(self.path, sample_type='over sampling', clean_data=clean_data, balance_data=balance_data)
        acc_list = []
        pred_list = []
        method_name = ('original', 'Min-Max', 'Z-score')
        y_pred, acc = self.randomForest(X, Y, test_X, test_Y, n_estimators, class_weight=class_weight)
        print(f"Run random forest by original              method / Accuracy: {acc:.4f}")
        acc_list.append(acc)
        pred_list.append(y_pred)

        norm_X, norm_test_X = dataProcess.norm(X, test_X)
        y_pred, acc = self.randomForest(norm_X, Y, norm_test_X, test_Y, n_estimators, class_weight=class_weight)
        print(f"Run random forest by Min-Max normalization method / Accuracy: {acc:.4f}")
        acc_list.append(acc)
        pred_list.append(y_pred)
        
        standard_X, standard_test_X = dataProcess.standard(X, test_X)
        y_pred, acc = self.randomForest(standard_X, Y, standard_test_X, test_Y, n_estimators, class_weight=class_weight)
        acc_list.append(acc)
        pred_list.append(y_pred)
        print(f"Run random forest by Z-score standard      method / Accuracy: {acc:.4f}")
        # x = excutePca(standard_X, 2)
        test_list = [test_X, norm_test_X, standard_test_X]
        t_x = dataProcess.excutePCA(test_list[np.argmax(acc_list)], 2)

        name = self.path.split('/')
        fig_title = 'RandomForest ' + name[-2] + f' / Accuracy: {max(acc_list):.4f} by {method_name[np.argmax(acc_list)]} method'
        print(f'\n{name[-2]} use {method_name[np.argmax(acc_list)]} method has the highest accurancy.')
        print(f'accuracy: {max(acc_list):.4f}\n')
        dataProcess.plot(t_x, test_Y, pred_list[np.argmax(acc_list)], fig_title, self.fig) if show_result else print("Don't plot the result\n")
        plot_data = [t_x, test_Y, pred_list[np.argmax(acc_list)]]
        return self.algorithm_name, name[-2], method_name[np.argmax(acc_list)], max(acc_list), acc_list, plot_data


def main():
    """
    Check path points to dataset
    use run function with parameter 
    path          :path points to dataset
    plot location :Where will the results be displayed on the diagram
    clean_data    :Whether to perform outlier detection on the dataset
    balance_data  :Whether to class balance the dataset
    class_weight  :class weight setting to balance unbalanced data
    """
    print('Random Forest algorithm\n')
    start_time = time.time()
    PATH_A = 'midterm/data/labA/'
    PATH_B = 'midterm/data/labB/'
    forest = RandomForest(PATH_A, [2, 1, 1])
    print('lab A')
    forest.run()
    forest.updateAttributes(PATH_B, [2, 1, 2])
    print('lab B')
    forest.run()
    print(f'Execution time: {time.time() - start_time:.4f} s')
    plt.show()

if __name__ == "__main__":
    main()