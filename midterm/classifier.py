import knn
import forest
import matplotlib.pyplot as plt
import time
import numpy as np
import dataProcess

def excuteOneAlgorithm(algorithm, path, show=False, balance_data=True):
    data_name = path.split('/')[-2]
    print(f'Excute {algorithm.algorithm_name} algorithm...\n')
    print(f'datasets from {data_name}')
    algorithm.updateAttributes(path, [1, 1, 1])
    return algorithm.run(show_result=False, balance_data=balance_data)

def excuteAllAlgorithm(path):
    data_name = path.split('/')[-2]
    print(f'Excute all algorithm...\n')
    print(f'datasets from {data_name}')
    output_data = []
    acc_list = []
    for i, algorithm in enumerate(algorithm_list):
        algorithm.updateAttributes(path, [2, 1, i+1])
        output_data.append(algorithm.run())
        acc_list.append(output_data[i][-1])
    best_acc = np.argmax(acc_list)
    print(f'\n{output_data[best_acc][0]} algorithm in {output_data[best_acc][1]} use {output_data[best_acc][2]} method has the highest accurancy.')
    print(f'accuracy: {max(acc_list):.4f}\n')

def excuteAllAlgorithmWithAllPath(clean_data, balance_data):
    print(f'Excute all algorithm with all datasets...')
    output_data = []
    acc_list = []
    count = 0
    for path in PATH_list:
        for algorithm in algorithm_list:
            print(f'{algorithm.algorithm_name} algorithm - {path.split("/")[-2]}')
            algorithm.updateAttributes(path, [2, 2, count+1])
            output_data.append(algorithm.run(clean_data=clean_data[count], balance_data=balance_data[count]))
            acc_list.append(output_data[count][-1])
            count += 1
    best_acc = np.argmax(acc_list)
    print(f'\n{output_data[best_acc][0]} algorithm in {output_data[best_acc][1]} use {output_data[best_acc][2]} method has the highest accurancy.')
    print(f'accuracy: {max(acc_list):.4f}\n')

def excuteAlgorithm(epoch):
    method_name = ('original', 'Min-Max', 'Z-score')
    result_txt = open('midterm/output/output_data.txt', 'w')
    data = np.array(([[None, None, None, None], [None, None, None, None]]))
    for k in range(2):
        balance_data = True if k == 0 else False
        count = 0
        for algorithm in algorithm_list:
            for p in PATH_list:
                ori_acc_list = []
                Min_Max_acc = []
                Z_acc = []
                acc_list = [ori_acc_list, Min_Max_acc, Z_acc]
                max_plot_data = [None, None, None]
                for i in range(epoch):
                    result = excuteOneAlgorithm(algorithm, path=p, balance_data=balance_data)
                    acc = result[-2]
                    for j in range(3):
                        acc_list[j].append(acc[j])
                        if acc[j] == max(acc_list[j]):
                            max_plot_data[j] = result[-1]
                data[k][count] = [result[0], p.split('/')[-2], acc_list, max_plot_data, balance_data]
                count += 1

    for k in range(2):
        for i in range(4):
            balance = 'with' if data[k][i][-1] == True else 'without'
            print(f'{data[k][i][0]} algorithm in {data[k][i][1]} {balance} data balance repeat {epoch} times')
            print(f'1. original data\naverage accuracy: {np.mean(data[k][i][2][0])*100:.2f}% / max accuracy: {max(data[k][i][2][0])*100:.2f}%')
            print(f'2. Min-Max normalization\naverage accuracy: {np.mean(data[k][i][2][1])*100:.2f}% / max accuracy: {max(data[k][i][2][1])*100:.2f}%')
            print(f'3. Z-score standard\naverage accuracy: {np.mean(data[k][i][2][2])*100:.2f}% / max accuracy: {max(data[k][i][2][2])*100:.2f}%')
            print('===========================================================================')
            result_txt.write(f'{data[k][i][0]} algorithm in {data[k][i][1]} {balance} data balance repeat {epoch} times\n')
            result_txt.write(f'1. original data\naverage accuracy: {np.mean(data[k][i][2][0])*100:.2f}% / max accuracy: {max(data[k][i][2][0])*100:.2f}%\n')
            result_txt.write(f'2. Min-Max normalization\naverage accuracy: {np.mean(data[k][i][2][1])*100:.2f}% / max accuracy: {max(data[k][i][2][1])*100:.2f}%\n')
            result_txt.write(f'3. Z-score standard\naverage accuracy: {np.mean(data[k][i][2][2])*100:.2f}% / max accuracy: {max(data[k][i][2][2])*100:.2f}%\n')
            result_txt.write('===========================================================================\n')
            plt.clf()
            plt.figure(figsize=(12, 6), num='accurancy_output')
            for j in range(3):
                plt.subplot(3, 1, j+1)
                plt.plot(np.round(data[k][i][2][j], 2), 'b-')
                plt.title(f'{data[k][i][0]} algorithm use {method_name[j]} in {data[k][i][1]} {balance} data balance repeat {epoch} times with max accuracy')
                # plt.ylim((60, 100))
                plt.xlabel('epoch')
                plt.ylabel('accurancy (%)')
                plt.tight_layout()
            plt.savefig(f'midterm/output/{k}{i}_accurancy.png')


            plt.clf()
            plt.figure(figsize=(12, 12), num='classifier_output')
            for j in range(3):
                title = f'{data[k][i][0]} algorithm use {method_name[j]} method in {data[k][i][1]} {balance} data balance repeat {epoch} times with max accuracy'
                dataProcess.plot(data[k][i][-2][j][0], data[k][i][-2][j][1], data[k][i][-2][j][2], title, [3, 1, j+1])
            plt.savefig(f'midterm/output/{k}{i}_classifier.png')
    


def main():
    start_time = time.time()
    # clean_data = [True, True, True, True]
    # balance_data = [False, False, False, False]
    # plt.figure(figsize=(12, 6), num='output')
    # excuteAllAlgorithmWithAllPath(clean_data, balance_data)
    # excuteAllAlgorithm(path=PATH_A)
    excuteAlgorithm(1)
    print(f'Execution time: {time.time() - start_time:.4f} s')
    # plt.tight_layout()
    # plt.show()

PATH_A = 'midterm/data/labA/'
PATH_B = 'midterm/data/labB/'


PATH_list = [PATH_A, PATH_B]
algorithm_list = [knn.KNN(), forest.RandomForest()]

if __name__ == '__main__':
    main()