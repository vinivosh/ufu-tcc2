import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE_PATH = './comparisons.csv'
OUTPUT_FILE_PATH = './comparisonsGraph.png'

if __name__ == '__main__':
    print('\n############################## Plotting Graphs ##############################\n')

    with open(DATA_FILE_PATH, 'r') as file:
        data = pd.read_csv(file)

    # Uncomment the line below to not include the two largest datasets
    # data = data.drop(index=[3, 4])

    print(f'Data loaded:\n{data}\n\n')

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Tempo (s)')

    ax1.plot(data['dataset'], data['avg_cpu'], 'o-r', linewidth=2, label='CPU (Exec. Média)')
    ax1.plot(data['dataset'], data['avg_gpu'], 'o-g', linewidth=2, label='GPU (Exec. Média)')

    # ax1.plot(data['dataset'], data['slow_cpu'], 'x--r', label='CPU (Ex. mais Lenta)')
    # ax1.plot(data['dataset'], data['slow_gpu'], 'x--g', label='GPU (Exec. mais Lenta)')

    # ax1.plot(data['dataset'], data['fast_cpu'], '+--r', label='CPU (Exec. mais Rápida)')
    # ax1.plot(data['dataset'], data['fast_gpu'], '+--g', label='GPU (Exec. mais Rápida)')

    ax1.legend()

    # ax2 = ax1.twinx()

    # ax2.set_ylabel('#Instâncias', color='tab:green')
    # ax2.plot(data['dataset'], data['instances'], 'p-g')
    # # plt.yscale('log')

    fig.savefig(OUTPUT_FILE_PATH, bbox_inches='tight')