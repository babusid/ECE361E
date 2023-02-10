import matplotlib.pyplot as plt
import numpy as np
import os

def lossacc_csv_to_dict(fname):
    print(f'processing {fname}...')
    data = {}
    with open(fname, 'r') as f:
        for line in f:
            tokens = line.split(',')
            data[tokens[0]] = list(map(int if tokens[0] == 'Epoch' else float, tokens[1:]))
    return data

def can_you_please_plot_the_following(data, dirname, title_suffix='', fname_suffix=''):
    print(f'gonna save these pngs to {dirname}...')
    for a in ('Loss', 'Acc'):
        b = a if a == 'Loss' else a + 'uracy'
        plt.figure(figsize=(12,5))
        for e in ((f'Train{a}', f'Training {b}'), (f'Test{a}', f'Testing {b}')):
            plt.plot(data['Epoch'], data[e[0]], '-o', label=e[1])
        plt.xticks(np.asarray(np.arange(1, data['Epoch'][-1])))
        plt.xlabel('Epoch')
        plt.ylabel(b)
        plt.title(f'{b} per Epoch{title_suffix}')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(dirname, f'{a.lower()}plot{fname_suffix}.png'))
        plt.close()

BASE_DIRECTORY = 'HW1_files'
DIRECTORY_NAMES = ('starter', 'simpleCNN_2', 'simpleFC')
for e in DIRECTORY_NAMES:
    dirname = os.path.join(BASE_DIRECTORY, e)
    filename = os.path.join(dirname, 'lossacc.csv')
    can_you_please_plot_the_following(lossacc_csv_to_dict(filename), dirname)
MORE_DIRECTORY_NAMES = ('simpleFC_2', 'simpleFC_3')
DROPOUT_VALUES = ('0.0', '0.2', '0.5', '0.8')
for e in MORE_DIRECTORY_NAMES:
    dirname = os.path.join(BASE_DIRECTORY, e)
    for d in DROPOUT_VALUES:
        filename = os.path.join(dirname, f'lossacc_{d}.csv')
        title_suffix = f', Dropout = {d}'
        can_you_please_plot_the_following(lossacc_csv_to_dict(filename), dirname, title_suffix, f'_0d{d[-1]}')
