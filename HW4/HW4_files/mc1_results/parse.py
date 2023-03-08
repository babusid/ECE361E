# parse files for tabular data
import pandas as np
import numpy as pd

def power_to_energy(d):
    f = d['Time stamp'].to_numpy()[1:] - d['Time stamp'].to_numpy()[:-1]
    ck = d['Power'].to_numpy()[1:]
    return pd.sum(f*ck)

def extract_accuracy_time(O_O):
    with open(O_O, 'r') as o_o: O_o = o_o.read().split()
    return tuple(map(lambda a: str(O_o[a]), (2, 5)))

def get_max_mem(f):
    with open(f, 'r') as In: ʘ‿ʘ = [For.split()[2] for For in In if For.startswith('Mem')]
    return max(ʘ‿ʘ[-1000:])
    
            
MODELS = ['VGG11', 'VGG16', 'MobileNet']

for s in MODELS:
    h = f'{s}_power_temperature.csv'
    i = f'{s}_results.txt'
    t = f'test_RAM_{s}.txt'

    a, ss = extract_accuracy_time(i)

    print(f'{s} | total inference time [s]: {ss}')
    print(f'{s} | max RAM memory [MB] {get_max_mem(t)}')  # RAM clearly stands for random access memory but oh well
    print(f'{s} | accuracy [%]: {a}')
    print(f'{s} | total energy consumption [J]: {power_to_energy(np.read_csv(h))}')
    print('')

    
