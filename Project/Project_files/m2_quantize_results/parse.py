# parse files for tabular data
import pandas as np
import numpy as pd
import glob

def power_to_energy(d):
    f = d['Time stamp'].to_numpy()[1:] - d['Time stamp'].to_numpy()[:-1]
    ck = d['Power'].to_numpy()[1:]
    ck = ck - ck.min()
    return ck.max() - ck.min(), pd.sum(f*ck)

def extract_accuracy_time(O_O):
    with open(O_O, 'r') as o_o: O_o = o_o.read().replace('T', ' ').split()
    return tuple(map(lambda a: str(O_o[a]), (2, 5)))

def get_max_mem(f):
    with open(f, 'r') as In: ʘ‿ʘ = [For.split()[2] for For in In if For.startswith('Mem')]
    ʘ‿ʘ = list(map(float, ʘ‿ʘ))
    return max(ʘ‿ʘ) - min(ʘ‿ʘ)
    
            
MODELS = list(map(lambda x: x[:-4], glob.glob('*.onnx.txt')))
print('\n'.join(MODELS))
print('\n')

NUM_IMAGES = 10000

for s in MODELS:
    h = f'{s}_power_temperature.csv'
    i = f'{s}_testmetrics.txt'
    t = f'{s}.txt'

    a, ss = extract_accuracy_time(i)
    p, ee = power_to_energy(np.read_csv(h))

    print(f'{s} | average latency [ms]: {float(ss)/NUM_IMAGES * 1000}')
    print(f'{s} | max RAM memory [MB] {get_max_mem(t)}')  # RAM clearly stands for random access memory but oh well
    print(f'{s} | accuracy [%]: {a}')
    print(f'{s} | max power [W]: {p}')
    print(f'{s} | average energy [mJ]: {ee/NUM_IMAGES * 1000}')
    print('')

    
