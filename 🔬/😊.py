
# nvidia-smi(ly face) extract uuid, memory used, and application memory
# we assume that application is using 1 gpu
def smi_ly_face(fname):
    print(f'parsing {fname} for gpu memory utilization')

    utilization = {}

    with open(fname, 'r') as f:
        for line in f:
            if not line.startswith('GPU'): continue

            tokens = line.split(',')
            if tokens[0] not in utilization.keys(): utilization[tokens[0]] = []
            utilization[tokens[0]].append(int(tokens[-1].split(' ')[1]))  # MiB

    memory_used = []
    for key in utilization:
        memory_used.append(max(utilization[key]))
    memory_used_app = max(memory_used) - min(memory_used)

    print(f'max application memory used is {memory_used_app} MiB')

smi_ly_face('HW1_files\starter\memcheck.txt')    
