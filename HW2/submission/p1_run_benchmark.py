import sysfs_paths as sysfs
import subprocess
import time
import multiprocessing as mp
import measurement

def get_avail_freqs(cluster):
    """
    Obtain the available frequency for a cpu. Return unit in khz by default!
    """
    # Read cpu freq from sysfs_paths.py
    freqs = open(sysfs.fn_cluster_freq_range.format(cluster)).read().strip().split(' ')
    return [int(f.strip()) for f in freqs]


def get_cluster_freq(cluster_num):
    """
    Read the current cluster freq. cluster_num must be 0 (little) or 4 (big)
    """
    with open(sysfs.fn_cluster_freq_read.format(cluster_num), 'r') as f:
        return int(f.read().strip())


def set_user_space(clusters=None):
    """
    Set the system governor as 'userspace'. This is necessary before you can change the
    cluster/cpu freq to customized values
    """
    print("Setting userspace")
    clusters = [0, 4]
    for i in clusters:
        with open(sysfs.fn_cluster_gov.format(i), 'w') as f:
            f.write('userspace')


def set_cluster_freq(cluster_num, frequency):
    """
    Set customized freq for a cluster. Accepts frequency in khz as int or string
    """
    with open(sysfs.fn_cluster_freq_set.format(cluster_num), 'w') as f:
        f.write(str(frequency))

def start_measuring(filename, msg_queue):
    measurement.GoAheadAndMeasureSomeThingsForMe(filename, msg_queue).run()

if __name__ == '__main__':

    print('Available freqs for LITTLE cluster:', get_avail_freqs(0))
    print('Available freqs for big cluster:', get_avail_freqs(4))
    set_user_space()
    set_cluster_freq(4, 2000000)   # big cluster
    # print current freq for the big cluster
    print('Current freq for big cluster:', get_cluster_freq(4))

    stop_measurement = mp.Queue()
    filename = 'bodytrack_log.csv'
    measurement_process = mp.Process(target=start_measuring, args=(filename, stop_measurement))
    measurement_process.start()

    # execution of your benchmark    
    start=time.time()
    # run the benchmark
    command = "taskset --all-tasks 0xF0 parsec_files/bodytrack parsec_files/sequenceB_261 4 260 3000 8 3 4 0"
    proc_ben = subprocess.call(command.split())

    total_time = time.time() - start
    print("Benchmark runtime:", total_time)

    stop_measurement.put('please stop running ♥‿♥')
    measurement_process.join()
    