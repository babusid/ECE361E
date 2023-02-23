import psutil
import telnetlib as tel
import sysfs_paths as sysfs
import time
import argparse


def get_telnet_power(telnet_connection, last_power):
    """
    Read power values using telnet.
    """
    # Get the latest data available from the telnet connection without blocking
    tel_dat = str(telnet_connection.read_very_eager())
    print('telnet reading:', tel_dat)
    # Find the latest power measurement in the data
    idx = tel_dat.rfind('\n')
    idx2 = tel_dat[:idx].rfind('\n')
    idx2 = idx2 if idx2 != -1 else 0
    ln = tel_dat[idx2:idx].strip().split(',')
    if len(ln) < 2:
        total_power = last_power
    else:
        total_power = float(ln[-2])
    return total_power


def get_cpu_load():
    """
    Returns the cpu load as a value from the interval [0.0, 1.0]
    """
    loads = [x / 100 for x in psutil.cpu_percent(interval=None, percpu=True)]
    return loads


def get_temps():
    """
    Obtain the temp values from sysfs_paths.py
    """
    templ = []
    # Get temp from temp zones 0-3 (the big cores)
    for i in range(4):
        temp = float(open(sysfs.fn_thermal_sensor.format(i), 'r').readline().strip()) / 1000
        templ.append(temp)
    # Note: on the Exynos5422, cpu temperatures 5 and 7 (big cores 1 and 3, counting from 0) appear to be swapped.
    # Therefore, swap them back.
    t1 = templ[1]
    templ[1] = templ[3]
    templ[3] = t1
    return templ


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


def get_core_freq(core_num):
    with open(sysfs.fn_cpu_freq_read.format(core_num), 'r') as f:
        return int(f.read().strip())


# solution - writing the temperature, power and cpu usage readings in a csv file 
class Computer___Show_Me_The_Power_And_The_Temperature():

    def __init__(self, filename, msg_queue):
        self.fname = filename
        self.q = msg_queue

    def run(self):

        with open(self.fname, 'w+') as csv_file:
            csv_file.write(f"Time stamp,Power,avg_temp\n")
            # measurement   
            telnet_connection = tel.Telnet("192.168.4.1")
            total_power = 0.0
            start_time = time.time()

            while self.q.empty():
                last_time = time.time()

                # system power
                total_power = get_telnet_power(telnet_connection, total_power)
                print('telnet power : ', total_power)

                # temp for big cores
                temps = get_temps()
                print('temp of big cores : ', temps)

                # average temp of big cores
                avg_temp = sum(temps) / len(temps)
                print('average temp : ', avg_temp)

                # writing the readings to the csv file 
                time_stamp = last_time
                csv_file.write(f"{time_stamp}, {total_power}, {avg_temp}\n")
                elapsed = time.time() - last_time
                DELAY = 0.2
                time.sleep(max(0., DELAY - elapsed))
