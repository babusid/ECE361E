import telnetlib as tel
import sysfs_paths as sysfs
import time
import argparse


def get_telnet_power(SP2_tel, last_power):
    """
    read power values using telnet.
    """
    # Get the latest data available from the telnet connection without blocking
    tel_dat = str(SP2_tel.read_very_eager())
    print('telnet reading:', tel_dat)
    # find latest power measurement in the data
    findex = tel_dat.rfind('\n')
    findex2 = tel_dat[:findex].rfind('\n')
    findex2 = findex2 if findex2 != -1 else 0
    ln = tel_dat[findex2:findex].strip().split(',')
    if len(ln) < 2:
        total_power = last_power
    else:
        total_power = float(ln[-2])
    return total_power


def get_temps():
    """
    obtain the temp values from sysfs_paths.py
    """
    templ = []
    # get temp from temp zones 0-3 (the big cores)
    for i in range(4):
        temp = float(open(sysfs.fn_thermal_sensor.format(i),
                     'r').readline().strip())/1000
        templ.append(temp)
        # Note: on the 5422, cpu temperatures 5 and 7 (big cores 1 and 3, counting from 0) appear to be swapped. Therefore, swap them back.
    t1 = templ[1]
    templ[1] = templ[3]
    templ[3] = t1
    return templ


def get_avail_freqs(cpu_num):
    """
    obtain the available frequency for a cpu. Return unit in khz by default!
    """
    # map from a cpu to its cluster (big or little)
    cluster = (cpu_num//4) * 4
    # read cpu freq from sysfs_paths.py
    freqs = open(sysfs.fn_cluster_freq_range.format(
        cluster)).read().strip().split(' ')
    return [int(f.strip()) for f in freqs]


def get_cluster_freq(cluster_num):
    """
    read the current cluster freq. cluster_num must be 0 (little) or 4 (big)
    """
    with open(sysfs.fn_cluster_freq_read.format(cluster_num), 'r') as f:
        return int(f.read().strip())


def get_core_freq(core_num):
    with open(sysfs.fn_cpu_freq_read.format(core_num), 'r') as f:
        return int(f.read().strip())

# solution - writing the temperature, power and cpu usage readings in a csv file


parser = argparse.ArgumentParser(
    description='temperature and power measurement code')
# added one argument to specify the location of csv file to store the output
parser.add_argument('--csv_loc', type=str, default='VGG11_readings.csv',
                    help='csv file to store the temperature and power measurements')
args = parser.parse_args()

with open(args.csv_loc, 'w+') as csv_file:
    csv_file.write(f"Time stamp,Power,avg_temp\n")
    # measurement
    SP2_tel = tel.Telnet("192.168.4.1")
    total_power = 0.0
    start_time = time.time()

    while True:
        last_time = time.time()

        # system power
        total_power = get_telnet_power(SP2_tel, total_power)
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
        time.sleep(max(0, DELAY - elapsed))
