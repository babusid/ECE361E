#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Maverick2 GTX nodes
#----------------------------------------------------

SBATCH -J HW1_gr0                        # Job name
SBATCH -o HW1_gr0.o%j                    # Name of stdout output file (%j corresponds to the job id)
SBATCH -e HW1_gr0.e%j                    # Name of stderr error file (%j corresponds to the job id)
SBATCH -p gtx                            # Queue (partition) name
SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
SBATCH --mail-user=sidharth.n.babu@utexas.edu
SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
SBATCH -A ECE361E                        # Allocation name

# Other commands must follow all #SBATCH directives...

module load intel/18.0.2 python3/3.7.0
module load cuda/11.3 cudnn/8.2.1 nccl/2.9.9
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/11.3/lib64

# Environment code
source $WORK/HW1_virtualenv/bin/activate

# Launch code...
mkdir $WORK/HW1/HW1_files/starter # Problem 1
python $WORK/HW1/HW1_files/starter.py --batch_size=128 --epochs=25 --lr=0.01 > $WORK/HW1/HW1_files/starter/starter_out.txt

mkdir $WORK/HW1/HW1_files/simpleFC # Problem 2.1
python $WORK/HW1/HW1_files/simpleFC.py --batch_size=128 --epochs=25 --lr=0.01 > $WORK/HW1/HW1_files/simpleFC/simpleFC_out.txt

mkdir $WORK/HW1/HW1_files/simpleFC_2 # Problem 2.2
python $WORK/HW1/HW1_files/simpleFC_2.py --batch_size=128 --epochs=25 --lr=0.01 > $WORK/HW1/HW1_files/simpleFC2/simpleFC2_out.txt

mkdir $WORK/HW1/HW1_files/simpleFC_3 # Problem 2.3
python $WORK/HW1/HW1_files/simpleFC_3.py --batch_size=128 --epochs=25 --lr=0.01 > $WORK/HW1/HW1_files/simpleFC3/simpleFC3_out.txt

mkdir $WORK/HW1/HW1_files/simpleCNN # Problem 3
mkdir $WORK/HW1/HW1_files/simpleCNN/model_saves
python $WORK/HW1/HW1_files/simpleCNN.py --batch_size=128 --epochs=25 --lr=0.01 > $WORK/HW1/HW1_files/simpleCNN/simpleCNN_out.txt

mkdir $WORK/HW1/HW1_files/simpleCNN_2 # Problem 3.2
mkdir $WORK/HW1/HW1_files/simpleCNN_2/model_saves
python $WORK/HW1/HW1_files/simpleCNN_2.py --batch_size=128 --epochs=25 --lr=0.01 > $WORK/HW1/HW1_files/simpleCNN_2/simpleCNN_2_out.txt
# ---------------------------------------------------

