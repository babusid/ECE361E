#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Maverick2 GTX nodes
#----------------------------------------------------

#SBATCH -J HW3_gr0                        # Job name
#SBATCH -o HW3_gr0.o%j                    # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e HW3_gr0.e%j                    # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gtx                            # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=<UT_MAIL_INSERT_HERE>@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A ECE361E                        # Allocation name

# Other commands must follow all #SBATCH directives...

module load intel/18.0.2 python3/3.7.0
module load cuda/11.3 cudnn/8.2.1 nccl/2.9.9
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/11.3/lib64

# Environment code
source $WORK/HW3_virtualenv/bin/activate

# Launch code...
python $WORK/HW3_files/starter.py --batch_size=128 --epochs=100 > $WORK/HW3_files/out

# ---------------------------------------------------

