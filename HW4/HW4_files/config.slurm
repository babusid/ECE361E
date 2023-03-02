#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Maverick2 GTX nodes
#----------------------------------------------------

#SBATCH -J HW4_g0                        # Job name
#SBATCH -o HW4_g0.o%j                    # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e HW4_g0.e%j                    # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gtx                           # Queue (partition) name
#SBATCH -N 1                             # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                             # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                      # Run time (hh:mm:ss)
#SBATCH --mail-user=<UT_MAIL_INSERT_HERE>@utexas.edu
#SBATCH --mail-type=all                  # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A EE361E                 # Allocation name

# Other commands must follow all #SBATCH directives...

module load python3/3.7.0 cuda/11.0 cudnn/8.0.5

source $WORK/HW4_virtualenv/bin/activate

# Launch code...

# ---------------------------------------------------

