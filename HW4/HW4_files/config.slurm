#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Maverick2 GTX nodes
#----------------------------------------------------

#SBATCH -J a                             # Job name
#SBATCH -o a.o%j                         # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e a.e%j                         # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gtx                           # Queue (partition) name
#SBATCH -N 1                             # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                             # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 04:00:00                      # Run time (hh:mm:ss)
#SBATCH --mail-user=thuang1@utexas.edu
#SBATCH --mail-type=all                  # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A ECE361E                 # Allocation name

# Other commands must follow all #SBATCH directives...

module load python3/3.7.0 cuda/11.0 cudnn/8.0.5

source $WORK/HW4_virtualenv/bin/activate

# TODO: change this if it's different
HW4_DIR=$WORK/ECE361E_HW1/HW4/HW4_files
MODELS_TO_RUN=("VGG11" "VGG16" "MobileNet")
# MODELS_TO_RUN=("VGG11")

for model in ${MODELS_TO_RUN[*]}; do
    echo "creating directory $HW4_DIR/$model ..."
    mkdir $HW4_DIR/$model/
done

# Launch code...
i=0
for model in ${MODELS_TO_RUN[*]}; do
    CUDA_VISIBLE_DEVICES=$i python $HW4_DIR/main_tf.py --model=$model > $HW4_DIR/$model/out.txt &
    ((i++))
done
wait

echo "done"

# ---------------------------------------------------

