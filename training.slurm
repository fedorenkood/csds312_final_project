#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=5g
#SBATCH -p gpuk40 --gres=gpu:1

#Copy the script and other input files to the scratch directory and change directory
cp train_models.py $PFSDIR
cd $PFSDIR

echo "running Python ..."
python train_models.py 

cp -r * $SLURM_SUBMIT_DIR
