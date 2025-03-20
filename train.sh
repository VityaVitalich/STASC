#!/bin/bash

#SBATCH --job-name=llmcompr

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=V.Moskvoretskii@skoltech.ru

#SBATCH --output=train.txt
#SBATCH --time=0-10

#SBATCH --mem=100G

#SBATCH --nodes=1

#SBATCH -c 16

#SBATCH --gpus=2


srun singularity exec --bind /trinity/home/v.moskvoretskii/:/home -f --nv /trinity/home/v.moskvoretskii/images/stasc_math.sif bash -c "
    export HF_TOKEN=<>;
    export WANDB_API_KEY=<>;
    cd /home/STaSC/;
    nvidia-smi;
    pip list;
    sh bin/stasc.sh
"

