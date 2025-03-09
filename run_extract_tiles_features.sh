#!/bin/bash
#BSUB -J "histolab"
#BSUB -P acc_tauomics
#BSUB -q premium
#BSUB -R rusage[mem=100G]
#BSUB -R span[hosts=1]
#BSUB -n 1
#BSUB -W 80:00
#BSUB -oo /sc/arion/projects/tauomics/Shrishtee/HistoLab/GRID_TILER/logs/HistoLab_out_%J.txt
#BSUB -eo /sc/arion/projects/tauomics/Shrishtee/HistoLab/GRID_TILER/logs/HistoLab_err_%J.txt

ml purge
ml proxies
ml anaconda3/2023.09
ml openslide

# source activate /sc/arion/projects/tauomics/jkauffman/hc_graph_exp/graph_env
source activate /hpc/users/kandos01/.conda/envs/graph_env

cd /sc/arion/projects/tauomics/Shrishtee/HistoLab/GRID_TILER/scripts

# Run the script

# python extract_tiles_and_features_March8.py
/hpc/users/kandos01/.conda/envs/graph_env/bin/python extract_features.py
