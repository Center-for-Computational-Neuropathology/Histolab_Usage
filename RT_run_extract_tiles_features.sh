#!/bin/bash
#BSUB -J "random_tiler"
#BSUB -P acc_tauomics
#BSUB -q premium
#BSUB -R rusage[mem=50G]
#BSUB -R span[hosts=1]
#BSUB -n 4
#BSUB -W 40:00
#BSUB -oo /sc/arion/projects/tauomics/Shrishtee/HistoLab/GRID_TILER/logs/RandomTiler_out_%J.txt
#BSUB -eo /sc/arion/projects/tauomics/Shrishtee/HistoLab/GRID_TILER/logs/RandomTiler_err_%J.txt

ml purge
ml proxies
ml anaconda3/2023.09
ml openslide

# source activate /sc/arion/projects/tauomics/jkauffman/hc_graph_exp/graph_env
source activate /hpc/users/kandos01/.conda/envs/graph_env

# Path to your CSV file
CSV_FILE="/sc/arion/projects/tauomics/Shrishtee/HistoLab/GRID_TILER/csv_file/slide_json_paths.csv"
BASE_OUTPUT_DIR="/sc/arion/projects/tauomics/Shrishtee/HistoLab/RandomTiler_output"

# Create logs directory
mkdir -p "${BASE_OUTPUT_DIR}/logs"

# Read the CSV file line by line
while IFS=',' read -r slide_path json_path
do
    # Extract the slide ID (without extension)
    slide_id=$(basename "${slide_path}" .svs)
    
    # Create a slide-specific output directory
    slide_output_dir="${BASE_OUTPUT_DIR}/${slide_id}"
    mkdir -p "${slide_output_dir}"
    
    # Submit job for this slide
    bsub -J "tile_${slide_id}" \
         -P acc_tauomics \
         -q premium \
         -R "rusage[mem=100G]" \
         -R "span[hosts=1]" \
         -n 1 \
         -W 40:00 \
         -oo "${BASE_OUTPUT_DIR}/logs/tile_${slide_id}_%J.out" \
         -eo "${BASE_OUTPUT_DIR}/logs/tile_${slide_id}_%J.err" \
         "/hpc/users/kandos01/.conda/envs/graph_env/bin/python /sc/arion/projects/tauomics/Shrishtee/HistoLab/GRID_TILER/scripts/extract_features_2.py ${slide_path} ${json_path} ${slide_output_dir} random"

    echo "Submitted job for ${slide_id} with output to ${slide_output_dir}"
    
    # Optional: Add a small delay between job submissions
    sleep 1
    
done < "${CSV_FILE}"
