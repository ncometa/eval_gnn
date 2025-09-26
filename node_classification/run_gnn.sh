#!/bin/bash

# This script runs node classification experiments for various GNNs
# on a specific list of datasets, with special handling for ogbn-arxiv.

# --- Configuration ---
# Use the first command-line argument as the GPU ID, or default to 0
GPU_ID=${1:-0}
echo "Running on GPU: $GPU_ID"

# --- Common Hyperparameters (from image) ---
HIDDEN_DIM=128
DROPOUT=0.2
EPOCHS=2000
LR=0.001
RUNS=10 # You can set the number of runs per experiment

# --- Datasets to Evaluate ---
# As per your request, the script will run on these specific datasets
DATASETS=("squirrel-filtered" "amazon-ratings" "questions" "ogbn-arxiv")

# --- Model-Specific Hyperparameters ---
MPNN_MODELS=("gcn" "gat" "sage")
MPNN_LAYERS_GENERAL=(1 3 4 5 7 10)
MPNN_LAYERS_ARXIV=(1 3 5 7)

FSGNN_LAYERS=(1 3 5)
FSGNN_FEAT_TYPES=("all" "homophily" "heterophily")

# --- Main Experiment Loop ---
for dataset in "${DATASETS[@]}"; do

    # --- Conditional Execution based on Dataset ---
    if [ "$dataset" == "ogbn-arxiv" ]; then
        # --- OGBN-ARXIV EXPERIMENTS ---
        echo "================================================================"
        echo "               Running Experiments for OGBN-ARXIV               "
        echo "================================================================"
        
        # Use main_arxiv.py for this dataset
        PYTHON_SCRIPT="node_classification/main_arxiv.py"
        CURRENT_MPNN_LAYERS=("${MPNN_LAYERS_ARXIV[@]}")

        # --- Run MPNN Models for ogbn-arxiv ---
        for model in "${MPNN_MODELS[@]}"; do
            for layers in "${CURRENT_MPNN_LAYERS[@]}"; do
                echo "----------------------------------------------------------------"
                echo "RUNNING: Dataset=$dataset, Model=$model, Layers=$layers"
                echo "----------------------------------------------------------------"
                python $PYTHON_SCRIPT \
                  --dataset "$dataset" --gnn "$model" --lr "$LR" \
                  --hidden_channels "$HIDDEN_DIM" --dropout "$DROPOUT" \
                  --local_layers "$layers" --epochs "$EPOCHS" \
                  --device "$GPU_ID" --runs "$RUNS" --bn --res
            done
        done
    else
        # --- EXPERIMENTS FOR OTHER DATASETS ---
        echo "================================================================"
        echo "        Running Experiments for $dataset        "
        echo "================================================================"

        # Use the standard main.py script
        PYTHON_SCRIPT="node_classification/main.py"
        CURRENT_MPNN_LAYERS=("${MPNN_LAYERS_GENERAL[@]}")

        # --- Run MPNN Models for general datasets ---
        for model in "${MPNN_MODELS[@]}"; do
            for layers in "${CURRENT_MPNN_LAYERS[@]}"; do
                echo "----------------------------------------------------------------"
                echo "RUNNING: Dataset=$dataset, Model=$model, Layers=$layers"
                echo "----------------------------------------------------------------"
                python $PYTHON_SCRIPT \
                  --dataset "$dataset" --gnn "$model" --lr "$LR" \
                  --hidden_channels "$HIDDEN_DIM" --dropout "$DROPOUT" \
                  --local_layers "$layers" --epochs "$EPOCHS" \
                  --device "$GPU_ID" --runs "$RUNS" --bn --res
            done
        done
    fi

    # --- FSGNN model loop runs for ALL specified datasets ---
    for layers in "${FSGNN_LAYERS[@]}"; do
        for feat_type in "${FSGNN_FEAT_TYPES[@]}"; do
            echo "----------------------------------------------------------------"
            echo "RUNNING: Dataset=$dataset, Model=fsgcn, Layers=$layers, FeatType=$feat_type"
            echo "----------------------------------------------------------------"
            python node_classification/main_fsgcn.py \
              --dataset "$dataset" --gnn "fsgcn" --lr "$LR" \
              --hidden_channels "$HIDDEN_DIM" --dropout "$DROPOUT" \
              --fsgcn_num_layers "$layers" --fsgcn_feat_type "$feat_type" \
              --epochs "$EPOCHS" --device "$GPU_ID" --runs "$RUNS"
        done
    done

done

echo "All specified node classification experiments finished."