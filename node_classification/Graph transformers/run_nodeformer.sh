


#!/bin/bash

echo "=================================================="
echo "      🚀 Starting nodeformer Grid Search 🚀"
echo "=================================================="
echo ""

# Define the GPU device ID from the first script argument, default to 0
DEVICE=${1:-0}
echo "Running on GPU: $DEVICE"

# Define the metrics to iterate over
METRICS="rocauc acc prauc balacc"

# --- Block 1: Prioritized Datasets (Minesweeper & Squirrel) ---
echo "=================================================="
echo "  🏃 Kicking off prioritized datasets first..."
echo "=================================================="

for metric in $METRICS
do
for head in 2
do
for layer in 1 5
do
for hidden_channels in 128
do
for dropout in 0.2
do

echo ""
echo "--- Running [Metric: $metric, Layers: $layer, Hidden: $hidden_channels, Dropout: $dropout, Heads: $head] ---"

# --- Minesweeper ---
echo "  -> Training on Minesweeper"
python main_nodeformer.py --dataset minesweeper --hidden_channels $hidden_channels --epochs 2000 --lr 0.01 --runs 10 --local_layers $layer --dropout $dropout --num_heads $head --metric $metric --ln --res --device $DEVICE

# --- Squirrel-filtered ---
echo "  -> Training on Squirrel"
python main_nodeformer.py --dataset squirrel --hidden_channels $hidden_channels --epochs 1500 --lr 0.01 --runs 10 --local_layers $layer --dropout $dropout --num_heads $head --weight_decay 5e-4 --metric $metric --ln --res --device $DEVICE

done
done
done
done
done


# --- Block 2: Remaining Datasets ---
echo ""
echo "=================================================="
echo "  👍 Prioritized runs complete. Starting others..."
echo "=================================================="

for metric in $METRICS
do
for head in 2
do
for layer in 1 5
do
for hidden_channels in 128
do
for dropout in 0.2
do

echo ""
echo "--- Running [Metric: $metric, Layers: $layer, Hidden: $hidden_channels, Dropout: $dropout, Heads: $head] ---"

# --- Amazon Ratings ---
echo "  -> Training on Amazon Ratings"
python main_nodeformer.py --dataset amazon-ratings --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 10 --local_layers $layer --dropout $dropout --num_heads $head --metric $metric --ln --res --device $DEVICE

# --- Questions ---
echo "  -> Training on Questions"
python main_nodeformer.py --dataset questions --hidden_channels $hidden_channels --epochs 1500 --lr 3e-5 --runs 10 --local_layers $layer --dropout $dropout --num_heads $head --metric $metric --ln --res --device $DEVICE

# --- Tolokers ---
echo "  -> Training on Tolokers"
python main_nodeformer.py --dataset tolokers --hidden_channels $hidden_channels --epochs 1000 --lr 3e-5 --runs 10 --local_layers $layer --dropout $dropout --num_heads $head --metric $metric --ln --res --device $DEVICE

# --- Wiki-cooc ---
echo "  -> Training on Wiki-cooc"
python main_nodeformer.py --dataset wiki-cooc --hidden_channels $hidden_channels --epochs 1500 --lr 0.01 --runs 10 --local_layers $layer --dropout $dropout --num_heads $head --weight_decay 5e-4 --metric $metric --ln --res --device $DEVICE

done
done
done
done
done

echo ""
echo "=================================================="
echo "      ✅ All nodeformer grid searches completed. ✅"
echo "=================================================="