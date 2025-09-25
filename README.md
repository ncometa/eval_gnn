A Unified Framework for Evaluating Graph Neural Networks
This repository provides a comprehensive and unified framework for training and evaluating a wide range of Graph Neural Network (GNN) models on both node classification and graph classification benchmarks. The codebase is designed for modularity and ease of experimentation, allowing researchers and practitioners to quickly benchmark standard GNNs and advanced architectures like Graph Transformers.

Key Features
Unified Pipeline: A single, streamlined training and evaluation pipeline for diverse models and datasets.

Multiple Tasks: Supports both node-level and graph-level classification tasks.

Broad Model Support: Includes implementations of classic GNNs (GCN, GAT, GraphSAGE, GIN) and advanced models (FSGCN, GPRGNN, GPS, Subgraphormer, GraphViT).

Standard Benchmarks: Integrated loaders for popular datasets from PyG, OGB, and other common sources.

Comprehensive Evaluation: Automatically calculates multiple metrics (Accuracy, Balanced Accuracy, ROC-AUC, PR-AUC) and generates detailed classification reports.

Reproducibility: Structured output directories save model weights, metrics, and summaries for each run, ensuring reproducible results.

Prerequisites
Python 3.8+

PyTorch 1.10+

CUDA (for GPU acceleration)

Installation
It is highly recommended to use a virtual environment to manage dependencies.

Clone the repository:

git clone [https://github.com/ncometa/eval_gnn.git](https://github.com/ncometa/eval_gnn.git)
cd eval_gnn

Create and activate a virtual environment (optional but recommended):

python3 -m venv gnn_env
source gnn_env/bin/activate

Install the required packages:
The core dependencies can be installed via pip.

pip install torch torchvision torchaudio
pip install torch-geometric
pip install ogb
pip install numpy scikit-learn matplotlib gdown torcheval

Directory Structure
The repository is organized by task to maintain a clean and understandable structure.

eval_gnn/
├── graph_classification/       # Code for graph classification tasks
│   ├── gnn_models.py           # GCN, GAT, SAGE, GIN models
│   ├── gps_model.py            # GPS model
│   ├── subgraphormer_model.py  # Subgraphormer model
│   ├── main_graphclass.py      # Main script for running experiments
│   ├── parse_graphclass.py     # Argument parser for graph classification
│   ├── training_utils.py       # Training and evaluation loops
│   ├── run_*.sh                # Example bash scripts for experiments
│   └── ...
├── node_classification/        # Code for node classification tasks
│   ├── model.py                # Standard GNNs (MPNNs wrapper)
│   ├── FSGCN_models.py         # FSGCN model
│   ├── gprgnn_models.py        # GPR-GNN model
│   ├── main.py                 # Main script for running experiments
│   ├── parse.py                # Argument parser for node classification
│   ├── dataset.py              # Data loaders
│   └── ...
└── results/                    # Directory where all experiment outputs are saved
└── README.md

Usage Guide
All experiments can be launched from the command line. Results, including model checkpoints and performance metrics, will be saved to the results/ directory, organized by dataset, model, and hyperparameters.

1. Node Classification
Experiments are run using the node_classification/main.py script.

Basic Example (GCN on Cora)
To train a GCN model on the Cora dataset for 3 independent runs:

python node_classification/main.py \
    --dataset cora \
    --gnn gcn \
    --runs 3 \
    --epochs 500 \
    --hidden_channels 256 \
    --local_layers 7 \
    --lr 0.001 \
    --dropout 0.5 \
    --device 0

Advanced Example (FSGCN on Chameleon)
To train the FSGCN model, which requires special feature pre-computation:

python node_classification/main_fsgcn.py \
    --dataset chameleon \
    --gnn fsgcn \
    --runs 5 \
    --epochs 1000 \
    --lr 0.01 \
    --weight_decay 0.0005 \
    --fsgcn_num_layers 3 \
    --fsgcn_feat_type all \
    --device 0

Key Arguments:

--dataset: Name of the dataset (e.g., cora, citeseer, pubmed, ogbn-arxiv, chameleon).

--gnn: GNN architecture to use (gcn, gat, sage, gin, fsgcn, gprgnn, mlp).

--runs: Number of different seeds to run.

--metric: Metric for selecting the best model (acc, rocauc, balacc, prauc).

--hidden_channels, --local_layers, --lr, --dropout: Key hyperparameters.

--device: GPU device ID to use.

For a full list of options, run python node_classification/main.py --help.

2. Graph Classification
Experiments are run using the graph_classification/main_graphclass.py script. The repository also includes several shell scripts (graph_classification/run_*.sh) that demonstrate how to launch hyperparameter sweeps.

Basic Example (GIN on MUTAG)
python graph_classification/main_graphclass.py \
    --dataset_type tu \
    --dataset MUTAG \
    --model_name gin \
    --runs 3 \
    --epochs 100 \
    --batch_size 32 \
    --hidden_channels 128 \
    --lr 0.001 \
    --pool add \
    --device 0

Running an Experiment Script (GPS on COX2)
The provided shell scripts automate the process. For example, to run the GPS model on the COX2 dataset:

bash graph_classification/run_gps_cox2.sh

(You may need to modify the GPU device ID inside the script)

Key Arguments:

--dataset_type: Type of dataset (tu or ogb).

--dataset: Name of the dataset (e.g., MUTAG, PROTEINS, ogbg-molhiv).

--model_name: GNN architecture to use.

--pool: Graph pooling method (mean or add).

--batch_size, --epochs, --lr: Training hyperparameters.

Output and Results
All results are saved in the results/ directory, following this structure:

results/
└── [dataset_name]/
    └── [model_name]/
        └── [hyperparameter_string]/
            ├── run_1/
            │   ├── metrics.json
            │   └── model.pt
            ├── run_2/
            │   └── ...
            └── summary/
                ├── summary.json
                ├── reports_default.json
                └── reports_optimal.json

metrics.json: Contains detailed performance metrics for a single run.

model.pt: The saved weights of the best performing model from that run.

summary.json: Aggregates the mean and standard deviation of metrics across all runs.

reports_*.json: Contains detailed classification reports.

Implemented Models & Supported Datasets
Models
Task

Implemented Models

Node Classification

GCN, GAT, GraphSAGE, GIN, FSGCN, GloGNN, GPRGNN, MLP

Graph Classification

GCN, GAT, GraphSAGE, GIN, GPS, Subgraphormer, GraphViT

Datasets
Category

Datasets

Planetoid

cora, citeseer, pubmed

OGB

ogbn-arxiv, ogbn-products, ogbg-molhiv

TUDatasets

MUTAG, PROTEINS, COLLAB, COX2, DD, etc.

PyG Heterophilous

roman-empire, amazon-ratings, minesweeper, tolokers, questions

Geom-GCN

chameleon, squirrel

Other

wikics, pokec, wiki-cooc, amazon-photo, coauthor-cs, etc.

Citation
If you use this code in your research, please consider citing the repository:

@misc{eval_gnn,
  author = {ncometa},
  title = {A Unified Framework for Evaluating Graph Neural Networks},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/ncometa/eval_gnn](https://github.com/ncometa/eval_gnn)}},
}
