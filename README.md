# S-DATE-SDN Multi-Environment Balancing Analysis

## Overview
This project implements advanced machine learning models for network intrusion detection across various environments (UNSW, IoT, SDN). It includes comprehensive functionalities for data loading, feature selection, dimensionality reduction, data balancing, and model training with GPU acceleration.

## Features
- Multi-environment data analysis (UNSW, IoT, SDN)
- Calculation and selection of important features
- Dimensionality reduction using PCA and t-SNE
- Data balancing with synthetic data generation techniques
- Multiple machine learning models:
  - Traditional ML: Random Forest, AdaBoost, ExtraTrees
  - Deep Learning: LSTM, CNN, GRU
- GPU acceleration for both traditional machine learning and deep learning models
- Ensemble optimization using meta-heuristic algorithms (PSO, GA, MFO, DE)
- Multiple data balancing techniques (SMOTE, ADASYN, CTGAN)

## System Requirements
You can install the necessary libraries using:
```
pip install -r requirements.txt
```

## Data Structure
The project uses the following datasets:
- UNSW dataset: `UNSW_NB15_testing-set.csv`, `UNSW_NB15_training-set.csv`
- IoT dataset: `IoT Network Intrusion Dataset.csv`
- SDN dataset: `Normal_data.csv`, `OVS.csv`

## Dataset
You can download the data from [this link](https://husteduvn-my.sharepoint.com/:f:/g/personal/thanh_nxc235623_sis_hust_edu_vn/EhUEEyE-g_ZLp5Ev928M3B8B3NQJIFBoN70tBG4Lt5qoAA?e=pnbVq1).

## How to Use
To run the main script:
```
python m-en-s-date-gpu.py
```

The script will perform the following:
1. Check for GPU availability
2. Load and prepare data from all sources
3. Select important features
4. Generate visualization plots
5. Balance the datasets
6. Train and evaluate multiple machine learning models
7. Save results to the output directory

## Output Results
Results are saved in timestamped directories within the `output/` folder, including:
- Feature importance plots
- t-SNE visualizations
- Model performance metrics
- Training/validation graphs
- Confusion matrices

## GPU Acceleration
The code automatically detects and utilizes GPU acceleration when available, with PyTorch as the supporting backend. GPU support is implemented for both deep learning models and traditional machine learning algorithms.

## Main Components
- `setup_gpu()`: Configures PyTorch to use GPU if available
- `load_data()`: Loads data from various sources
- `select_features()`: Selects important features using ExtraTrees
- `data_balancing()`: Balances data using dimensionality reduction and synthetic data generation
- `train_evaluate_model()`: Trains and evaluates models with optional GPU acceleration

## Models
- Traditional ML: RandomForest, AdaBoost, ExtraTrees
- Deep Learning:
  - `LSTMModel`: LSTM-based network for sequence classification
  - `CNNModel`: CNN-based network for classification
  - `GRUModel`: GRU-based network for sequence classification

## Optimization
The project includes optimization techniques for ensemble models using:
- PSO: Particle Swarm Optimization
- GA: Genetic Algorithm
- MFO: Moth-Flame Optimization
- DE: Differential Evolution