#!/usr/bin/env python
# coding: utf-8

"""
Multi-Environment Balanced S-DATE-SDN
This script implements the functionality from the Jupyter Notebook for analyzing and processing
network intrusion data from multiple environments (UNSW, IoT, SDN), including feature selection,
dimensionality reduction, data balancing, and various machine learning models.
"""

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import cv2
import glob
import os
import time
import io
from collections import Counter
import re
import string
import math
import statistics
import random
import datetime

# Machine learning imports
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from imblearn.over_sampling import SMOTE, ADASYN

# Deep learning imports

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential, Model, load_model
from keras.layers import (
        Input, Dense, Reshape, Flatten, Activation, LeakyReLU, Dropout,
        ZeroPadding2D, BatchNormalization, Conv2D, AveragePooling2D, 
        MaxPooling2D, GlobalMaxPooling2D, LSTM, Conv1D, MaxPooling1D, 
        Embedding, Bidirectional
    )
from keras.initializers import glorot_uniform
import keras.backend as K
from keras.datasets import mnist

# Optimization imports

from mealpy.swarm_based.PSO import OriginalPSO  # Changed from PSO
from mealpy.evolutionary_based.GA import BaseGA  # Changed from GA
from mealpy.evolutionary_based.DE import OriginalDE  # Changed from DE
from mealpy.swarm_based.MFO import OriginalMFO  # Changed from MFO
from mealpy.utils.space import IntegerVar, FloatVar  # Add this line


# CTGAN for synthetic data generation
from sdv.single_table import CTGANSynthesizer as CTGAN
from sdv.metadata import SingleTableMetadata



def create_output_directory():
    """
    Creates an output directory based on current date and time
    
    Returns:
        str: Path to the created output directory
    """
    # Get current date and time for folder name
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create full path
    output_dir = os.path.join(os.getcwd(), "output", folder_name)
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    return output_dir


def load_data(sample_size=10000):
    """
    Load and prepare data from multiple sources (UNSW, IoT, SDN)
    
    Args:
        sample_size (int): Number of samples to take from each dataset
        
    Returns:
        Tuple of DataFrames: processed datasets
    """
    print("Loading datasets...")
    # Load UNSW datasets
    dataUN = pd.read_csv('UNSW_NB15_testing-set.csv')
    dataUN1 = pd.read_csv('UNSW_NB15_training-set.csv')
    dataF = pd.concat([dataUN, dataUN1], ignore_index=True)
    
    # Load IoT dataset
    dataI = pd.read_csv('IoT Network Intrusion Dataset.csv')
    
    # Load SDN datasets
    sdn1 = pd.read_csv('Normal_data.csv')
    sdn2 = pd.read_csv('OVS.csv')
    sdn = pd.concat([sdn1, sdn2], ignore_index=True)
    
    # Shuffle and sample data
    dataI = dataI.sample(frac=1)
    dataF = dataF.sample(frac=1)
    sdn = sdn.sample(frac=1)
    
    # Sample data to specified size
    dataI = dataI[:sample_size]
    dataF = dataF[:sample_size]
    sdn = sdn[:sample_size]
    
    print(f"UNSW dataset size: {len(dataF)}")
    print(f"IoT dataset size: {len(dataI)}")
    print(f"SDN dataset size: {len(sdn)}")
    
    # Map SDN labels to binary format
    vals_to_replace = {
        'BFA': 'Anomaly', 
        'DDoS ': 'Anomaly',
        'DoS': 'Anomaly',
        'Probe': 'Anomaly',
        'Web-Attack': 'Anomaly',
        'BOTNET': 'Anomaly',
        'Normal': 'Normal'
    }
    sdn['Label'] = sdn['Label'].map(vals_to_replace)
    
    # Rename columns for consistency
    dataF.rename(columns={'label': 'Label'}, inplace=True)
    dataF.rename(columns={'attack_cat': 'Sub_Cat'}, inplace=True)
    
    return dataF, dataI, sdn


def prepare_features_and_targets(dataF, dataI, sdn):
    """
    Prepare features and target variables for all datasets
    
    Args:
        dataF: UNSW dataset
        dataI: IoT dataset  
        sdn: SDN dataset
        
    Returns:
        Tuple of DataFrames: features and targets for each dataset
    """
    # Prepare targets and features for IoT dataset
    DataITarget = dataI[["Label", "Cat", "Sub_Cat"]]
    dataIFeatures = dataI.drop(["Label", "Cat", "Sub_Cat"], axis=1)
    
    # Prepare targets and features for UNSW dataset
    DataFTarget = dataF[["Sub_Cat", "Label"]]
    dataFFeatures = dataF.drop(["Sub_Cat", "Label"], axis=1)
    
    # Prepare targets and features for SDN dataset
    sdnTarget = sdn[["Label"]]
    sdnFeature = sdn.drop(["Label"], axis=1)
    
    # Convert UNSW binary labels to text format
    DataFTarget['Label'] = DataFTarget['Label'].replace([0, 1], ['Normal', 'Anomaly'])
    
    print("Class distribution in IoT dataset:", Counter(DataITarget["Label"]))
    print("Class distribution in UNSW dataset:", Counter(DataFTarget["Label"]))
    print("Class distribution in SDN dataset:", Counter(sdnTarget["Label"]))
    
    # Apply label encoding to features
    le = preprocessing.LabelEncoder()
    dataIFeatures = dataIFeatures.apply(LabelEncoder().fit_transform)
    dataFFeatures = dataFFeatures.apply(LabelEncoder().fit_transform)
    sdnFeature = sdnFeature.apply(LabelEncoder().fit_transform)
    
    return dataIFeatures, DataITarget, dataFFeatures, DataFTarget, sdnFeature, sdnTarget


def visualize_tsne(features, target, filename_prefix, output_dir):
    """
    Perform t-SNE visualization and save the plot
    
    Args:
        features: Feature dataframe
        target: Target series
        filename_prefix: Prefix for the output files
        output_dir: Directory to save output files
    """
    print(f"Generating t-SNE visualization for {filename_prefix}...")
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    fe = features.copy()
    fe['Axis 1'] = tsne_results[:,0]
    fe['Axis 2'] = tsne_results[:,1]
    
    target_renamed = target.rename('Target')
    
    plt.figure(figsize=(5,5), dpi=300)
    plt.rcParams.update(plt.rcParamsDefault)
    sns.scatterplot(
        x="Axis 1", y="Axis 2",
        hue=target_renamed,
        data=fe,
        legend="full",
        alpha=1.0,
        linewidths=1.0,
        marker='o',
        s=9
    )
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}.pdf'))
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}.png'))
    plt.close()


def feature_importance(features, target, title, output_file, output_dir):
    """
    Calculate and visualize feature importance using ExtraTrees
    
    Args:
        features: Feature dataframe
        target: Target series
        title: Title for the plot
        output_file: Output file name
        output_dir: Directory to save output files
    
    Returns:
        List: Important feature names
    """
    print(f"Calculating feature importance for {title}...")
    dt = ExtraTreesClassifier(n_estimators=300, random_state=5, max_depth=30)
    
    # Training the model
    dt.fit(features, target)
    
    # Computing the importance of each feature
    feature_importance = dt.feature_importances_
    
    # Normalizing the individual importances
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in dt.estimators_], axis=0)
    
    plt.figure(figsize=(13,7), dpi=400)
    plt.xticks(rotation=90)
    # Plotting a Bar Graph to compare the models
    plt.bar(features.columns, feature_importance_normalized)
    plt.xlabel('Feature Labels', fontsize=12)
    plt.ylabel('Feature Importances', fontsize=12)
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(output_dir, output_file))
    plt.close()
    
    # Select important features
    threshold = 0.02
    if title == "SDN":
        threshold = 0.0001  # Different threshold for SDN
        
    important_features = []
    for i in range(len(features.columns)):
        if feature_importance_normalized[i] > threshold:
            important_features.append(features.columns[i])
            
    print(f"Selected {len(important_features)} important features for {title}")
    return important_features


def select_features(dataIFeatures, DataITarget, dataFFeatures, DataFTarget, sdnFeature, sdnTarget, top_n=20, output_dir=None):
    """
    Select important features for each dataset and standardize column names
    
    Args:
        dataIFeatures, DataITarget: IoT features and targets
        dataFFeatures, DataFTarget: UNSW features and targets
        sdnFeature, sdnTarget: SDN features and targets
        top_n: Number of top features to select
        output_dir: Directory to save output files
        
    Returns:
        Tuple: Selected features for each dataset
    """
    # Calculate feature importance for each dataset
    featuresimp = feature_importance(dataIFeatures, DataITarget["Label"], "IoT", 'IoTFeature.pdf', output_dir)
    featuresimpF = feature_importance(dataFFeatures, DataFTarget["Label"], "UNSW", 'UNSWFeature.pdf', output_dir)
    featuresimpS = feature_importance(sdnFeature, sdnTarget["Label"], "SDN", 'SDNFeature.pdf', output_dir)
    
    # Select top N features for each dataset
    dataIFeatures = dataIFeatures[featuresimp[:top_n]]
    dataFFeatures = dataFFeatures[featuresimpF[:top_n]]
    sdnFeature = sdnFeature[featuresimpS[:top_n]]
    
    # Standardize column names
    numeric_cols = list(range(top_n))
    dataIFeatures.columns = numeric_cols
    dataFFeatures.columns = numeric_cols
    sdnFeature.columns = numeric_cols
    
    return dataIFeatures, dataFFeatures, sdnFeature


def combine_datasets(dataIFeatures, DataITarget, dataFFeatures, DataFTarget, sdnFeature, sdnTarget):
    """
    Combine datasets into a single dataset
    
    Args:
        dataIFeatures, DataITarget: IoT features and targets
        dataFFeatures, DataFTarget: UNSW features and targets
        sdnFeature, sdnTarget: SDN features and targets
        
    Returns:
        DataFrame: Combined dataset
    """
    print("Combining datasets...")
    # Combine features using concat instead of append
    newfeatures = pd.concat([dataFFeatures, dataIFeatures], ignore_index=True)
    newfeatures = pd.concat([newfeatures, sdnFeature], ignore_index=True)
    
    # Combine targets using concat instead of append
    newtarget = pd.concat([DataFTarget["Label"], DataITarget["Label"]], ignore_index=True)
    newtarget = pd.concat([newtarget, sdnTarget["Label"]], ignore_index=True)
    
    # Log scale transformation
    transformer = FunctionTransformer(np.log1p)
    ScDATA = transformer.transform(newfeatures)
    newfeatures = pd.DataFrame(ScDATA)
    
    # Add target column
    newfeatures['target'] = newtarget
    
    # Shuffle data
    newfeatures = newfeatures.sample(frac=1)
    
    return newfeatures


def data_balancing(Data25):
    """
    Balance data using dimensional reduction and data generation
    
    Args:
        Data25: Combined dataset
        
    Returns:
        DataFrame: Balanced dataset
    """
    print("Balancing data...")
    # Split by class
    dataF9 = Data25.loc[Data25['target'] == "Normal"]
    print(f"Normal samples: {len(dataF9)}")
    dataF10 = Data25.loc[Data25['target'] == "Anomaly"]
    print(f"Anomaly samples: {len(dataF10)}")
    
    # Convert to numeric for processing
    dataF9.columns = dataF9.columns.astype(str)
    dataF10.columns = dataF10.columns.astype(str)
    dataF9['target'] = dataF9['target'].map({'Normal': 0})
    
    # Generate synthetic samples for minority class
    Z, Mini = dimRed(dataF9)
    
    arr = []
    l = 0
    
    # Generate synthetic data
    print("Generating synthetic data...")
    for i in range(1):
        if(l < len(dataF9)):
            print(f"Iteration {i}, data point {l}")
            Z['distance'] = 0
            Z['distance'] = pairwise_distances(Z[i:i+1], Z[:]).reshape(-1)
            
            # Create new sample by averaging nearest neighbors
            a = Mini.iloc[[Z['distance'][l+1:].idxmin(), Z['distance'][l+1:].nsmallest(2).index[1], l]].mean(axis=0)
            arr.append(a)
            l += 1
        else:
            Genrated = pd.DataFrame(arr)
            new_header = Mini.columns
            Genrated.columns = new_header
            Genrated['target'] = Genrated['target']
            dataF9 = dataF9.drop(['pca-one', 'pca-two'], axis=1)
            # Replace append with concat
            dataF9 = pd.concat([Genrated, dataF9], ignore_index=True)
            dataF9 = dataF9.reset_index()
            dataF9 = dataF9.drop('index', axis=1)
            arr.clear()
            l = 0
    
    # Process final generated data
    Genrated = pd.DataFrame(arr)
    new_header = Mini.columns
    Genrated.columns = new_header
    Genrated['target'] = Genrated['target'].astype(int)
    
    dataF9 = dataF9.drop(['pca-one', 'pca-two'], axis=1)
    # Replace append with concat
    dataF9 = pd.concat([Genrated, dataF9], ignore_index=True)
    
    # Map back to original label
    dataF9['target'] = dataF9['target'].map({0: 'Normal'})
    Genrated['target'] = Genrated['target'].map({0: 'Normal'})
    
    # Combine balanced data
    # Replace append with concat
    dataF = pd.concat([dataF9, dataF10], ignore_index=True)
    print(f"Class distribution after balancing: {Counter(dataF['target'])}")
    
    return dataF


def dimRed(df):
    """
    Perform dimensionality reduction using PCA
    
    Args:
        df: DataFrame to reduce
        
    Returns:
        Tuple: Reduced data and original data without PCA columns
    """
    # Extract features and target
    features = df.drop('target', axis=1)
    target = df['target']
    
    # Apply PCA
    svd = PCA(n_components=2)
    svd.fit(features)
    pca_result = svd.transform(features)
    
    # Add PCA results to dataframe
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    
    # Create dataframe with PCA results and target
    Z = df[['pca-one', 'pca-two']]
    Z['target'] = target
    
    # Create copy without PCA columns
    Mini = df.drop(['pca-one', 'pca-two'], axis=1)
    
    # Reset indices
    Z = Z.reset_index()
    Mini = Mini.reset_index()
    Mini = Mini.drop('index', axis=1)
    
    print(f"Dimensionality reduction: {Z.shape}, {Mini.shape}")
    
    return Z, Mini


def train_test_split_data(features, target):
    """
    Split data into training and testing sets
    
    Args:
        features: Feature dataframe
        target: Target series
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.30, random_state=1, 
        stratify=target, shuffle=True
    )
    return X_train, X_test, y_train, y_test


def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, is_nn=False):
    """
    Train and evaluate a machine learning model
    
    Args:
        model: The model to train
        X_train, y_train, X_test, y_test: Training and test data
        model_name: Name of the model for reporting
        is_nn: Whether the model is a neural network
        
    Returns:
        float: Accuracy score
    """
    print(f"Training and evaluating {model_name} model...")
    
    if is_nn:
        # For neural network models
        y_train1 = pd.get_dummies(y_train).values
        y_test1 = pd.get_dummies(y_test).values
        
        history = model.fit(X_train, y_train1, validation_split=0.1, epochs=50)
        
        predict_x = model.predict(X_test) 
        classes_x = np.argmax(predict_x, axis=1)
        rounded_labels = np.argmax(y_test1, axis=1)
        
        print(classification_report(rounded_labels, classes_x))
        print(confusion_matrix(rounded_labels, classes_x))
        
        return history, accuracy_score(rounded_labels, classes_x)
    else:
        # For traditional ML models
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        print(classification_report(y_test, pred))
        print(confusion_matrix(y_test, pred))
        accuracy = accuracy_score(y_test, pred)
        print(f"Accuracy: {accuracy}")
        
        return accuracy


def plot_epochs_results(loss_values, vlloss_values, accuracy_values, vlaccuracy_values, model_name, output_dir):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        loss_values: Training loss values
        vlloss_values: Validation loss values
        accuracy_values: Training accuracy values
        vlaccuracy_values: Validation accuracy values
        model_name: Name of the model for the plot title
        output_dir: Directory to save output files
    """
    epochs = range(1, len(loss_values) + 1)
    
    # Plotting loss values
    plt.figure(figsize=(5, 3), dpi=300)
    plt.rcParams.update(plt.rcParamsDefault)
    plt.scatter(epochs, loss_values, c='b', label='Training Loss', s=15)
    plt.scatter(epochs, vlloss_values, c='c', label='Validation Loss', s=15)
    plt.plot(epochs, loss_values, 'b')
    plt.plot(epochs, vlloss_values, 'c')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.margins(x=0)
    plt.savefig(os.path.join(output_dir, model_name + 'Loss.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Plotting accuracy values
    plt.figure(figsize=(5, 3), dpi=300)
    plt.scatter(epochs, accuracy_values, c='r', label='Training Accuracy', s=15)
    plt.scatter(epochs, vlaccuracy_values, c='b', label='Validation Accuracy', s=15)
    plt.plot(epochs, accuracy_values, 'r')
    plt.plot(epochs, vlaccuracy_values, 'b')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.margins(x=0)
    plt.savefig(os.path.join(output_dir, model_name + 'acc.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()


def create_lstm_model(input_shape):
    """
    Create an LSTM model for sequence classification
    
    Args:
        input_shape: Shape of the input data
        
    Returns:
        Keras model: Compiled LSTM model
    """
    model = Sequential()
    model.add(Embedding(1000, 100, input_length=input_shape))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def create_cnn_model(input_shape):
    """
    Create a CNN model for sequence classification
    
    Args:
        input_shape: Shape of the input data
        
    Returns:
        Keras model: Compiled CNN model
    """
    model = Sequential()
    model.add(Embedding(1000, 100, input_length=input_shape))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def create_gru_model(input_shape):
    """
    Create a GRU model for sequence classification
    
    Args:
        input_shape: Shape of the input data
        
    Returns:
        Keras model: Compiled GRU model
    """
    model = Sequential()
    model.add(Embedding(1000, 100, input_length=input_shape))
    model.add(layers.GRU(128))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def optimize_ensemble(X_train, y_train, X_test, y_test, optimization_method='PSO'):
    """
    Optimize an ensemble of classifiers using metaheuristic algorithms
    
    Args:
        X_train, y_train, X_test, y_test: Training and test data
        optimization_method: The optimization algorithm to use ('PSO', 'GA', etc.)
        
    Returns:
        Tuple: Best position and fitness (hyperparameters and accuracy)
    """
    accE = []
    
    def fitness_function(solution):
        print(f"Testing solution: {solution}")
        rf_model1 = RandomForestClassifier(n_estimators=int(solution[0]), max_depth=int(solution[1]))
        rf_model2 = AdaBoostClassifier(n_estimators=int(solution[2]), random_state=5, learning_rate=solution[3])
        rf_model3 = ExtraTreesClassifier(n_estimators=int(solution[4]), max_depth=int(solution[5]))
        
        # Create the ensemble classifier
        ensemble_model = VotingClassifier(
            estimators=[('rf', rf_model1), ('rf1', rf_model2), ('rf2', rf_model3)],
            voting='soft'
        )
        
        # Fit the ensemble
        ensemble_model.fit(X_train, y_train)
        pred = ensemble_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, pred)
        print(f"Accuracy: {accuracy}")
        print(classification_report(y_test, pred))
        print(confusion_matrix(y_test, pred))
        
        accE.append(accuracy)
        return accuracy
    
    # Define the problem
    # For Mealpy 3.x+, 'bounds' should be a list of Var objects.
    bounds_new = [
        IntegerVar(lb=2, ub=300, name="rf_n_estimators"),          # n_estimators for RandomForestClassifier
        IntegerVar(lb=2, ub=200, name="rf_max_depth"),             # max_depth for RandomForestClassifier
        IntegerVar(lb=2, ub=300, name="ada_n_estimators"),         # n_estimators for AdaBoostClassifier
        FloatVar(lb=0.01, ub=1.0, name="ada_learning_rate"),       # learning_rate for AdaBoostClassifier
        IntegerVar(lb=2, ub=300, name="et_n_estimators"),          # n_estimators for ExtraTreesClassifier
        IntegerVar(lb=2, ub=200, name="et_max_depth")              # max_depth for ExtraTreesClassifier
    ]

    problem = {
        "obj_func": fitness_function,
        "bounds": bounds_new,
        "minmax": "max",
        "log_to": None,
        "save_population": False,
    }
    
    # Select optimization algorithm
    if optimization_method == 'PSO':
        model = OriginalPSO(epoch=10, pop_size=5) # Removed 'problem' argument
    elif optimization_method == 'GA':
        model = BaseGA(epoch=10, pop_size=5, pc=0.9, pm=0.05) # Removed 'problem' argument
    elif optimization_method == 'MFO':
        model = OriginalMFO(epoch=10, pop_size=5) # Removed 'problem' argument
    elif optimization_method == 'DE':
        model = OriginalDE(epoch=10, pop_size=5, wf=0.8, cr=0.9) # Removed 'problem' argument
    else:
        model = OriginalPSO(epoch=2, pop_size=5)  # Default to OriginalPSO, removed 'problem' argument
    
    # Solve the optimization problem
    # For newer versions of Mealpy, model.solve() returns the best agent object
    start_time = time.time()
    best_agent = model.solve(problem)
    end_time = time.time()
    optimization_duration = end_time - start_time
    print(f"Optimization completed in {optimization_duration:.2f} seconds.")
    best_position = best_agent.solution
    best_fitness = best_agent.target # For single-objective problems

    print(f"Best solution: {best_position}, Best fitness: {best_fitness}")

    
    return best_position, best_fitness


def use_smote_oversampling(features, target):
    """
    Apply SMOTE oversampling to balance the dataset
    
    Args:
        features: Feature dataframe
        target: Target series
        
    Returns:
        Tuple: Oversampled features and targets
    """
    print("Applying SMOTE oversampling...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(features, target)
    print(f"Class distribution after SMOTE: {Counter(y_res)}")
    return X_res, y_res


def use_adasyn_oversampling(features, target):
    """
    Apply ADASYN oversampling to balance the dataset
    
    Args:
        features: Feature dataframe
        target: Target series
        
    Returns:
        Tuple: Oversampled features and targets
    """
    print("Applying ADASYN oversampling...")
    ada = ADASYN()
    X_resA, y_resY = ada.fit_resample(features, target)
    print(f"Class distribution after ADASYN: {Counter(y_resY)}")
    return X_resA, y_resY


def use_ctgan_oversampling(Data25, num_samples=67824):
    """
    Apply CTGAN synthetic data generation to balance the dataset
    
    Args:
        Data25: Dataset to oversample
        num_samples: Number of synthetic samples to generate
        
    Returns:
        DataFrame: Dataset with synthetic samples
    """
    print("Applying CTGAN synthetic data generation...")
    dataNCTGAN = Data25.loc[Data25['target'] == "Normal"]
    print(f"Normal samples before CTGAN: {len(dataNCTGAN)}")
    dataACTGAN = Data25.loc[Data25['target'] == "Anomaly"]
    print(f"Anomaly samples before CTGAN: {len(dataACTGAN)}")
    
    dataNCTGAN.columns = dataNCTGAN.columns.astype(str)
    dataACTGAN.columns = dataACTGAN.columns.astype(str)
    
    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=dataNCTGAN)
    
    # Create and train CTGAN model
    model = CTGAN(metadata, epochs=100, verbose=True, batch_size=32)
    model.fit(dataNCTGAN)
    
    # Generate synthetic data
    new_data_pii = model.sample(num_samples)
    
    # Combine with original data using concat instead of append
    NewCTGANDATA = pd.concat([new_data_pii, dataNCTGAN], ignore_index=True)
    CTGANDATA = pd.concat([NewCTGANDATA, dataACTGAN], ignore_index=True)
    
    return CTGANDATA


def plot_confusion_matrix(conf_matrix_data, title, output_dir):
    """
    Plot a confusion matrix
    
    Args:
        conf_matrix_data: Confusion matrix data
        title: Title for the plot
        output_dir: Directory to save output files
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = ['Anomaly', 'Normal']
    ax = sns.heatmap(conf_matrix_data, annot=True, cmap="Blues", fmt='g', cbar=False, annot_kws={"size": 25})
    plt.title(title, fontsize=20)
    ax.xaxis.set_ticklabels(labels, fontsize=17) 
    ax.yaxis.set_ticklabels(labels, fontsize=17)
    ax.set_ylabel('Test', fontsize=20)
    ax.set_xlabel('Predicted', fontsize=20)
    plt.savefig(os.path.join(output_dir, f'conf_{title}.pdf'))
    plt.close()


def main():
    """Main function to execute the entire workflow"""
    # Create output directory
    output_dir = create_output_directory()
    
    # Load data
    dataF, dataI, sdn = load_data(sample_size=10000)
    
    # Prepare features and targets
    dataIFeatures, DataITarget, dataFFeatures, DataFTarget, sdnFeature, sdnTarget = prepare_features_and_targets(dataF, dataI, sdn)
    
    # Visualize SDN data
    visualize_tsne(sdnFeature, sdnTarget["Label"], "SDNscatter", output_dir)
    
    # Select important features
    dataIFeatures, dataFFeatures, sdnFeature = select_features(
        dataIFeatures, DataITarget, dataFFeatures, DataFTarget, sdnFeature, sdnTarget, 
        top_n=20, output_dir=output_dir
    )
    
    # Combine all datasets
    combined_data = combine_datasets(dataIFeatures, DataITarget, dataFFeatures, DataFTarget, sdnFeature, sdnTarget)
    
    # Save combined data
    combined_data.to_csv(os.path.join(output_dir, 'MENImbalanced.csv'), index=False)
    
    # Visualize imbalanced data
    visualize_tsne(combined_data.drop('target', axis=1), combined_data['target'], "MENImbalanced", output_dir)
    
    # Balance data
    balanced_data = data_balancing(combined_data)
    
    # Save balanced data
    balanced_data.to_csv(os.path.join(output_dir, 'ProposedUpsamplingData1.csv'), index=False)
    
    # Visualize balanced data
    visualize_tsne(balanced_data.drop('target', axis=1), balanced_data['target'], "MENbalanced", output_dir)
    
    # Extract features and target
    target1 = balanced_data['target']
    feature1 = balanced_data.drop('target', axis=1)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split_data(feature1, target1)
    
    # Traditional Machine Learning Models
    print("\n=== Training Traditional ML Models ===")
    print("\nRandom Forest:")
    rf = RandomForestClassifier(n_estimators=100, random_state=2, max_depth=10)
    rf_acc = train_evaluate_model(rf, X_train, y_train, X_test, y_test, "Random Forest")
    
    print("\nAdaBoost:")
    ada = AdaBoostClassifier(n_estimators=100, random_state=5, learning_rate=0.2)
    ada_acc = train_evaluate_model(ada, X_train, y_train, X_test, y_test, "AdaBoost")
    
    print("\nExtra Trees:")
    etc = ExtraTreesClassifier(n_estimators=100, random_state=5, max_depth=10)
    etc_acc = train_evaluate_model(etc, X_train, y_train, X_test, y_test, "Extra Trees")
    
    print("\nLogistic Regression:")
    lr = LogisticRegression(random_state=1000, solver='liblinear', multi_class='ovr', C=3.0)
    lr_acc = train_evaluate_model(lr, X_train, y_train, X_test, y_test, "Logistic Regression")
    
    # Deep Learning Models
    try:
        print("\n=== Training Deep Learning Models ===")
        
        print("\nLSTM:")
        lstm_model = create_lstm_model(feature1.shape[1])
        lstm_history, lstm_acc = train_evaluate_model(
            lstm_model, X_train, y_train, X_test, y_test, "LSTM", is_nn=True
        )
        plot_epochs_results(
            lstm_history.history['loss'], lstm_history.history['val_loss'],
            lstm_history.history['accuracy'], lstm_history.history['val_accuracy'],
            'LSTMSDATE'
        )
        
        print("\nCNN:")
        cnn_model = create_cnn_model(feature1.shape[1])
        cnn_history, cnn_acc = train_evaluate_model(
            cnn_model, X_train, y_train, X_test, y_test, "CNN", is_nn=True
        )
        plot_epochs_results(
            cnn_history.history['loss'], cnn_history.history['val_loss'],
            cnn_history.history['accuracy'], cnn_history.history['val_accuracy'],
            'CNNSDATE'
        )
        
        print("\nGRU:")
        gru_model = create_gru_model(feature1.shape[1])
        gru_history, gru_acc = train_evaluate_model(
            gru_model, X_train, y_train, X_test, y_test, "GRU", is_nn=True
        )
        plot_epochs_results(
            gru_history.history['loss'], gru_history.history['val_loss'],
            gru_history.history['accuracy'], gru_history.history['val_accuracy'],
            'GRUSDATE'
        )
    except Exception as e:
        print(f"Error training deep learning models: {e}")
    
    # Optimized ensemble
    try:
        print("\n=== Optimizing Ensemble Model ===")
        print("\n----Optimizing with PSO:----")
        best_params_pso, best_acc_pso = optimize_ensemble(
            X_train, y_train, X_test, y_test, optimization_method='PSO'
        )
        # print("\n----Optimizing with GA:----")
        # best_params_ga, best_acc_ga = optimize_ensemble(
        #     X_train, y_train, X_test, y_test, optimization_method='GA'
        # )
    except Exception as e:
        print(f"Error optimizing ensemble: {e}")
    
    # Comparison with other oversampling methods
    print("\n=== Comparing with Other Oversampling Methods ===")
    
    # Original data (imbalanced)
    print("\nOriginal Data (Imbalanced):")
    target = combined_data['target']
    feature = combined_data.drop('target', axis=1)
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split_data(feature, target)
    
    print("Random Forest on original data:")
    rf = RandomForestClassifier(n_estimators=100, random_state=2, max_depth=10)
    rf_acc_orig = train_evaluate_model(rf, X_train_o, y_train_o, X_test_o, y_test_o, "RF Original")
    
    # SMOTE oversampling
    print("\nSMOTE Oversampling:")
    X_res_smote, y_res_smote = use_smote_oversampling(feature, target)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split_data(X_res_smote, y_res_smote)
    
    print("Random Forest with SMOTE:")
    rf_acc_smote = train_evaluate_model(rf, X_train_s, y_train_s, X_test_s, y_test_s, "RF SMOTE")
    
    # ADASYN oversampling
    print("\nADASYN Oversampling:")
    X_res_adasyn, y_res_adasyn = use_adasyn_oversampling(feature, target)
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split_data(X_res_adasyn, y_res_adasyn)
    
    print("Random Forest with ADASYN:")
    rf_acc_adasyn = train_evaluate_model(rf, X_train_a, y_train_a, X_test_a, y_test_a, "RF ADASYN")
    
    #Try CTGAN if available
    # try:
    #     print("\nCTGAN Synthetic Data:")
    #     ctgan_data = use_ctgan_oversampling(combined_data)
    #     ctgan_target = ctgan_data['target']
    #     ctgan_feature = ctgan_data.drop('target', axis=1)
    #     X_train_c, X_test_c, y_train_c, y_test_c = train_test_split_data(ctgan_feature, ctgan_target)
        
    #     print("Random Forest with CTGAN:")
    #     rf_acc_ctgan = train_evaluate_model(rf, X_train_c, y_train_c, X_test_c, y_test_c, "RF CTGAN")
    # except Exception as e:
    #     print(f"Error using CTGAN: {e}")
    
    # Print summary of results
    print("\n=== Results Summary ===")
    print(f"Proposed Method - RF: {rf_acc:.4f}, AdaBoost: {ada_acc:.4f}, Extra Trees: {etc_acc:.4f}")
    
    try:
        print(f"Proposed Method - LSTM: {lstm_acc:.4f}, CNN: {cnn_acc:.4f}, GRU: {gru_acc:.4f}")
    except:
        pass
    
    print(f"Original - RF: {rf_acc_orig:.4f}")
    print(f"SMOTE - RF: {rf_acc_smote:.4f}")
    print(f"ADASYN - RF: {rf_acc_adasyn:.4f}")
    
    try:
        print(f"PSO - RF: {best_acc_pso:.4f}")
        # print(f"GA - RF: {best_acc_ga:.4f}")
    except:
        pass
    # try:
    #     print(f"CTGAN - RF: {rf_acc_ctgan:.4f}")
    # except:
    #     pass


if __name__ == "__main__":
    main()
