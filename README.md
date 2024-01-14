# Fraud Detection Project

This project implements machine learning models for fraud detection using financial data. The models are trained on a dataset with 28 raw financial items, 14 financial ratios, and both combined as input features to predict fraudulent activities.

## Folder Structure

The project is organized into the following folder structure:

- **data/**: Contains data files for training and testing the ML models.
- **GrowNet/**: Ensemble Neural Networks.
  - **data/**
    - `data.py`: Data model.
  - **misc/**
    - `auc.py`: Scripts to calculate AUC scores.
  - **models/**
    - `dynamic_net.py`: Dynamic neural network model.
    - `mlp.py`: MLP models with varying params.
    - `splinear.py`: Custom linear layers.
- **fraudDetec/**: Data processing and model evaluation scripts and files.
  - `data_processing.py`: Data processing script.
  - `features.json`: Feature mapping JSON file.
  - `models.py`: Model-related functions.
  - `utils.py`: Utility functions.
- **MLP/**: Code for the MLP model.
  - `ensemble.py`: Script to train an AdaBoost ensemble classifier.
  - `features.json`: Feature mapping JSON file.
  - `main.py`: Main script for the MLP model.
  - `utils.py`: Utility functions for MLP.
 **results/**: Results for detecting fraud.
  - `grownet_train.ipynb`: Jupyter Notebook script to train an ensemble neural_nets.
  - `oversample_res.ipynb`: Jupyter Notebook for evaluating results using oversampling.
  - `undersample_res.ipynb`: Jupyter Notebook for evaluating results using undersampling.

- **WordLists/**: Word lists used in different papers.

## Dependencies

- note: requirements.txt is not fully updated


**Note:** This is readme file is not up to date