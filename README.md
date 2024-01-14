# Fraud Detection MLP

This project implements different ML model's for fraud detection using financial data. These models are trained on a dataset with 28 raw financial Items, 14 financial ratios and also combining both as input features to predict fraudulent activities.


## Folder Structure

The project has the following folder structure:

- `data/`: A directory containing data files for training and testing the ML model.
- `GrowNet/`: Ensmeble Neural Networks.
  - `data`
    - `data.py` : data model
  - `misc`
    - `auc.py` : scripts to get auc scores
  - `models`
    - `dynamic_net.py` : 
    - `mlp.py`:
    - `splinear.py`:
- `results/`: A directory containing Python scripts and files for data processing and model evaluation.
  - `data_processing.py`: Python script for data processing.
  - `features.json`: JSON file specifying feature mapping.
  - `models.py`: Python script for model-related functions.
  - `utils.py`: Utility functions.
- `MLP/`: A directory containing the code for the MLP model.
  - `ensemble.py`: A Python script that trains an AdaBoost ensemble classifier using a DecisionTreeClassifier as the base model and evaluates its performance.
  - `features.json`: JSON file specifying the mapping of features in the dataset.
  - `main.py`: The main script for the MLP model.
  - `utils.py`: Utility functions used in the MLP model.
- `Notebooks/`: A directory containing Jupyter Notebook files with detailed explanations and implementations.
  - `Beneish_model.ipynb`: A Jupyter Notebook for replicating the Beneish model with explanations.
  - `MLP and RusBoost.ipynb`: A Jupyter Notebook for implementing MLP and RusBoost with explanations.
  - `MLP_Bao_finanancial_data.ipynb`: A Jupyter Notebook with detailed explanations for the MLP model.
  - `model-tuning.ipynb`: A Jupyter Notebook for exploring different model configurations and tuning hyperparameters.
  - `RusBoost, XgBoost, SVM.ipynb`: A Jupyter Notebook for implementing MLP with different methods and models.
- `WordLists/`: Folder containing word lists used in different papers.
- `README.md`: This README file.


**Note:** This is readme file is not up to date


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
- **results/**: Data processing and model evaluation scripts and files.
  - `data_processing.py`: Data processing script.
  - `features.json`: Feature mapping JSON file.
  - `models.py`: Model-related functions.
  - `utils.py`: Utility functions.
- **MLP/**: Code for the MLP model.
  - `ensemble.py`: Script to train an AdaBoost ensemble classifier.
  - `features.json`: Feature mapping JSON file.
  - `main.py`: Main script for the MLP model.
  - `utils.py`: Utility functions for MLP.

- **WordLists/**: Word lists used in different papers.

## Dependencies

- note: requirements.txt is not fully updated
