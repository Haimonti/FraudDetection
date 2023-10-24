# Fraud Detection MLP

This project implements different ML model's for fraud detection using financial data. These models are trained on a dataset with 28 raw financial Items, 14 financial ratios and also combining both as input features to predict fraudulent activities.

## Folder Structure

The project has the following folder structure:

- `data/`: A directory containing data files for training and testing the ML model.
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
- `results/`: A directory containing Python scripts and files for data processing and model evaluation.
  - `data_processing.py`: Python script for data processing.
  - `features.json`: JSON file specifying feature mapping.
  - `models.py`: Python script for model-related functions.
  - `utils.py`: Utility functions.
- `README.md`: This README file.


