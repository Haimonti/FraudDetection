# Fraud Detection MLP

This project implements a Multilayer Perceptron (MLP) model for fraud detection using financial data. The MLP model is trained on a dataset with 28 financial input features to predict fraudulent activities.

## Folder Structure

The project has the following folder structure:

- `.gitignore`: Specifies files and directories to be ignored by Git version control.
- `data_FraudDetection_JAR2020.csv`: The dataset file for training and testing the MLP model.
- `MLP/`: A directory containing the code for the MLP model.
  - `ensemble.py`: A Python script that trains an AdaBoost ensemble classifier using a DecisionTreeClassifier as the base model and evaluates its performance.
  - `features.json`: JSON file specifying the mapping of features in the dataset.
  - `main.py`: The main script for the MLP model.
  - `utils.py`: Utility functions used in the MLP model.
  - `__pycache__/`: A directory containing cached Python files.
- `WordLists`: Folder contains word lists used in different papers.
- `MLP_Bao_finanancial_data.ipynb`: A Jupyter Notebook file containing the MLP model implementation with detailed explanations.
- `RusBoost, XgBoost, SVM.ipynb`: A Jupyter Notebook file containing the implementation of the MLP model with different train and test periods, sliding window methods, and RusBoost (AdaBoost with Random Undersampling), XgBoost and SVM implementation.
- `model-tuning.ipynb`: A Jupyter Notebook file containing various evaluations of CNN (Convolutional Neural Network) and MLP models over the data, exploring different model configurations and tuning hyperparameters.
- `README.md`: This README file.
- `requirements.txt`: A text file specifying the required Python packages and their versions.
- `venv/`: The virtual environment directory containing the Python environment for this project.

## Usage

To train and evaluate the MLP or ensemble model, follow these steps:

1. Install the required dependencies specified in `requirements.txt` by running `pip install -r requirements.txt`.
2. Run the `main.py` script inside the `MLP/` directory to train the MLP model and generate the evaluation results.

For more detailed implementation and explanations, refer to the `MLP_Bao_finanancial_data.ipynb` Jupyter Notebook.

For more details about the different train and test periods, sliding window methods, and RusBoost implementation, refer to the `RusBoost.ipynb` Jupyter Notebook.

To explore various model configurations and tune hyperparameters for the MLP and CNN models, refer to the `model-tuning.ipynb` Jupyter Notebook.

Note: The dataset file `data_FraudDetection_JAR2020.csv` should be present in the project directory before running the scripts or notebooks.

