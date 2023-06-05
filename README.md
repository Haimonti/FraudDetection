# Fraud Detection MLP

This project implements a Multilayer Perceptron (MLP) model for fraud detection using financial data. The MLP model is trained on a dataset with 28 financial input features to predict fraudulent activities.

## Folder Structure

The project has the following folder structure:

- `.gitignore`: Specifies files and directories to be ignored by Git version control.
- `data_FraudDetection_JAR2020.csv`: The dataset file for training and testing the MLP model.
- `MLP/`: A directory containing the code for the MLP model.
  - `features.json`: JSON file specifying the mapping of features in the dataset.
  - `main.py`: The main script for the MLP model.
  - `utils.py`: Utility functions used in the MLP model.
  - `__pycache__/`: A directory containing cached Python files.
- `MLP_Bao_finanancial_data.ipynb`: A Jupyter Notebook file containing the MLP model implementation with detailed explanations.
- `README.md`: The README file you're currently reading.
- `requirements.txt`: A text file specifying the required Python packages and their versions.
- `venv/`: The virtual environment directory containing the Python environment for this project.

## Usage

To train and evaluate the MLP model, follow these steps:

1. Install the required dependencies specified in `requirements.txt` by running `pip install -r requirements.txt`.
2. Run the `main.py` script inside the `MLP/` directory to train the MLP model and generate the evaluation results.

For more detailed implementation and explanations, refer to the `MLP_Bao_finanancial_data.ipynb` Jupyter Notebook.


