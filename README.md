# AI-Driven Image Classification and Analysis

## Overview

This project implements an AI-driven image classification pipeline using a neural network and logistic regression model. It processes a dataset of handwritten digit images, trains models, visualizes results, and interprets model predictions using SHAP (SHapley Additive exPlanations).

## Features

- **Data Preprocessing**: Reads and processes image data.
- **Model Training**:
  - Logistic Regression Model
  - Neural Network Model
- **Evaluation & Visualization**:
  - Training history (accuracy & loss)
  - Predictions with true labels
  - SHAP-based interpretability analysis

## Installation

To run this project, install the necessary dependencies:

```bash
pip install numpy pandas matplotlib keras idx2numpy shap
```

## Dataset

- The dataset (`data.csv`) should be placed inside `./dataset_aiml_task/`.
- The images dataset (`images-idx3-ubyte`) should also be in the same directory.

## Usage

Run the script using:

```bash
python main.py
```

This will:

1. Display sample images.
2. Load and summarize dataset statistics.
3. Train and evaluate the logistic regression and neural network models.
4. Visualize model predictions.
5. Generate SHAP interpretability plots.

## Code Structure

- **`plot_figure()`**: Displays sample images from the dataset.
- **`stats()`**: Loads image data and prints dataset statistics.
- **`visualize()`**: Plots pixel intensity distribution.
- **`plot_training_history(history, model_name)`**: Displays training accuracy and loss curves.
- **`visualize_predictions(x_test, y_pred, y_true, model_name)`**: Shows model predictions with true labels.
- **`analyze(model, x_test, x_train)`**: Performs SHAP analysis to interpret model decisions.
- **`create_model()`**: Trains the logistic regression and neural network models.
- **`main()`**: Runs the entire pipeline.

## Interpretation using SHAP

SHAP (SHapley Additive exPlanations) is used to understand how input features influence model predictions. The script generates:

1. A **summary plot**, highlighting feature importance.
2. A **beeswarm plot**, showing how individual pixels contribute to predictions.

## Example Output

After running the script, expect the following outputs:

- **Training accuracy and loss graphs**
- **Predicted vs. actual labels for test samples**
- **SHAP-based interpretability plots**

## License

This project is licensed under the MIT License.

