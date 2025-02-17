# Logistic Regression from Scratch

This repository contains a simple implementation of **Logistic Regression** from scratch using **NumPy**. The model is trained using gradient descent to classify data into binary classes.

## Description

Logistic Regression is a supervised machine learning algorithm commonly used for binary classification tasks. This implementation uses the sigmoid function to output probabilities and the log loss function to evaluate the model during training. The model is trained using gradient descent to minimize the loss function.

## Files

- `logistic_regression.py`: Contains the `LogisticRegression` class with methods for training the model (`fit`), predicting values (`predict`), and calculating the log loss (`log_loss`).

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/logistic-regression-from-scratch.git
   ```

2. Import the `LogisticRegression` class:

   ```python
   from logistic_regression import LogisticRegression
   ```

3. Create an instance of the `LogisticRegression` class:

   ```python
   model = LogisticRegression(lr=0.001, epochs=500)
   ```

4. Fit the model with your training data (`X` as features and `y` as labels):

   ```python
   model.fit(X_train, y_train)
   ```

5. Make predictions:

   ```python
   y_pred = model.predict(X_test)
   ```

## Methods

### `__init__(self, epochs=500, lr=0.001)`
- Initializes the learning rate (`lr`) and the number of epochs (`epochs`).

### `sigmoid(self, z)`
- The sigmoid activation function that maps values to a range between 0 and 1.

### `log_loss(self, y_pred, y)`
- Computes the log loss (binary cross-entropy) between the predicted probabilities (`y_pred`) and the true labels (`y`).

### `fit(self, X, y)`
- Trains the logistic regression model using gradient descent on the training data (`X` and `y`).

### `predict(self, X)`
- Predicts binary class labels (0 or 1) for new data (`X`).

## Example

```python
import numpy as np
from logistic_regression import LogisticRegression

# Example data (features and labels)
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([[0], [0], [1], [1]])

# Create and train the model
model = LogisticRegression(lr=0.01, epochs=1000)
model.fit(X_train, y_train)

# Predict on new data
X_test = np.array([[5, 6]])
y_pred = model.predict(X_test)

print(f"Prediction: {y_pred}")
```

## Requirements

- Python 3.x
- NumPy

Install dependencies:

```bash
pip install numpy
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
