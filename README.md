# Lab Task: Linear Regression in Google Colab

This README provides an outline of the lab tasks for implementing and comparing linear regression models in Google Colab.

---

## Coding Exercise 1: Run Built-in Linear Regression Model

1. **Upload the Dataset**
   - Load the "student grades" dataset into Google Colab using the provided loader cell.

2. **Run the Built-in Linear Regression Model**
   - Use a prebuilt linear regression model to predict the target variable (e.g., student grades).

---

## Coding Exercise 2: Implement a Manual Linear Regression Model

### Algorithm Implementation:

#### 1. Initialization:
- **Weights and Bias**:
  - Initialize weights (`theta`) and bias (`theta_0`) to random values or zeros.
- **Hyperparameters**:
  - Define the learning rate (`alpha`) and the number of iterations (`num_iterations`).

#### 2. Training Loop:
Iterate `num_iterations` times:
- **Predictions**:
  - Compute predictions:  
    \[
    \hat{y} = X \cdot \theta + \theta_0
    \]
- **Cost Function**:
  - Calculate the Mean Squared Error (MSE):  
    \[
    J = \frac{1}{2m} \sum (\hat{y} - y)^2
    \]
- **Gradients**:
  - Compute gradients:  
    \[
    d\theta = \frac{1}{m} \cdot X^T \cdot (\hat{y} - y)
    \]  
    \[
    d\theta_0 = \frac{1}{m} \cdot \sum (\hat{y} - y)
    \]
- **Update Parameters**:
  - Update weights and bias:  
    \[
    \theta = \theta - \alpha \cdot d\theta
    \]  
    \[
    \theta_0 = \theta_0 - \alpha \cdot d\theta_0
    \]

#### 3. Prediction:
- For new input features (`X_test`), predict the target variable (`y_pred`):  
  \[
  y_{\text{pred}} = X_{\text{test}} \cdot \theta + \theta_0
  \]

---

### Manual Model Functions:
python
class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.alpha = learning_rate
        self.num_iterations = num_iterations
    def train(self, X, y):
        # Initialize parameters
        self.theta = np.zeros(X.shape[1])
        self.theta_0 = 0

        m = len(y)  # Number of training examples

        for _ in range(self.num_iterations):
            # Compute predictions
            y_hat = X.dot(self.theta) + self.theta_0

            # Compute cost
            cost = (1 / (2 * m)) * np.sum((y_hat - y) ** 2)

            # Compute gradients
            dtheta = (1 / m) * X.T.dot(y_hat - y)
            dtheta_0 = (1 / m) * np.sum(y_hat - y)

            # Update parameters
            self.theta -= self.alpha * dtheta
            self.theta_0 -= self.alpha * dtheta_0

    def predict(self, X_test):
        # Compute predictions
        return X_test.dot(self.theta) + self.theta_0
