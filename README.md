---

# Neural Network Regression with Dropout Comparison

This repository implements a neural network regression model in TensorFlow/Keras to predict values using synthetic data, with a comparison between models with and without dropout regularization. The model is trained on a simple dataset with features and targets, and the performance is evaluated using mean squared error (MSE) for both training and test datasets.

## Dataset
- `x_train`, `y_train`: Training data (features and labels)
- `x_test`, `y_test`: Testing data (features and labels)
- **Data Visualization**: Red and blue scatter plots represent training and test data, respectively.

## Model Architectures
1. **Model 1** - Neural Network without Dropout
   - Layers:
     - Dense (128 units, ReLU activation)
     - Dense (128 units, ReLU activation)
     - Dense (1 unit, Linear activation)
   - **Optimizer**: Adam with a learning rate of 0.01
   - **Loss Function**: Mean Squared Error (MSE)
   - **Evaluation Metrics**: MSE for both training and testing data

2. **Model 2** - Neural Network with Dropout
   - Adds `Dropout` layers between dense layers to improve generalization and reduce overfitting.

## Results
- **Model 1 (without Dropout)**: Achieved a training MSE of approximately `0.0066` and a test MSE of `0.0358`.
- **Model 2 (with Dropout)**: Further experiments will show the MSE differences and performance.

## Requirements
- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`

To install the requirements:
```bash
pip install numpy pandas matplotlib tensorflow
```

## Usage

1. **Data Visualization**:
   ```python
   plt.scatter(x_train, y_train, c='red', label='Train')
   plt.scatter(x_test, y_test, c='blue', label='Test')
   plt.legend()
   plt.show()
   ```

2. **Model Training (Without Dropout)**:
   ```python
   model_1 = Sequential()
   model_1.add(Dense(128, input_dim=1, activation='relu'))
   model_1.add(Dense(128, activation='relu'))
   model_1.add(Dense(1, activation='linear'))
   model_1.compile(loss="mse", optimizer=Adam(learning_rate=0.01), metrics=['mse'])
   history_1 = model_1.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test), verbose=False)
   ```

3. **Model Training (With Dropout)**:
   ```python
   from tensorflow.keras.layers import Dropout
   model_2 = Sequential()
   model_2.add(Dense(128, input_dim=1, activation='relu'))
   model_2.add(Dropout(0.5))
   model_2.add(Dense(128, activation='relu'))
   model_2.add(Dropout(0.5))
   model_2.add(Dense(1, activation='linear'))
   ```

4. **Model Evaluation and Plotting**:
   ```python
   y_pred_1 = model_1.predict(x_test)
   plt.scatter(x_train, y_train, c='red', label='Train')
   plt.scatter(x_test, y_test, c='blue', label='Test')
   plt.plot(x_test, y_pred_1, label='Predictions')
   plt.ylim((-1.5, 1.5))
   plt.legend()
   plt.show()
   ```


---

Feel free to modify this template to best suit your project specifics!
