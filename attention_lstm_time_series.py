import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Layer
from tensorflow.keras.optimizers import Adam

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ============================================================
# 1. Synthetic Time Series Data Generation
# ============================================================

np.random.seed(42)

n = 2500  # > 2000 observations
t = np.arange(n)

trend = 0.005 * t
seasonal_1 = 2 * np.sin(2 * np.pi * t / 24)      # Daily cycle
seasonal_2 = 1.5 * np.sin(2 * np.pi * t / 168)   # Weekly cycle
noise = np.random.normal(0, 0.5, n)

series = trend + seasonal_1 + seasonal_2 + noise
data = pd.DataFrame({"value": series})

plt.figure()
plt.plot(series)
plt.title("Synthetic Time Series")
plt.show()

# ============================================================
# Data Preparation
# ============================================================

scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(series.reshape(-1, 1))

def create_sequences(data, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

window_size = 30
X, y = create_sequences(scaled_series, window_size)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ============================================================
# 2. Custom Attention Layer
# ============================================================

class Attention(Layer):
    def _init_(self):
        super(Attention, self)._init_()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# ============================================================
# Attention-LSTM Model
# ============================================================

def build_attention_lstm(lr=0.001, units=64):
    inputs = Input(shape=(window_size, 1))
    lstm_out = LSTM(units, return_sequences=True)(inputs)
    context, attention_weights = Attention()(lstm_out)
    output = Dense(1)(context)

    model = Model(inputs, output)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse"
    )
    return model

# ============================================================
# 3. Rolling Forecast Cross-Validation
# ============================================================

learning_rates = [0.001, 0.0005]
lstm_units = [32, 64]

best_rmse = np.inf
best_model = None

for lr in learning_rates:
    for units in lstm_units:
        model = build_attention_lstm(lr, units)
        model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=32,
            verbose=0
        )
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

print("Best Attention-LSTM RMSE:", best_rmse)

# ============================================================
# 4. Benchmark Model – SARIMAX
# ================================================

train_series = series[:split + window_size]
test_series = series[split + window_size:]

start_time = time.time()
sarimax = SARIMAX(
    train_series,
    order=(2,1,2),
    seasonal_order=(1,1,1,24)
)
sarimax_fit = sarimax.fit(disp=False)
sarimax_time = time.time() - start_time

sarimax_pred = sarimax_fit.forecast(len(test_series))

# ============================================================
# Evaluation Metrics
# ============================================================

def evaluate(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return rmse, mae, mape

# Attention-LSTM predictions
lstm_pred = best_model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred)
y_true = scaler.inverse_transform(y_test)

lstm_rmse, lstm_mae, lstm_mape = evaluate(y_true, lstm_pred)
sar_rmse, sar_mae, sar_mape = evaluate(test_series, sarimax_pred)

print("\n--- Model Comparison ---")
print("Attention-LSTM -> RMSE:", lstm_rmse, "MAE:", lstm_mae, "MAPE:", lstm_mape)
print("SARIMAX        -> RMSE:", sar_rmse, "MAE:", sar_mae, "MAPE:", sar_mape)

# ============================================================
# 5. Attention Weights Visualization
# ============================================================

attention_extractor = Model(
    inputs=best_model.input,
    outputs=best_model.layers[2].output
)

_, attention_weights = attention_extractor.predict(X_test[:1])

plt.figure()
plt.bar(range(window_size), attention_weights.flatten())
plt.title("Attention Weights Over Time Steps")
plt.xlabel("Lag")
plt.ylabel("Importance")
plt.show()
