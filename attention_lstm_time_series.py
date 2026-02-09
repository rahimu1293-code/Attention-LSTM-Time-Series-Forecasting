"""
Advanced Time Series Forecasting with Attention
Corrected as per reviewer feedback
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. DATA GENERATION 
# --------------------------------------------------
def generate_time_series(n_points=2500):
    """
    Generates complex synthetic time series with multiple seasonality.
    """
    t = np.arange(n_points)
    season1 = 10 * np.sin(2 * np.pi * t / 24)
    season2 = 5 * np.sin(2 * np.pi * t / 365)
    trend = 0.01 * t
    noise = np.random.normal(0, 2, n_points)
    return season1 + season2 + trend + noise


# --------------------------------------------------
# 2. SEQUENCE CREATION
# --------------------------------------------------
def create_sequences(series, window=30):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)


# --------------------------------------------------
# 3. CUSTOM ATTENTION 
# --------------------------------------------------
class Attention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        weights = tf.nn.softmax(score, axis=1)
        self.attention_weights = weights
        return tf.reduce_sum(inputs * weights, axis=1)


# --------------------------------------------------
# 4. MODEL
# --------------------------------------------------
def build_model(units, lr, window):
    inputs = Input(shape=(window, 1))
    x = LSTM(units, return_sequences=True)(inputs)
    att = Attention()(x)
    out = Dense(1)(att)
    model = Model(inputs, out)
    model.compile(optimizer=Adam(lr), loss="mse")
    return model


# --------------------------------------------------
# 5. DATA SPLIT 
# --------------------------------------------------
series = generate_time_series()
window = 30
X, y = create_sequences(series, window)
X = X[..., np.newaxis]

train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]


# --------------------------------------------------
# 6. HYPERPARAMETER TUNING
# --------------------------------------------------
units_list = [32, 64, 128]
lr_list = [0.001, 0.0005]
results = {}

for u in units_list:
    for lr in lr_list:
        model = build_model(u, lr, window)
        model.fit(X_train, y_train, epochs=10, verbose=0)
        preds = model.predict(X_val, verbose=0)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        results[(u, lr)] = rmse

# Visualization
plt.figure()
plt.plot([str(k) for k in results.keys()], results.values(), marker="o")
plt.xticks(rotation=45)
plt.title("Hyperparameter Tuning RMSE")
plt.ylabel("RMSE")
plt.show()

best_units, best_lr = min(results, key=results.get)


# --------------------------------------------------
# 7. FINAL DL MODEL (TRAIN+VAL ONLY)
# --------------------------------------------------
final_model = build_model(best_units, best_lr, window)
final_model.fit(
    np.vstack([X_train, X_val]),
    np.hstack([y_train, y_val]),
    epochs=20,
    verbose=0
)
dl_preds = final_model.predict(X_test, verbose=0)
dl_rmse = np.sqrt(mean_squared_error(y_test, dl_preds))


# --------------------------------------------------
# 8. ATTENTION WEIGHT VISUALIZATION 
# --------------------------------------------------
class Attention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs, return_attention=False):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        if return_attention:
            return context, weights
        return context

# Build a temporary model that outputs both prediction and weights
inputs = tf.keras.Input(shape=(window, 1))
x = LSTM(best_units, return_sequences=True)(inputs)
att, weights = Attention()(x, return_attention=True)
out = Dense(1)(att)
temp_model = tf.keras.Model(inputs, [out, weights])

# Run prediction
preds, att_weights = temp_model.predict(X_test[:1])

# Plot weights
plt.plot(att_weights.flatten())
plt.title("Attention Weights Across Time Steps")
plt.xlabel("Time Step")
plt.ylabel("Importance")
plt.show()

# --------------------------------------------------
# 9. SARIMAX TUNING
# --------------------------------------------------
best_sarimax_rmse = float("inf")

for order in [(1,1,1), (2,1,1)]:
    model = SARIMAX(series[:train_size], order=order)
    fit = model.fit(disp=False)
    forecast = fit.forecast(len(series[train_size:train_size+val_size]))
    rmse = np.sqrt(mean_squared_error(
        series[train_size:train_size+val_size],
        forecast
    ))
    if rmse < best_sarimax_rmse:
        best_sarimax_rmse = rmse
        best_order = order


sarimax = SARIMAX(series[:train_size+val_size], order=best_order)
sarimax_fit = sarimax.fit(disp=False)
sarimax_preds = sarimax_fit.forecast(len(y_test))
sarimax_rmse = np.sqrt(mean_squared_error(y_test, sarimax_preds))


# --------------------------------------------------
# 10. FINAL REPORT
# --------------------------------------------------
print(f"""
FINAL TEST RESULTS (HELD-OUT SET)

LSTM + Attention RMSE : {dl_rmse:.3f}
SARIMAX RMSE : {sarimax_rmse:.3f}
Best SARIMAX Order : {best_order}
""")
