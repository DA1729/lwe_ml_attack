import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import os

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def generate_dataset(num_samples, n, q, sigma):
    """Generates a balanced dataset of LWE and random (a,b) pairs."""
    s = np.random.randint(0, 2, size=n, dtype=np.int64)
    # Positive samples (LWE)
    num_positive = num_samples // 2
    a_positive = np.random.randint(0, q, size=(num_positive, n), dtype=np.int64)
    e = np.round(np.random.normal(0, sigma, size=num_positive)).astype(np.int64)
    b_positive = (a_positive @ s + e) % q
    x_positive = np.column_stack([a_positive, b_positive])
    y_positive = np.ones(num_positive, dtype=np.int64)
    # Negative samples (Random)
    num_negative = num_samples - num_positive
    x_negative = np.random.randint(0, q, size=(num_negative, n + 1), dtype=np.int64)
    y_negative = np.zeros(num_negative, dtype=np.int64)
    # Combine and shuffle
    x = np.vstack([x_positive, x_negative])
    y = np.hstack([y_positive, y_negative])
    indices = np.random.permutation(len(x))
    return x[indices], y[indices]

def create_mlp_model(input_shape):
    """Defines a standard MLP model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    n = 256
    q = 4093
    sigma = 2.0  # A low sigma that should be breakable
    num_samples = 30000

    print("--- Stage 1: Baseline MLP on raw (a,b) data ---")
    print(f"Parameters: n={n}, q={q}, sigma={sigma}\n")

    x_raw, y = generate_dataset(num_samples, n, q, sigma)
    x_normalized = x_raw.astype(np.float32) / q  # Normalize data
    x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

    model = create_mlp_model(input_shape=(n + 1,))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)

    history = model.fit(
        x_train, y_train, epochs=100, batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping], verbose=1
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")
    print("Expected outcome: Severe overfitting (high train accuracy, low validation accuracy near 50%).")

