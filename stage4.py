import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def generate_b_only_dataset(num_samples, n, q, sigma):
    """Generates a dataset of LWE b-values and uniformly random b-values."""
    s = np.random.randint(0, 2, size=n, dtype=np.int64)
    num_positive = num_samples // 2
    a_positive = np.random.randint(0, q, size=(num_positive, n), dtype=np.int64)
    e = np.round(np.random.normal(0, sigma, size=num_positive)).astype(np.int64)
    b_lwe = (a_positive @ s + e) % q
    y_lwe = np.ones(num_positive, dtype=np.int64)
    num_negative = num_samples - num_positive
    b_uniform = np.random.randint(0, q, size=num_negative, dtype=np.int64)
    y_uniform = np.zeros(num_negative, dtype=np.int64)
    x = np.hstack([b_lwe, b_uniform])
    y = np.hstack([y_lwe, y_uniform])
    indices = np.random.permutation(len(x))
    return x[indices].reshape(-1, 1), y[indices]

def circular_embedding(data, q):
    """Transforms linear integer data into a circular 2D representation."""
    theta = 2 * np.pi * data / q
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)

def create_distinguisher_model(input_shape):
    """Defines a 1D CNN optimized for analyzing the 1D signal of b-values."""
    l2_reg = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu', padding='same', kernel_regularizer=l2_reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu', padding='same', kernel_regularizer=l2_reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    n = 128
    sigmas_to_test = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    q = 4093
    num_samples = 50000 
    results = {}

    print("--- Stage 4: Focused LWE Security Analysis on b-value Distribution ---")
    total_start_time = time.time()

    for sigma in sigmas_to_test:
        print(f"\n----- Testing (n={n}, sigma={sigma}) -----")
        x_raw, y = generate_b_only_dataset(num_samples, n, q, sigma)
        x_embedded = circular_embedding(x_raw, q)
        x_train, x_test, y_train, y_test = train_test_split(x_embedded, y, test_size=0.2, random_state=42)
        model = create_distinguisher_model(input_shape=(1, 2))
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=0)
        model.fit(x_train, y_train, epochs=100, batch_size=256, validation_data=(x_test, y_test),
                  callbacks=[early_stopping], verbose=0)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        results[sigma] = accuracy
        print(f"-> Result: {accuracy*100:.2f}% accuracy.")

    print("\nGenerating final plot: b_value_accuracy_vs_sigma.png")
    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='--')
    plt.axhline(y=0.5, color='r', linestyle=':', label='Random Guessing (50%)')
    plt.title(f'Distinguisher Accuracy vs. Noise (n={n}, b-value only)', fontsize=16)
    plt.xlabel('Noise Standard Deviation (sigma)', fontsize=12)
    plt.ylabel('Final Test Accuracy', fontsize=12)
    plt.xscale('log')
    plt.xticks(sigmas_to_test, labels=[str(s) for s in sigmas_to_test])
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.ylim(0.45, 1.05)
    plt.savefig('b_value_accuracy_vs_sigma.png')
    print("Analysis complete. Plot saved successfully.")

