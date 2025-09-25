# An Empirical Analysis of LWE Robustness Against Machine Learning Distinguishers

This repository contains the Python scripts and findings from a research project investigating the practical security boundaries of the Learning With Errors (LWE) cryptosystem. The goal of this analysis was not to "break" LWE, but to empirically measure its robustness against a series of increasingly sophisticated machine learning-based distinguishers.

This work serves as a companion to the detailed blog post, **"An Empirical Analysis of LWE Robustness Against Machine Learning Distinguishers"**.

## Project Overview

The security of LWE relies on the Decision-LWE assumption, which posits that LWE-generated samples are computationally indistinguishable from uniformly random data. This project tests that assumption by training a series of neural networks to act as distinguishers.

The investigation proceeded through a multi-stage analytical process, where each stage's failure informed the next, more refined attempt. This iterative process is a core part of the research and demonstrates key challenges and concepts in both applied machine learning and cryptanalysis.

## The Analytical Journey: Four Stages of Investigation

The repository is organized into four distinct scripts, each representing a critical stage in our analysis.

### Stage 1: Baseline MLP (`stage_1_mlp.py`)

**Hypothesis:** A standard Multi-Layer Perceptron (MLP) can distinguish LWE samples (a, b) from random noise.

**Methodology:** A simple MLP was trained on the raw, high-dimensional (a, b) data.

**Outcome:** The model failed to generalize, exhibiting severe overfitting. While training accuracy increased, validation accuracy remained at ~50%. This demonstrated the inherent difficulty of the problem.

### Stage 2: MLP with Circular Embedding (`stage_2_mlp_circular.py`)

**Hypothesis:** The model's failure in Stage 1 was due to a data representation mismatch. An MLP with features engineered to respect modular arithmetic will succeed.

**Methodology:** We introduced a circular embedding to represent the modular integers x as 2D vectors (cos(θ), sin(θ)), providing the model with the concept of modular proximity.

**Outcome:** The model still failed to generalize, again due to overfitting. This crucial result indicated that the problem was not just data representation but likely architectural.

### Stage 3: Regularized 1D CNN (`stage_3_cnn_regularized.py`)

**Hypothesis:** The LWE problem has a sequential structure that requires a more specialized architecture. A 1D Convolutional Neural Network (CNN) is better suited for this task.

**Methodology:** The MLP was replaced with a regularized 1D CNN to capture local patterns in the (a, b) sequence.

**Outcome:** The training accuracy skyrocketed, proving the CNN was a more capable learner. However, validation accuracy remained flat at 50%. This was the definitive demonstration of the "curse of dimensionality"—the LWE signal was too diluted in the high-dimensional space for even a powerful model to find a generalizable pattern.

### Stage 4: Focused b-value Analysis (`stage_4_focused_b_analysis.py`)

**Hypothesis:** If the full (a, b) problem is intractable, a focused analysis might reveal a statistical bias in the distribution of the b values alone.

**Methodology:** We used our most sophisticated model (the regularized 1D CNN with circular embeddings) on a simplified task: distinguishing LWE-generated b values from uniformly random integers.

**Outcome:** The model failed to distinguish the distributions, with accuracy remaining at 50%. This was the final and most important finding: the LWE process, when well-parameterized, produces an output distribution that is statistically indistinguishable from random noise, even for an advanced, purpose-built ML model.

*An example of a potential outcome from the Stage 4 analysis, showing the phase transition from a breakable to a secure parameter set.*

## How to Run the Experiments

### Prerequisites

- Python 3.8+
- TensorFlow
- Scikit-learn
- Matplotlib
- Seaborn

Install the required packages using pip:

```bash
pip install tensorflow scikit-learn matplotlib seaborn
```

### Execution

Each script is self-contained and can be run directly from the terminal. For example, to run the final, focused analysis:

```bash
python stage_4_focused_b_analysis.py
```

Each script will print its progress to the console and, where applicable, save a plot of its results as a .png file.

## Conclusion

This repository documents a systematic, empirical investigation that ultimately serves as a testament to the resilience of the LWE problem. Our multi-stage analysis, involving iterative improvements to both the model architecture and data representation, consistently demonstrated that well-parameterized LWE is robust against this class of machine learning attacks. The project highlights fundamental concepts in both fields, including the curse of dimensionality, the importance of data representation, and the nature of statistical security in modern cryptography.
