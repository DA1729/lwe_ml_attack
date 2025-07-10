# LWE Machine Learning Attack

A comprehensive implementation of machine learning attacks against the Learning With Errors (LWE) cryptographic problem. This project demonstrates how neural networks can be trained to predict secret key bits from LWE ciphertexts, effectively breaking LWE-based encryption schemes under certain conditions.

⚠️ **Disclaimer**: This project is for educational and research purposes only. It demonstrates potential vulnerabilities in cryptographic implementations and should not be used maliciously.

## 🎯 Project Overview

The Learning With Errors (LWE) problem is a fundamental building block for post-quantum cryptography. This project explores how machine learning techniques can potentially compromise LWE security by:

- Training individual neural networks to predict each bit of the secret key
- Using LWE samples as training data to learn secret patterns
- Demonstrating complete secret key recovery through bit-wise prediction
- Evaluating attack effectiveness across different LWE parameters

## 🏗️ Architecture

### Core Components

```
lwe_ml_attack/
├── src/                   # Core implementation
│   ├── lwe_crypto.py      # LWE cryptographic primitives
│   ├── attack_engine.py   # Main attack orchestration
│   ├── model_trainer.py   # Neural network training
│   ├── data_generator.py  # Training data generation
│   └── lwe.cpp            # C++ implementation of LWE encryption/decryption
├── demos/                 # Demonstration scripts
│   ├── simple_demo.py     # Basic attack demonstration
│   ├── fast_demo.py       # Quick demo with subset of models
│   └── full_demo.py       # Complete attack with all models
├── models/                # Trained model storage
├── data/                  # Training data and secrets
├── docs/                  # Documentation
└── results/               # Attack results and analysis
```

### Attack Methodology

1. **Data Generation**: Generate LWE samples using C++ implementation
2. **Model Training**: Train 512 individual neural networks (one per secret bit)
3. **Attack Execution**: Use trained models to predict secret bits from new ciphertexts
4. **Secret Recovery**: Reconstruct complete secret key from bit predictions
5. **Decryption**: Use recovered secret to decrypt messages

## 🚀 Quick Start

### Prerequisites

```bash
# Python dependencies
pip install numpy tensorflow scikit-learn matplotlib

# System dependencies
sudo apt-get install g++ build-essential
```

### Installation

```bash
git clone https://github.com/DA1729/lwe_ml_attack.git
cd lwe_ml_attack
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run quick demonstration
python demos/simple_demo.py

# Train new models (if needed)
python src/model_trainer.py --chunk-size 100

# Run full attack demonstration
python demos/full_demo.py
```

## 🎮 Demonstrations

### 1. Simple Demo (`demos/simple_demo.py`)
- Loads 50 pre-trained models
- Shows detailed secret key comparison
- Demonstrates partial secret recovery
- **Runtime**: ~30 seconds

### 2. Fast Demo (`demos/fast_demo.py`)
- Loads 20 models for quick testing
- Interactive bit prediction
- Side-by-side secret comparison
- **Runtime**: ~15 seconds

### 3. Full Demo (`demos/full_demo.py`)
- Loads all 512 trained models
- Complete secret key recovery
- End-to-end attack demonstration
- **Runtime**: ~5 minutes

## 📊 Results

### Attack Performance

| Metric | Value |
|--------|-------|
| Models Trained | 512 |
| Secret Bits | 512 |
| Average Model Accuracy | 100% |
| Secret Recovery Rate | 100% |
| Decryption Success Rate | 100% |

### Example Output

```
=== Secret Key Comparison (First 20 bits) ===
Bit  | Original | Predicted | Confidence | Match
------------------------------------------------
  0  |    1     |     1     |   1.000    |  ✓
  1  |    1     |     1     |   1.000    |  ✓
  2  |    1     |     1     |   1.000    |  ✓
...
Accuracy: 20/20 (100.0%)

=== Side-by-Side Secret Key Comparison ===
Original:  1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 0 0 0 1 0
Predicted: 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 0 0 0 1 0
Match:     ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
```

## 🔬 Technical Details

### LWE Parameters
- **Dimension (n)**: 512
- **Modulus (q)**: 12289
- **Plaintext space (p)**: 4
- **Noise standard deviation**: 3.2

### Neural Network Architecture
- **Input layer**: 512 neurons (LWE vector a)
- **Hidden layers**: 256 → 128 → 64 neurons
- **Output layer**: 1 neuron (binary classification)
- **Activation**: ReLU (hidden), Sigmoid (output)
- **Optimizer**: Adam (lr=0.001)

### Training Details
- **Training samples**: 5,000 per model
- **Epochs**: 30-50 per model
- **Batch size**: 32
- **Validation split**: 20%
- **Training time**: ~2-3 hours for all 512 models

## ⚠️ Important Notes

### Security Considerations

1. **Educational Purpose**: This attack demonstrates theoretical vulnerabilities
2. **Parameter Choice**: Uses potentially weak LWE parameters for demonstration
3. **Perfect Confidence**: Achieving 100% confidence may indicate overfitting
4. **Real-world LWE**: Production systems use much stronger parameters

### Limitations

- **Simplified Scenario**: May not reflect real cryptographic conditions
- **Parameter Dependency**: Success heavily depends on LWE parameter choices
- **Computational Cost**: Requires training hundreds of neural networks
- **Overfitting Risk**: Perfect accuracy suggests potential methodology issues

## 📈 Research Implications

This project demonstrates:
- **ML Threat to Cryptography**: Neural networks can potentially break certain cryptographic schemes
- **Parameter Selection**: Importance of proper LWE parameter choice
- **Side-channel Analysis**: How ML can exploit patterns in cryptographic implementations
- **Defense Strategies**: Need for ML-resistant cryptographic designs

## 🛠️ Development

### Project Structure
```python
# Core attack engine
from src.attack_engine import LWEAttack
from src.lwe_crypto import encrypt_message, decrypt_message

# Initialize attack
attack = LWEAttack(model_dir="models/")
attack.load_models()

# Perform attack
secret_key = attack.recover_secret(ciphertext)
```

### Adding New Features
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Implement changes in appropriate modules
4. Add tests and documentation
5. Submit pull request

## 📚 References

1. Regev, O. "On lattices, learning with errors, random linear codes, and cryptography." STOC 2005.
2. Lyubashevsky, V., Peikert, C., & Regev, O. "On ideal lattices and learning with errors over rings." EUROCRYPT 2010.
3. Recent ML attacks on lattice-based cryptography research papers.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📞 Contact

- **Email**: dakshpandey177@gmail.com
- **GitHub**: [@DA1729](https://github.com/DA1729)

---

**⚠️ Ethical Use**: This software is provided for educational and research purposes. Users are responsible for ensuring ethical and legal use of this technology.
