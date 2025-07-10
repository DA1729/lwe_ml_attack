"""
LWE Cryptographic Primitives

This module implements the basic Learning With Errors (LWE) cryptographic operations
including key generation, encryption, and decryption.
"""

import numpy as np
import subprocess
import tempfile
import os
from typing import Tuple, List

# LWE Parameters
N = 512          # Dimension
Q = 12289        # Modulus
P = 4            # Plaintext space
DELTA = Q // P   # Scaling factor
NOISE_STD = 3.2  # Noise standard deviation


class LWECrypto:
    """LWE cryptographic operations"""
    
    def __init__(self, n: int = N, q: int = Q, p: int = P, noise_std: float = NOISE_STD):
        self.n = n
        self.q = q
        self.p = p
        self.delta = q // p
        self.noise_std = noise_std
        
    def generate_secret_key(self) -> np.ndarray:
        """Generate a random binary secret key"""
        return np.random.randint(0, 2, self.n)
        
    def sample_gaussian_noise(self) -> int:
        """Sample discrete Gaussian noise"""
        return int(np.round(np.random.normal(0, self.noise_std))) % self.q
        
    def encrypt(self, message: int, secret_key: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Encrypt a message using LWE
        
        Args:
            message: Message to encrypt (0 to p-1)
            secret_key: Secret key vector
            
        Returns:
            Tuple of (a_vector, b_value) representing the ciphertext
        """
        # Generate random vector a
        a = np.random.randint(0, self.q, self.n)
        
        # Sample noise
        e = self.sample_gaussian_noise()
        
        # Compute b = <a, s> + delta * m + e (mod q)
        dot_product = np.dot(a, secret_key) % self.q
        b = (dot_product + self.delta * message + e) % self.q
        
        return a, b
        
    def decrypt(self, ciphertext: Tuple[np.ndarray, int], secret_key: np.ndarray) -> int:
        """
        Decrypt a ciphertext using LWE
        
        Args:
            ciphertext: Tuple of (a_vector, b_value)
            secret_key: Secret key vector
            
        Returns:
            Decrypted message
        """
        a, b = ciphertext
        
        # Compute dot product
        dot_product = np.dot(a, secret_key) % self.q
        
        # Recover noisy message
        noisy_message = (b - dot_product) % self.q
        
        # Decode message by rounding
        message = round(noisy_message / self.delta) % self.p
        
        return message


class LWEDataGenerator:
    """Generate LWE training data using C++ implementation"""
    
    def __init__(self):
        self.cpp_code = '''
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

constexpr int n = 512;
constexpr int q = 12289;
constexpr int p = 4;
constexpr int delta = q / p;

int sample_discrete_gaussian(std::mt19937& gen, double sigma = 3.2) {
    std::normal_distribution<> dist(0.0, sigma);
    return static_cast<int>(std::round(dist(gen))) % q;
}

std::vector<int> key_gen(std::mt19937& gen) {
    std::vector<int> s(n);
    std::bernoulli_distribution bern(0.5);
    for (int& si : s) si = bern(gen);
    return s;
}

int dot_mod_q(const std::vector<int>& a, const std::vector<int>& b) {
    assert(a.size() == b.size());
    int64_t sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += static_cast<int64_t>(a[i]) * b[i];
    return static_cast<int>(sum % q);
}

std::pair<std::vector<int>, int> encrypt(int m, const std::vector<int>& s, std::mt19937& gen) {
    std::uniform_int_distribution<> uniform_q(0, q - 1);
    std::vector<int> a(n);
    for (int& ai : a) ai = uniform_q(gen);
    int e = sample_discrete_gaussian(gen);
    int b = (dot_mod_q(a, s) + delta * m + e) % q;
    return {a, b};
}

int main(int argc, char* argv[]) {
    int num_samples = 5000;
    if (argc > 1) num_samples = std::atoi(argv[1]);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> s = key_gen(gen);

    // Output secret key
    std::cout << "SECRET:";
    for (int si : s) std::cout << " " << si;
    std::cout << std::endl;

    // Generate training samples
    for (int i = 0; i < num_samples; ++i) {
        auto ct = encrypt(0, s, gen);  // Always encrypt 0 for key recovery
        std::cout << "SAMPLE:";
        for (int ai : ct.first) std::cout << " " << ai;
        std::cout << " " << ct.second << std::endl;
    }
    return 0;
}
'''

    def generate_data(self, num_samples: int = 5000) -> Tuple[np.ndarray, List[Tuple[List[int], int]]]:
        """
        Generate LWE training data using C++ implementation
        
        Args:
            num_samples: Number of LWE samples to generate
            
        Returns:
            Tuple of (secret_key, samples_list)
        """
        # Create temporary C++ file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(self.cpp_code)
            cpp_file = f.name

        try:
            # Compile
            exe_file = cpp_file.replace('.cpp', '')
            subprocess.run(['g++', '-o', exe_file, cpp_file], check=True)

            # Run and capture output
            result = subprocess.run([exe_file, str(num_samples)], 
                                  capture_output=True, text=True, check=True)

            # Parse output
            lines = result.stdout.strip().split('\n')

            # Extract secret key
            secret_line = [line for line in lines if line.startswith('SECRET:')][0]
            secret = np.array(list(map(int, secret_line.split()[1:])))

            # Extract samples
            sample_lines = [line for line in lines if line.startswith('SAMPLE:')]
            samples = []
            for line in sample_lines:
                parts = list(map(int, line.split()[1:]))
                a = parts[:-1]  # First n elements
                b = parts[-1]   # Last element
                samples.append((a, b))

            return secret, samples

        finally:
            # Cleanup
            for file in [cpp_file, exe_file]:
                if os.path.exists(file):
                    os.unlink(file)


def encrypt_message(message: int, secret_key: np.ndarray, 
                   n: int = N, q: int = Q, noise_std: float = NOISE_STD) -> Tuple[np.ndarray, int]:
    """
    Convenience function to encrypt a message
    
    Args:
        message: Message to encrypt
        secret_key: Secret key
        n: LWE dimension
        q: LWE modulus
        noise_std: Noise standard deviation
        
    Returns:
        LWE ciphertext (a, b)
    """
    crypto = LWECrypto(n, q, P, noise_std)
    return crypto.encrypt(message, secret_key)


def decrypt_message(ciphertext: Tuple[np.ndarray, int], secret_key: np.ndarray,
                   n: int = N, q: int = Q) -> int:
    """
    Convenience function to decrypt a message
    
    Args:
        ciphertext: LWE ciphertext (a, b)
        secret_key: Secret key
        n: LWE dimension
        q: LWE modulus
        
    Returns:
        Decrypted message
    """
    crypto = LWECrypto(n, q, P)
    return crypto.decrypt(ciphertext, secret_key)


def create_training_data(secret: np.ndarray, samples: List[Tuple[List[int], int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert LWE samples to ML training data
    
    Args:
        secret: Secret key
        samples: List of LWE samples
        
    Returns:
        Tuple of (X_train, y_train) where X is input vectors and y is secret
    """
    X = []
    for a, b in samples:
        X.append(a)
    
    return np.array(X), secret