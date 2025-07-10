#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

constexpr int n = 512;           // security parameter
constexpr int q = 12289;         // large prime modulus
constexpr int p = 4;             // message space {0,1,2,3}
constexpr int delta = q / p;     // scaling factor for encoding

// noise sampling
int sample_discrete_gaussian(std::mt19937& gen, double sigma = 3.2) {
    std::normal_distribution<> dist(0.0, sigma);
    return static_cast<int>(std::round(dist(gen))) % q;
}

// secret key
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

// encryption
std::pair<std::vector<int>, int> encrypt(int m, const std::vector<int>& s, std::mt19937& gen) {
    std::uniform_int_distribution<> uniform_q(0, q - 1);

    std::vector<int> a(n);
    for (int& ai : a) ai = uniform_q(gen);

    int e = sample_discrete_gaussian(gen);
    int b = (dot_mod_q(a, s) + delta * m + e) % q;

    return {a, b};
}

// decryption
int decrypt(const std::pair<std::vector<int>, int>& ct, const std::vector<int>& s) {
    const std::vector<int>& a = ct.first;
    int b = ct.second;

    int phase = (b - dot_mod_q(a, s) + q) % q;
    return static_cast<int>(std::round((double)phase / delta)) % p;
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int> s = key_gen(gen);

    std::cout << "Testing encryption over message space {0,1,2,3}:\n";

    for (int m = 0; m < p; ++m) {
        auto ct = encrypt(m, s, gen);
        int recovered = decrypt(ct, s);
        std::cout << "Original: " << m
                  << " | Decrypted: " << recovered
                  << ((m != recovered) ? " ❌" : " ✅") << "\n";
    }

    return 0;
}
