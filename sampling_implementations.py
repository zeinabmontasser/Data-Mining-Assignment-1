# ============================================================
# sampling_implementations.py
# Part 2 — Sampling From First Principles
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Simulated Vocabulary and Logits
# ============================================================

vocabulary = ["the", "cat", "sat", "on", "a", "mat",
              "dog", "ran", "fast", "slowly"]

logits = np.array([2.5, 1.2, 0.8, 3.1,
                   -0.5, 1.9, 0.3,
                   -1.2, 2.8, 0.1])

# ============================================================
# Task 1 — Softmax with Temperature
# ============================================================

def softmax_with_temperature(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")

    scaled = logits / temperature
    scaled = scaled - np.max(scaled)  # numerical stability
    exp_vals = np.exp(scaled)
    return exp_vals / np.sum(exp_vals)


# Required Tests
for T in [0.1, 0.5, 1.0, 2.0]:
    probs = softmax_with_temperature(logits, T)
    print(f"\nTemperature = {T}")
    for word, p in zip(vocabulary, probs):
        print(f"{word:8s}: {p:.4f}")


# Required Visualization (4-panel)
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
temperatures = [0.1, 0.5, 1.0, 2.0]

for ax, T in zip(axes, temperatures):
    probs = softmax_with_temperature(logits, T)
    ax.bar(vocabulary, probs)
    ax.set_title(f"Temperature = {T}")
    ax.set_xlabel("Token")
    ax.set_ylabel("Probability")
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("temperature_effect.png", dpi=150)
plt.show()

"""
Low temperature divides logits by a small value, increasing the
differences between them. After exponentiation in softmax,
the largest logit dominates and the distribution becomes
near-deterministic. High temperature shrinks the differences
between logits, flattening the distribution toward uniform.
Temperature therefore directly controls entropy by scaling
logit ratios before normalization.
"""

# ============================================================
# Task 2 — Top-k (Naive)
# ============================================================

def top_k_sampling(logits, k, temperature=1.0):
    working = logits.copy()

    sorted_indices = np.argsort(working)
    top_k_indices = sorted_indices[-k:]

    mask = np.ones_like(working, dtype=bool)
    mask[top_k_indices] = False
    working[mask] = -np.inf

    return softmax_with_temperature(working, temperature)


# Required Tests
for k in [1, 2, 3, 5]:
    probs = top_k_sampling(logits, k)
    nonzero = [(vocabulary[i], f"{probs[i]:.4f}")
               for i in range(len(probs)) if probs[i] > 0]
    print(f"top_k={k}: {nonzero}")

# ============================================================
# Task 3 — Top-k (Efficient, REQUIRED)
# ============================================================

def top_k_sampling_efficient(logits, k, temperature=1.0):
    top_k_indices = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_indices]

    top_k_probs = softmax_with_temperature(top_k_logits, temperature)

    full_probs = np.zeros_like(logits)
    full_probs[top_k_indices] = top_k_probs

    return full_probs


# Verification
print("\nVerifying naive == efficient:")
for k in [1, 2, 3, 5, 10]:
    naive = top_k_sampling(logits, k)
    efficient = top_k_sampling_efficient(logits, k)
    assert np.allclose(naive, efficient)
print("All checks passed.")

# ============================================================
# Task 4 — Top-p Sampling
# ============================================================

def top_p_sampling(logits, p, temperature=1.0):

    base_probs = softmax_with_temperature(logits, 1.0)

    sorted_indices = np.argsort(base_probs)[::-1]
    sorted_probs = base_probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)

    cutoff = np.searchsorted(cumulative, p)
    nucleus_indices = sorted_indices[:cutoff + 1]

    working = logits.copy()
    mask = np.ones_like(working, dtype=bool)
    mask[nucleus_indices] = False
    working[mask] = -np.inf

    return softmax_with_temperature(working, temperature)


# Required Tests
for p in [0.5, 0.75, 0.9, 0.95, 1.0]:
    probs = top_p_sampling(logits, p)
    nonzero = [(vocabulary[i], f"{probs[i]:.4f}")
               for i in range(len(probs)) if probs[i] > 0]
    print(f"top_p={p}: {nonzero}")

# ============================================================
# Task 5 — Logit Bias
# ============================================================

def logit_bias_sampling(logits, bias_dict, temperature=1.0):
    working = logits.copy()
    for idx, bias in bias_dict.items():
        working[idx] += bias
    return softmax_with_temperature(working, temperature)


# Required Tests
print(logit_bias_sampling(logits, {7: 5.0})[7])
print(logit_bias_sampling(logits, {3: -100.0})[3])

# ============================================================
# Task 6 — Combined Sampler
# ============================================================

def sample_with_all_parameters(
        logits,
        temperature=1.0,
        top_k=None,
        top_p=None,
        logit_bias=None):

    working = logits.copy()

    # Step 1: Logit bias
    if logit_bias:
        for idx, bias in logit_bias.items():
            working[idx] += bias

    # Step 2: Top-k
    if top_k is not None:
        top_k_indices = np.argpartition(working, -top_k)[-top_k:]
        mask = np.ones_like(working, dtype=bool)
        mask[top_k_indices] = False
        working[mask] = -np.inf

    # Step 3: Top-p
    if top_p is not None:
        base_probs = softmax_with_temperature(working, 1.0)
        sorted_indices = np.argsort(base_probs)[::-1]
        sorted_probs = base_probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative, top_p)
        nucleus_indices = sorted_indices[:cutoff + 1]

        mask = np.ones_like(working, dtype=bool)
        mask[nucleus_indices] = False
        working[mask] = -np.inf

    # Step 4: Temperature softmax
    return softmax_with_temperature(working, temperature)


# ============================================================
# 6-Panel Visualization
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

configs = [
    ("Baseline (T=1.0)", softmax_with_temperature(logits, 1.0)),
    ("Low Temperature (T=0.2)", softmax_with_temperature(logits, 0.2)),
    ("Top-k=3 (T=1.0)", top_k_sampling_efficient(logits, 3, 1.0)),
    ("Top-p=0.8 (T=1.0)", top_p_sampling(logits, 0.8, 1.0)),
    ("Logit bias: ban 'on'", logit_bias_sampling(logits, {3: -100.0}, 1.0)),
    ("Combined: T=0.7, k=4, p=0.9, ban 'on'",
     sample_with_all_parameters(logits, 0.7, 4, 0.9, {3: -100.0}))
]

for ax, (title, probs) in zip(axes, configs):
    ax.bar(vocabulary, probs)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("sampling_comparison.png", dpi=150)
plt.show()

"""
In the 6-panel chart, we observe different sampling techniques used with different temperatures.

In the first two charts, we look at a baseline sampling technique using only temperature. 
Here, we see the effect of low versus high temperature. 
When **T = 1**, the logits remain unchanged after scaling. 
If the original logits are already close in value, the differences between them stay small. 
After applying softmax, the probabilities become more evenly distributed, leading to a more balanced choice for the next token.

When the temperature is low, dividing the logits by a small value increases the differences between them. 
This makes the highest logit stand out more, resulting in a more deterministic choice for the next token.

In the third chart, we observe top-k sampling with **temperature = 1**. 
Only the top 3 probabilities are considered. 
There is a clear favorite for the next token, but the probabilities are still relatively close, so it will not always be the “on” token that gets selected. 
The same idea applies to the fourth chart using top-p sampling.

In the fifth chart, we apply a negative logit bias to the token “on.” 
This removes its probability completely, so it is effectively never selected.

In the last chart, multiple sampling techniques are combined — top-k, top-p, and logit bias — resulting in a more clear and choice for the next token.

"""