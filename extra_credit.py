import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# Load JSON Results
# ===============================

files = glob.glob("results_*.json")

if len(files) == 0:
    raise FileNotFoundError("No results_*.json files found in this directory.")

data = {}

for file in files:
    with open(file, "r") as f:
        content = json.load(f)
        T = float(content["temperature"])
        if T not in data:
            data[T] = []
        data[T].extend(content["responses"])

temperatures = sorted(data.keys())

print("Temperatures found:", temperatures)
print()

# ===============================
# EXTRA CREDIT 1
# Response Similarity Heatmap
# ===============================

print("Computing similarity matrix...")

n = len(temperatures)
similarity_matrix = np.zeros((n, n))

for i, T1 in enumerate(temperatures):
    for j, T2 in enumerate(temperatures):

        texts = data[T1] + data[T2]

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(texts)

        sim_matrix = cosine_similarity(tfidf)

        n1 = len(data[T1])

        # Similarity between group T1 and T2
        cross_sim = sim_matrix[:n1, n1:]

        similarity_matrix[i, j] = cross_sim.mean()

# Plot Heatmap
plt.figure(figsize=(6, 6))
plt.imshow(similarity_matrix)
plt.colorbar(label="Average Cosine Similarity")
plt.xticks(range(n), temperatures)
plt.yticks(range(n), temperatures)
plt.xlabel("Temperature")
plt.ylabel("Temperature")
plt.title("Response Similarity Across Temperatures")

plt.tight_layout()
plt.savefig("temperature_similarity.png", dpi=150)
plt.show()

print("Saved: temperature_similarity.png")
print()

# ===============================
# EXTRA CREDIT 2
# Response Length Distribution
# ===============================

print("Computing response length statistics...")

lengths_by_temp = []

for T in temperatures:
    lengths = [len(response.split()) for response in data[T]]
    lengths_by_temp.append(lengths)

    print(f"Temperature {T}:")
    print(f"  Mean length: {np.mean(lengths):.2f}")
    print(f"  Std dev:     {np.std(lengths):.2f}")
    print(f"  Min length:  {np.min(lengths)}")
    print(f"  Max length:  {np.max(lengths)}")
    print()

# Boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(lengths_by_temp, labels=temperatures)
plt.xlabel("Temperature")
plt.ylabel("Response Length (words)")
plt.title("Response Length Distribution by Temperature")

plt.tight_layout()
plt.savefig("length_distribution.png", dpi=150)
plt.show()

print("Saved: length_distribution.png")
print()
print("Extra credit analysis complete.")