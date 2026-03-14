import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

data = np.load("Features.npz")
Features = data["Features"]

print(Features.shape)
Features = Features[:, :100, :]
print(Features.shape)

min_val = Features.min()
max_val = Features.max()

Features = (Features - min_val) / (max_val - min_val)

def matcher(v1, v2):
    distance = np.linalg.norm(v1 - v2)
    score = 1 / (1 + distance)
    return score
genuine_scores = []

for person in range(100):

    for t1, t2 in combinations(range(10), 2):

        v1 = Features[t1, person]
        v2 = Features[t2, person]

        score = matcher(v1, v2)
        genuine_scores.append(score)

genuine_scores = np.array(genuine_scores)
print(len(genuine_scores))

imposter_scores = []

for t in range(10):

    for p1, p2 in combinations(range(100), 2):

        v1 = Features[t, p1]
        v2 = Features[t, p2]

        score = matcher(v1, v2)
        imposter_scores.append(score)

imposter_scores = np.array(imposter_scores)

print(len(imposter_scores))

plt.hist(genuine_scores, bins=50, alpha=0.6, label="Genuine")
plt.hist(imposter_scores, bins=50, alpha=0.6, label="Imposter")

plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Score Distribution")
plt.legend()

plt.show()

thresholds = np.linspace(0, 1, 500)

FAR = []
FRR = []

for t in thresholds:

    far = np.sum(imposter_scores >= t) / len(imposter_scores)
    frr = np.sum(genuine_scores < t) / len(genuine_scores)

    FAR.append(far)
    FRR.append(frr)

FAR = np.array(FAR)
FRR = np.array(FRR)

diff = np.abs(FAR - FRR)
eer_index = np.argmin(diff)

EER = FAR[eer_index]
eer_threshold = thresholds[eer_index]

print("EER:", EER)
print("EER Threshold:", eer_threshold)

plt.figure()

plt.plot(thresholds, FAR, label="FAR")
plt.plot(thresholds, FRR, label="FRR")

plt.scatter(eer_threshold, EER, color="red", label="EER")

plt.xlabel("Threshold")
plt.ylabel("Error Rate")
plt.title("FAR - FRR vs Threshold")

plt.legend()

plt.show()


plt.figure()

plt.plot(FAR, FRR)

plt.xlabel("FAR")
plt.ylabel("FRR")
plt.title("ROC Curve")

plt.show()

plt.figure()

plt.hist(genuine_scores, bins=50, alpha=0.5, label="Genuine")
plt.hist(imposter_scores, bins=50, alpha=0.5, label="Imposter")

plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Score Distribution")

plt.legend()

plt.show()