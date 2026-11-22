import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

seed = 11
k = 15
m_values = [5, 10, 15, 20, 25]

D = np.load("D.npy")
n, d = D.shape
#assigns each row of X to its nearest centre in B and returns total sq dist
def compute_cost(X, B, batch_size=50000):
    B = np.asarray(B, dtype=np.float32)
    B_sq = np.sum(B**2, axis=1)
    total_cost = 0.0
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size]
        dist_sq = np.sum(batch**2, axis=1, keepdims=True) - 2 * (batch @ B.T) + B_sq
        total_cost += np.sum(np.min(dist_sq, axis=1))
    return float(total_cost)

#returns cluster centroids by averaging original data points in each cluster
def compute_cluster_means(X, labels, k):
    centres = np.zeros((k, X.shape[1]), dtype=np.float64)
    for j in range(k):
        points = X[labels == j]
        if len(points) > 0:
            centres[j] = points.mean(axis=0)
    return centres

#computes clustering cost costX(C)
def compute_partition_cost(X, labels, k):
    centres = compute_cluster_means(X, labels, k)
    total_cost = 0.0
    for j in range(k):
        points = X[labels == j]
        if len(points) > 0:
            total_cost += np.sum((points - centres[j]) ** 2)
    return float(total_cost)

#gaussian matrix with N(0, 1/m) entries
def draw_random_matrix(m, d, random_seed):
    rng = np.random.default_rng(random_seed)
    return rng.normal(0.0, 1.0 / np.sqrt(m), size=(m, d)).astype(np.float32)

start_time = time.perf_counter()
kmeans_D = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(D)
time_D = time.perf_counter() - start_time
B = kmeans_D.cluster_centers_
labels_D = kmeans_D.labels_
cost_D_B = compute_cost(D, B)
print("(a) Baseline k-means on full data D")
print(f"cost(D, B): {cost_D_B:,.4f}")
print(f"Wall-clock time: {time_D:.2f} seconds\n")

records = []
for m in m_values:
    for rep in range(5):
        A = draw_random_matrix(m, d, [seed, m, rep])
        start = time.perf_counter()
        E = D @ A.T
        kmeans_E = KMeans(n_clusters=k, n_init=5, random_state=seed + rep).fit(E)

        time_E = time.perf_counter() - start
        B_hat = compute_cluster_means(D, kmeans_E.labels_, k)
        cost_hat = compute_cost(D, B_hat)
        ratio_hat = cost_hat / cost_D_B
        
        B_plus = kmeans_E.cluster_centers_ @ np.linalg.pinv(A).T
        cost_plus = compute_cost(D, B_plus)
        ratio_plus = cost_plus / cost_D_B
        
        records.append({
            "m": m,
            "rep": rep,
            "cost_hat": cost_hat,
            "ratio_hat": ratio_hat,
            "cost_plus": cost_plus,
            "ratio_plus": ratio_plus,
            "time": time_E
        })

records_df = pd.DataFrame(records)
records_df.to_csv("q1_runs.csv", index=False)

# Part (b)cost(D, B^) and ratio cost(D, B^)/cost(D, B)
table_b = records_df.groupby("m").agg(
    mean_cost=("cost_hat", "mean"),
    mean_ratio=("ratio_hat", "mean"),
    min_ratio=("ratio_hat", "min"),
    max_ratio=("ratio_hat", "max")
).rename(columns={
    "mean_cost": "Mean cost(D, B^)",
    "mean_ratio": "Mean Ratio",
    "min_ratio": "Smallest Ratio",
    "max_ratio": "Largest Ratio"
})
print("(b) k-means with B^ ")
print(table_b.round(4).to_string())
print()

# Part (c) Table: B+ ratio reported next to B^ ratio
table_c = records_df.groupby("m").agg(
    b_hat_mean_ratio=("ratio_hat", "mean"),
    b_plus_mean_ratio=("ratio_plus", "mean"),
    b_plus_min_ratio=("ratio_plus", "min"),
    b_plus_max_ratio=("ratio_plus", "max")
).rename(columns={
    "b_hat_mean_ratio": "B^ Mean Ratio",
    "b_plus_mean_ratio": "B+ Mean Ratio",
    "b_plus_min_ratio": "B+ Smallest Ratio",
    "b_plus_max_ratio": "B+ Largest Ratio"
})

print("(c) Comparison: B^ vs B+ (pseudo-inverse centres)")
print(table_c.round(4).to_string())
print()

# Part (d): Wall-clock times vs baseline
table_d = records_df.groupby("m").agg(
    mean_time=("time", "mean")
).rename(columns={"mean_time": "Mean Sketched Time (s)"})
table_d["Baseline Time (s)"] = time_D
table_d["Speedup"] = (time_D / table_d["Mean Sketched Time (s)"]).map(lambda s: f"{s:.2f}x")
print("(d) Wall-clock times: Sketched vs Baseline (a)")
print(table_d.round(4).to_string())
print()

summary = records_df.groupby("m").agg(
    ratio_hat_mean=("ratio_hat", "mean"),
    ratio_hat_min=("ratio_hat", "min"),
    ratio_hat_max=("ratio_hat", "max"),
    ratio_plus_mean=("ratio_plus", "mean"),
    ratio_plus_min=("ratio_plus", "min"),
    ratio_plus_max=("ratio_plus", "max"),
    time_mean=("time", "mean")
)
summary.to_csv("q1_summary.csv")

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(m_values))
bar_width = 0.35

yerr_hat = [
    summary["ratio_hat_mean"] - summary["ratio_hat_min"],
    summary["ratio_hat_max"] - summary["ratio_hat_mean"],
]
ax.bar(x - bar_width / 2, summary["ratio_hat_mean"], bar_width, yerr=yerr_hat, capsize=4, label="B^")

yerr_plus = [
    summary["ratio_plus_mean"] - summary["ratio_plus_min"],
    summary["ratio_plus_max"] - summary["ratio_plus_mean"],
]
ax.bar(x + bar_width / 2, summary["ratio_plus_mean"], bar_width, yerr=yerr_plus, capsize=4, label="B+")

ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(m_values)
ax.set_xlabel("m")
ax.set_ylabel("Cost Ratio")
ax.set_title("k-means Cost Ratio: Comparison of B^ and B+")
ax.legend()
plt.tight_layout()
plt.savefig("q1_bc_ratio.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(m_values, summary["time_mean"], "o-", label="Sketched k-means (E)")
ax.axhline(time_D, color="red", linestyle="--", label="Full k-means (D)")
ax.set_xlabel("m")
ax.set_ylabel("Wall-clock time (seconds)")
ax.set_title("Runtime Comparison vs m")
ax.legend()
plt.tight_layout()
plt.savefig("q1_d_time.png", dpi=150)
plt.close()

rng = np.random.default_rng(seed + 1)
pairs_i, pairs_l = [], []
while len(pairs_i) < 10000:
    sample_i = rng.integers(0, n, size=20000)
    sample_l = rng.integers(0, n, size=20000)
    valid = np.sum((D[sample_i] - D[sample_l]) ** 2, axis=1) > 0
    pairs_i.extend(sample_i[valid])
    pairs_l.extend(sample_l[valid])

idx_i = np.array(pairs_i[:10000])
idx_l = np.array(pairs_l[:10000])
dist_D_sq = np.sum((D[idx_i] - D[idx_l]) ** 2, axis=1)

cost_D_CD = compute_partition_cost(D, labels_D, k)

pair_ratios = []
cost_ratios = []
for m in m_values:
    A = draw_random_matrix(m, d, [seed, m, 999])
    E = D @ A.T
    dist_E_sq = np.sum((E[idx_i] - E[idx_l]) ** 2, axis=1)
    pair_ratios.append(dist_E_sq / dist_D_sq)

    ratios_m = []
    for rep in range(20):
        A = draw_random_matrix(m, d, [seed, m, 1000 + rep])
        E = D @ A.T
        ratios_m.append(compute_partition_cost(E, labels_D, k) / cost_D_CD)
    cost_ratios.append(np.array(ratios_m))

fig, ax = plt.subplots(figsize=(8, 4.5))
positions = np.arange(len(m_values)) * 2.0

bp1 = ax.boxplot(pair_ratios, positions=positions - 0.4, widths=0.6, patch_artist=True)
bp2 = ax.boxplot(cost_ratios, positions=positions + 0.4, widths=0.6, patch_artist=True)

for box in bp1["boxes"]:
    box.set_facecolor("tab:blue")
for box in bp2["boxes"]:
    box.set_facecolor("tab:orange")

ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
ax.set_xticks(positions)
ax.set_xticklabels(m_values)
ax.set_xlabel("m")
ax.set_ylabel("Ratio")
ax.set_title("Distance Distortion vs Clustering Cost Distortion")
ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
          ["Pairwise distance ratio", "Clustering cost ratio"])
plt.tight_layout()
plt.savefig("q1_e_boxplots.png", dpi=150)
plt.close()