import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99

#fetching data
kdd = fetch_kddcup99(subset=None, as_frame=True)
X = kdd.data.copy()
text_features = ["protocol_type", "service", "flag"]
# Text features arrive as bytes so we have to decode them to strings
for c in text_features:
    X[c] = X[c].str.decode("utf-8")
numeric_features = [c for c in X.columns if c not in text_features]
X_num = X[numeric_features].astype(float)
#onehot encoding text features
X_text = pd.get_dummies(X[text_features], dtype=float)
D = pd.concat([X_num, X_text], axis=1)
#droping constant columns
const_cols = D.columns[D.nunique() == 1]
D = D.drop(columns=const_cols)
#min max scaling
min_val, max_val = D.min(), D.max()
D = (D - min_val)/(max_val - min_val)
#final matrix in float32
D = D.to_numpy(dtype=np.float32)
n, d = D.shape
np.save("D.npy", D)
#for q2
y = (kdd.target != b"normal.").to_numpy(dtype=float)
np.save("y.npy", y)

# stats
print("Dropped constant columns:", list(const_cols))
print(f"final matrix rows(n) = {n}, columns(d) = {d}")
print("attack fraction:", y.mean())