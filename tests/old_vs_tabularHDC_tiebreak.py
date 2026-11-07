import numpy as np, time, math
import common_func as cf
from tabular_hdc_user import TabularHDCClassifier as T

# 1. ties in bundling: a bit is a tie when exactly half of the bound bits are 1
for F in (32, 617):
    p_tie = math.comb(F, F//2) / 2**F if F % 2 == 0 else 0.0
    print(f"F={F}: P(tie per bit) = {p_tie:.3f}  -> expected share of 1-bits if ties go to 0: {(1-p_tie)/2:.3f}")

rng = np.random.default_rng(0)
N, F, D ,E = 2000, 32, 10000,6
X = rng.normal(0, 1, (N, F))
y = (rng.random((N, E)) < 0.05).astype(int)
t = T(D=D, num_features=F)
H = t.encode_dataset(X)
print("tabular encoded 1-bit share (measured):", round(H.mean(), 4))

# 2. same data through the notebook encoder (bipolar, data-driven bins)
mins, maxs = X.min(0), X.max(0)
edges = [mins]
for _ in range(20):
    edges.append(edges[-1] + (maxs - mins) / 20)
edges = np.array(edges)
mids = np.transpose(edges[:-1] + ((maxs - mins) / 20) / 2)
cim = cf.generate_CIM(21, D); im = cf.generate_IM(F, D)
Hn = cf.encode_as_HVs_record_based(X, cim, im, mids)
print("notebook encoded +1-bit share (measured):", round((Hn == 1).mean(), 4))

# 3. timing of one encoding pass
t0 = time.time(); t.encode_dataset(X); t_tab = time.time() - t0
t0 = time.time(); cf.encode_as_HVs_record_based(X, cim, im, mids); t_nb = time.time() - t0
print(f"encode {N} frames (F={F}): tabular {t_tab:.2f}s, notebook {t_nb:.2f}s")

# 4. fit BEFORE timing prediction (this was the missing step)
t.fit(X, y)
print("prototypes shape:", t.prototypes.shape)
t0 = time.time(); t.predict_proba(X[:500]); el = time.time() - t0
print(f"tabular predict_proba: {el/500*1e3:.3f} ms per frame")