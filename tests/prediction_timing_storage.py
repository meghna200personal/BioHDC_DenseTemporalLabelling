import numpy as np, time
import common_func as cf
from tabular_hdc_user import TabularHDCClassifier as T

rng = np.random.default_rng(0)
N, F, D, E = 2000, 32, 10000, 6
X = rng.normal(0, 1, (N, F)); y = (rng.random((N, E)) < 0.05).astype(int)
t = T(D=D, num_features=F); t.fit(X, y)

t0 = time.time(); t.predict_proba(X[:500]); el = time.time() - t0
print(f"tabular predict_proba: {el/500*1e3:.3f} ms per frame ({500/el:.0f} frames/s)")

# notebook prediction: one matrix product for all frames
Pn = np.where(rng.random((26, D)) > 0.5, 1, -1).astype(np.int8)   # unused leftover line
Hn = np.where(rng.random((500, D)) > 0.5, 1, -1).astype(np.int8)
t0 = time.time(); cf.evaluate_HDC(Hn, Pn, np.zeros(500, int)); el = time.time() - t0
print(f"notebook evaluate_HDC (26 classes): {el/500*1e3:.3f} ms per frame")

# stored model sizes at 32 channels, D=10000
mb = lambda n: n / 1e6
print("tabular stored: feature_IM %.2f MB (uint8), CIM %.2f MB, prototypes %.2f MB"
      % (mb(32*D), mb(21*D), mb(6*D)))
print("notebook stored, 617 features: IM %.2f MB (int8), CIM %.2f MB, AM (26 classes) %.2f MB"
      % (mb(617*D), mb(21*D), mb(26*D)))
print("per encoded frame, unpacked: %.0f KB; packed to bits: %.2f KB" % (D/1e3, D/8/1e3))