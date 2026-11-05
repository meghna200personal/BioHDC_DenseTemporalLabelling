
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

class TabularHDCClassifier:
    """
    Tabular Hyperdimensional Computing (HDC) Classifier using MAP/BSC Record Encoding.
    Processes each row as a static record of 32 channel features (ignoring temporal order).
    """
    #what is a good range of number of levels for quantization? 
    
    def __init__(self, D=10000, num_features=32, num_levels=21, seed=42):
        self.D = D
        self.num_features = num_features
        self.num_levels = num_levels
        np.random.seed(seed)

        # 1. Feature ID Item Memory (Orthogonal dense binary hypervectors for each channel column)
        self.feature_IM = np.random.randint(0, 2, size=(num_features, D), dtype=np.uint8)

        # 2. Continuous Item Memory (CIM) for quantized amplitude levels
        self.CIM = np.zeros((num_levels, D), dtype=np.uint8)
        base_vector = np.random.randint(0, 2, size=D, dtype=np.uint8)
        self.CIM[0] = base_vector

        bits_to_flip = np.random.permutation(D)
        step = D // (2 * (num_levels - 1))

        # Generate continuous similarity structure across adjacent voltage levels
        for m in range(1, num_levels):
            level_vector = self.CIM[m - 1].copy()
            flip_indices = bits_to_flip[(m - 1) * step : m * step]
            level_vector[flip_indices] = 1 - level_vector[flip_indices]
            self.CIM[m] = level_vector

    def _quantize(self, X, min_val=-4.0, max_val=4.0):
        """Quantizes continuous feature matrix into CIM level indices."""
        clipped = np.clip(X, min_val, max_val)
        normalized = (clipped - min_val) / (max_val - min_val)
        level_indices = np.floor(normalized * (self.num_levels - 1)).astype(int)
        return level_indices

    def encode_row(self, level_indices_row):
        """
        Encodes a single tabular row (record):
        H_t = threshold( sum( Feature_ID_i XOR Level_Vector_i ) )
        """
        level_vectors = self.CIM[level_indices_row]  # Shape: (32, D)
        bound_features = np.bitwise_xor(self.feature_IM, level_vectors)  # XOR binding
        summed = np.sum(bound_features, axis=0)
        row_hypervector = (summed > (self.num_features / 2)).astype(np.uint8)  # Majority bundling
        return row_hypervector

    def encode_dataset(self, X):
        """Encodes tabular dataset (N_samples, N_features) into hypervectors."""
        num_samples = X.shape[0]
        level_indices_mat = self._quantize(X)
        H_matrix = np.zeros((num_samples, self.D), dtype=np.uint8)

        for i in range(num_samples):
            H_matrix[i] = self.encode_row(level_indices_mat[i])

        return H_matrix

    def fit(self, X_train, y_train):
        """
        Trains multi-label event prototypes using bipolar accumulation (+1 / -1).
        """
        num_samples, num_events = y_train.shape
        H_train = self.encode_dataset(X_train)
        bipolar_H = np.where(H_train == 1, 1, -1)

        self.prototypes_accum = np.zeros((num_events, self.D), dtype=int)

        for k in range(num_events):
            for i in range(num_samples):
                if y_train[i, k] == 1:
                    self.prototypes_accum[k] += bipolar_H[i]
                else:
                    self.prototypes_accum[k] -= bipolar_H[i]

        # Threshold accumulator to produce final binary prototype hypervectors
        self.prototypes = (self.prototypes_accum > 0).astype(np.uint8)

    def predict_proba(self, X_test):
        """
        Computes per-event prediction probabilities using normalized Hamming similarity.
        """
        H_test = self.encode_dataset(X_test)
        num_samples = H_test.shape[0]
        num_events = self.prototypes.shape[0]
        probabilities = np.zeros((num_samples, num_events))

        for i in range(num_samples):
            for k in range(num_events):
                hamming_dist = np.sum(H_test[i] != self.prototypes[k])
                probabilities[i, k] = 1.0 - (hamming_dist / self.D)

        return probabilities

# =========================================================================
# EXAMPLE USAGE ON SYNTHETIC / REAL GAL TABULAR DATA
# =========================================================================
# if __name__ == "__main__":
#     np.random.seed(42)
#     num_samples = 2000
#     num_features = 32
#     num_events = 6

#     # Generate synthetic 32-channel tabular EEG data with 6 target event labels
#     X = np.random.normal(0, 1.0, size=(num_samples, num_features))
#     y = np.random.randint(0, 2, size=(num_samples, num_events))

#     # Train / Test Split (70% train, 30% test)
#     split = int(0.7 * num_samples)
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]

#     # Train Tabular HDC Classifier
#     hdc = TabularHDCClassifier(D=10000, num_features=num_features, num_levels=21)
#     hdc.fit(X_train, y_train)

#     # Predict Probabilities on Test Data
#     probs = hdc.predict_proba(X_test)

#     # Evaluate ROC-AUC and Average Precision (mAP)
#     cols = ["HandStart", "FirstDigitTouch", "BothStartLoadPhase", "LiftOff", "Replace", "BothReleased"]
#     print("========================================================================")
#     print("           TABULAR HDC CLASSIFICATION METRICS (GAL)                     ")
#     print("========================================================================")
    
#     auc_list, ap_list = [], []
#     for k in range(num_events):
#         auc = roc_auc_score(y_test[:, k], probs[:, k])
#         ap = average_precision_score(y_test[:, k], probs[:, k])
#         auc_list.append(auc)
#         ap_list.append(ap)
#         print(f"Event {k+1} ({cols[k]:< 20}): ROC-AUC = {auc:.4f} | Average Precision = {ap:.4f}")

#     print("------------------------------------------------------------------------")
#     print(f"Mean ROC-AUC across 6 events:            {np.mean(auc_list):.4f}")
#     print(f"Mean Average Precision (mAP) across 6 events: {np.mean(ap_list):.4f}")
#     print("========================================================================")
